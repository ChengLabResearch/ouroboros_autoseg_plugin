use std::{fs, path::Path};

use super::AppConfig;

fn config_at(root: &Path) -> AppConfig {
    let mut config = AppConfig::from_env().unwrap();
    config.checkpoint_dir = root.join(
        config
            .checkpoint_dir
            .strip_prefix(&config.internal_volume_path)
            .unwrap(),
    );
    config.internal_volume_path = root.to_path_buf();
    config
}

#[test]
fn checkpoint_path_uses_persistent_plugin_storage() {
    let config = AppConfig::from_env().unwrap();
    for model in ["sam3", "medical_sam3"] {
        assert_eq!(
            config.checkpoint_path(model),
            config
                .internal_volume_path
                .join("persist/sam3-segmentation/chkpts")
                .join(format!("{model}.pt"))
        );
    }
    assert_eq!(
        config.plugin_root(),
        config.internal_volume_path.join("sam3-segmentation"),
        "scratch storage must stay outside persist"
    );
}

#[test]
fn migration_leaves_missing_or_incomplete_checkpoints_uninstalled() {
    let root = tempfile::tempdir().unwrap();
    let config = config_at(root.path());
    config.migrate_legacy_checkpoints().unwrap();
    assert!(!config.checkpoint_dir.exists());

    let legacy = config.plugin_root().join("chkpts");
    fs::create_dir_all(legacy.join("directory.pt")).unwrap();
    fs::write(legacy.join("sam3.pt.part"), b"incomplete").unwrap();
    fs::write(legacy.join("notes.txt"), b"not a checkpoint").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("notes.txt", legacy.join("symlink.pt")).unwrap();

    config.migrate_legacy_checkpoints().unwrap();
    assert!(!config.checkpoint_dir.exists());
    assert_eq!(
        fs::read(legacy.join("sam3.pt.part")).unwrap(),
        b"incomplete"
    );
    assert!(legacy.join("directory.pt").is_dir());
}

#[test]
fn migration_preserves_destinations_and_moves_each_other_completed_file_once() {
    let root = tempfile::tempdir().unwrap();
    let config = config_at(root.path());
    let legacy = config.plugin_root().join("chkpts");
    fs::create_dir_all(&legacy).unwrap();
    config.ensure_checkpoint_dir().unwrap();
    fs::write(legacy.join("sam3.pt"), b"old weights").unwrap();
    fs::write(config.checkpoint_path("sam3"), b"installed weights").unwrap();
    // Migration covers completed .pt files, not just the tracked model catalog.
    fs::write(legacy.join("other.pt"), b"other weights").unwrap();
    #[cfg(unix)]
    let old_inode = {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(legacy.join("other.pt")).unwrap().ino()
    };

    for _ in 0..2 {
        config.migrate_legacy_checkpoints().unwrap();
        assert_eq!(
            fs::read(config.checkpoint_path("sam3")).unwrap(),
            b"installed weights"
        );
        assert_eq!(fs::read(legacy.join("sam3.pt")).unwrap(), b"old weights");
        assert_eq!(
            fs::read(config.checkpoint_path("other")).unwrap(),
            b"other weights"
        );
        assert!(!legacy.join("other.pt").exists());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            fs::metadata(config.checkpoint_path("other")).unwrap().ino(),
            old_inode
        );
    }
}

#[test]
fn migration_failure_keeps_legacy_bytes_and_identifies_both_locations() {
    let root = tempfile::tempdir().unwrap();
    let config = config_at(root.path());
    let legacy = config.plugin_root().join("chkpts");
    fs::create_dir_all(&legacy).unwrap();
    fs::write(legacy.join("medical_sam3.pt"), b"only copy").unwrap();
    // A file blocks creation of the persistent directory on every test platform.
    fs::write(root.path().join("persist"), b"obstruction").unwrap();

    let error = config.migrate_legacy_checkpoints().unwrap_err().to_string();
    assert!(error.contains(&legacy.display().to_string()));
    assert!(error.contains(&config.checkpoint_dir.display().to_string()));
    assert_eq!(
        fs::read(legacy.join("medical_sam3.pt")).unwrap(),
        b"only copy"
    );
}
