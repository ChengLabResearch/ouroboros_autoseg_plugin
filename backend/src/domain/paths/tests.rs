use super::{parse_mixed_path, PathFlavor};

#[test]
fn parses_posix_paths() {
    let parsed = parse_mixed_path(" '/tmp/data/stack.tif' ").expect("path should parse");
    assert_eq!(parsed.raw, "/tmp/data/stack.tif");
    assert_eq!(parsed.file_name, "stack.tif");
    assert_eq!(parsed.stem, "stack");
    assert_eq!(parsed.parent.as_deref(), Some("/tmp/data"));
    assert_eq!(parsed.flavor, PathFlavor::Posix);
}

#[test]
fn parses_windows_paths() {
    let parsed = parse_mixed_path(r#""C:\Users\dev\stack.tif""#).expect("path should parse");
    assert_eq!(parsed.raw, r#"C:\Users\dev\stack.tif"#);
    assert_eq!(parsed.file_name, "stack.tif");
    assert_eq!(parsed.stem, "stack");
    assert_eq!(parsed.parent.as_deref(), Some(r#"C:\Users\dev"#));
    assert_eq!(parsed.flavor, PathFlavor::Windows);
}

#[test]
fn parses_unc_directory_paths() {
    let parsed = parse_mixed_path(r#"\\server\share\folder\"#).expect("path should parse");
    assert_eq!(parsed.file_name, "folder");
    assert_eq!(parsed.stem, "folder");
    assert_eq!(parsed.parent.as_deref(), Some(r#"\\server\share"#));
    assert_eq!(parsed.flavor, PathFlavor::Windows);
}

#[test]
fn rejects_empty_paths() {
    let error = parse_mixed_path(" '' ").expect_err("empty path should fail");
    assert_eq!(error.to_string(), "Path must not be empty");
}

#[test]
fn replaces_file_names_using_the_original_host_path_flavor() {
    for (original, expected) in [
        ("/data.v1/stack.tif", "/data.v1/overlay.tif"),
        ("/stack.tif", "/overlay.tif"),
        ("stack.tif", "overlay.tif"),
        (r"C:\data.v1\stack.tif", r"C:\data.v1\overlay.tif"),
        (r"C:\stack.tif", r"C:\overlay.tif"),
        (r"C:\data/stack.tif", r"C:\data/overlay.tif"),
        (r"\\server\share\stack.tif", r"\\server\share\overlay.tif"),
    ] {
        let parsed = parse_mixed_path(original).expect("path");
        assert_eq!(parsed.with_file_name("overlay.tif"), expected);
    }
}
