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
