use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock: toml::Table = fs::read_to_string("Cargo.lock")?.parse()?;
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .ok_or("Cargo.lock has no package list")?;
    let mut engines = packages
        .iter()
        .filter(|package| package.get("name").and_then(toml::Value::as_str) == Some("candle-sam3"));
    let engine = engines
        .next()
        .ok_or("Cargo.lock has no candle-sam3 package")?;
    if engines.next().is_some() {
        return Err(
            "Cargo.lock has multiple candle-sam3 packages; engine revision is ambiguous".into(),
        );
    }

    let revision = match engine.get("source") {
        // A documented local [patch] has no Git provenance in the lockfile.
        None => "local",
        Some(source) => source
            .as_str()
            .filter(|source| source.starts_with("git+"))
            .and_then(|source| source.rsplit_once('#').map(|(_, revision)| revision))
            .filter(|revision| {
                revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
            })
            .ok_or("Cargo.lock candle-sam3 source has no resolved Git commit")?,
    };
    println!("cargo:rustc-env=CANDLE_SAM3_GIT_SHA={revision}");
    Ok(())
}
