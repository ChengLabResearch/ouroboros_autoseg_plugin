use crate::error::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFlavor {
    Posix,
    Windows,
}

#[derive(Debug, Clone)]
pub struct MixedPathParts {
    pub raw: String,
    pub file_name: String,
    pub stem: String,
    pub parent: Option<String>,
    pub flavor: PathFlavor,
}

pub fn parse_mixed_path(raw: &str) -> Result<MixedPathParts, AppError> {
    let cleaned = raw.trim().trim_matches('"').trim_matches('\'').to_string();
    if cleaned.is_empty() {
        return Err(AppError::bad_request("Path must not be empty"));
    }

    let flavor = detect_path_flavor(&cleaned);
    let separators: &[char] = match flavor {
        PathFlavor::Posix => &['/'],
        PathFlavor::Windows => &['/', '\\'],
    };
    let normalized = cleaned.trim_end_matches(separators);
    let normalized = if normalized.is_empty() {
        cleaned.as_str()
    } else {
        normalized
    };

    let file_name = cleaned
        .trim_end_matches(separators)
        .rsplit(separators)
        .find(|segment| !segment.is_empty())
        .ok_or_else(|| AppError::bad_request("Path must include a file or directory name"))?
        .to_string();

    let stem = file_name
        .rsplit_once('.')
        .map(|(left, _)| left)
        .filter(|value| !value.is_empty())
        .unwrap_or(file_name.as_str())
        .to_string();

    let parent = normalized
        .rsplit_once(separators)
        .map(|(left, _)| left)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    Ok(MixedPathParts {
        raw: cleaned,
        file_name,
        stem,
        parent,
        flavor,
    })
}

fn detect_path_flavor(path: &str) -> PathFlavor {
    if path.starts_with("\\\\")
        || path.contains('\\')
        || path.as_bytes().get(1).is_some_and(|byte| *byte == b':')
    {
        PathFlavor::Windows
    } else {
        PathFlavor::Posix
    }
}

#[cfg(test)]
mod tests;
