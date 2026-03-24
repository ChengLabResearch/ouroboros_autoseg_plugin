use crate::error::AppError;

#[derive(Debug, Clone)]
pub struct MixedPathParts {
    pub raw: String,
    pub file_name: String,
    pub stem: String,
}

pub fn parse_mixed_path(raw: &str) -> Result<MixedPathParts, AppError> {
    let cleaned = raw.trim().trim_matches('"').trim_matches('\'').to_string();
    if cleaned.is_empty() {
        return Err(AppError::bad_request("Path must not be empty"));
    }

    let separators: &[char] = &['/', '\\'];
    let file_name = cleaned
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

    Ok(MixedPathParts {
        raw: cleaned,
        file_name,
        stem,
    })
}
