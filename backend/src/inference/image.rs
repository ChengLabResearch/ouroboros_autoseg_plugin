use async_trait::async_trait;

use crate::{
    error::AppError,
    imaging::{annotations::AnnotationOrigin, tiff_io::ImageFrame},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointPromptKind {
    Positive(AnnotationOrigin),
    Negative,
}

impl PointPromptKind {
    /// Convert polarity to the numeric label expected by SAM prompt adapters.
    pub fn label(self) -> u32 {
        match self {
            Self::Positive(_) => 1,
            Self::Negative => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointPrompt {
    pub x: f32,
    pub y: f32,
    pub kind: PointPromptKind,
}

impl PointPrompt {
    pub fn positive(x: f32, y: f32, origin: AnnotationOrigin) -> Self {
        Self {
            x,
            y,
            kind: PointPromptKind::Positive(origin),
        }
    }

    pub fn negative(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            kind: PointPromptKind::Negative,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameMask {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

#[async_trait]
pub trait ImageSegmenter: Send + Sync {
    async fn segment(
        &self,
        frame: &ImageFrame,
        prompts: &[PointPrompt],
    ) -> Result<FrameMask, AppError>;
}
