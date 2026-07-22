use std::{fs, path::PathBuf};

use image::{codecs::jpeg::JpegEncoder, ExtendedColorType};

use crate::{
    error::AppError,
    imaging::tiff_io::{write_tiff_pages, ImageFrame, WritableTiffPage},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedFrameEncoding {
    Tiff,
    Jpeg,
}

#[derive(Debug, Clone)]
pub struct PreparedFrame {
    pub frame: ImageFrame,
    pub target: PathBuf,
    pub encoding: PreparedFrameEncoding,
}

pub async fn stage_input_frames(frames: &[PreparedFrame]) -> Result<(), AppError> {
    stage_input_frames_with_progress(frames, |_, _| {}).await
}

pub async fn stage_input_frames_with_progress<F>(
    frames: &[PreparedFrame],
    mut on_frame: F,
) -> Result<(), AppError>
where
    F: FnMut(usize, usize),
{
    for (frame_index, prepared) in frames.iter().enumerate() {
        if let Some(parent) = prepared.target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        match prepared.encoding {
            PreparedFrameEncoding::Tiff => stage_tiff_frame(prepared)?,
            PreparedFrameEncoding::Jpeg => stage_jpeg_frame(prepared)?,
        }
        on_frame(frame_index, frames.len());
    }

    Ok(())
}

fn stage_tiff_frame(prepared: &PreparedFrame) -> Result<(), AppError> {
    let rgb_pixels = rgb_pixels(&prepared.frame)?;
    write_tiff_pages(
        &prepared.target.with_extension("tif"),
        &[WritableTiffPage {
            width: prepared.frame.width,
            height: prepared.frame.height,
            channels: 3,
            pixels: rgb_pixels,
            description: None,
        }],
    )
}

fn stage_jpeg_frame(prepared: &PreparedFrame) -> Result<(), AppError> {
    let rgb_pixels = rgb_pixels(&prepared.frame)?;
    let file = fs::File::create(prepared.target.with_extension("jpg"))?;
    let mut encoder = JpegEncoder::new_with_quality(file, 90);
    encoder
        .encode(
            &rgb_pixels,
            prepared.frame.width as u32,
            prepared.frame.height as u32,
            ExtendedColorType::Rgb8,
        )
        .map_err(|error| AppError::upstream(format!("JPEG encoding failed: {error}")))?;
    Ok(())
}

fn rgb_pixels(frame: &ImageFrame) -> Result<Vec<u8>, AppError> {
    if frame.width == 0 || frame.height == 0 || frame.channels == 0 {
        return Err(AppError::bad_request(
            "Prepared frames must have non-zero dimensions and channels",
        ));
    }

    let expected_len = frame
        .width
        .checked_mul(frame.height)
        .and_then(|value| value.checked_mul(frame.channels))
        .ok_or_else(|| AppError::bad_request("Prepared frame dimensions are too large"))?;
    if frame.pixels.len() != expected_len {
        return Err(AppError::bad_request(
            "Prepared frame pixels do not match the declared geometry",
        ));
    }

    if frame.channels == 3 {
        return Ok(frame.pixels.clone());
    }

    // Match the current Python preprocessing shape handling: grayscale is expanded,
    // two-channel inputs are padded, and larger channel sets are truncated to RGB.
    let mut rgb = Vec::with_capacity(frame.width * frame.height * 3);
    for pixel in frame.pixels.chunks_exact(frame.channels) {
        match frame.channels {
            1 => rgb.extend_from_slice(&[pixel[0], pixel[0], pixel[0]]),
            2 => rgb.extend_from_slice(&[pixel[0], pixel[1], pixel[1]]),
            _ => rgb.extend_from_slice(&pixel[..3]),
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests;
