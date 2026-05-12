use candle_core::{Device, Tensor};
use candle_nn::ops;
use candle_transformers::models::sam3;
use image::GenericImageView;

use crate::{
    error::AppError,
    imaging::tiff_io::ImageFrame,
    inference::image::{FrameMask, PositivePointPrompt},
};

/// Convert an `ImageFrame` (u8 pixels, HWC layout) to a float CHW tensor in [0, 1].
/// Single-channel frames are replicated to three channels for SAM3's RGB encoder.
pub fn frame_to_chw_tensor(frame: &ImageFrame, device: &Device) -> Result<Tensor, AppError> {
    let hw = frame.height * frame.width;
    let channels = frame.channels;
    let mut chw: Vec<f32> = vec![0.0; channels * hw];
    for (idx, &pixel) in frame.pixels.iter().enumerate() {
        let pixel_idx = idx / channels;
        let channel_idx = idx % channels;
        chw[channel_idx * hw + pixel_idx] = pixel as f32 / 255.0;
    }
    let chw = if channels == 1 {
        let mut rgb = vec![0.0f32; 3 * hw];
        for i in 0..hw {
            let v = chw[i];
            rgb[i] = v;
            rgb[hw + i] = v;
            rgb[2 * hw + i] = v;
        }
        rgb
    } else {
        chw
    };
    Tensor::from_vec(chw, (3, frame.height, frame.width), device)
        .map_err(|e| AppError::internal(e.to_string()))
}

/// Resize a BCHW float tensor to `target_size`×`target_size` and apply per-channel
/// mean/std normalization as expected by the SAM3 encoder.
pub fn normalize_for_sam3(
    image_bchw: &Tensor,
    target_size: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor, AppError> {
    let resized = image_bchw
        .upsample_bilinear2d(target_size, target_size, false)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let mean_t = Tensor::from_vec(mean.to_vec(), (1, 3, 1, 1), device)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let std_t = Tensor::from_vec(std.to_vec(), (1, 3, 1, 1), device)
        .map_err(|e| AppError::internal(e.to_string()))?;
    resized
        .broadcast_sub(&mean_t)
        .and_then(|t| t.broadcast_div(&std_t))
        .map_err(|e| AppError::internal(e.to_string()))
}

/// Build a SAM3 `GeometryPrompt` from a list of positive click prompts.
/// Pixel coordinates are normalised to [0, 1] by dividing by frame dimensions.
pub fn build_geometry_prompt(
    prompts: &[PositivePointPrompt],
    frame_width: usize,
    frame_height: usize,
    device: &Device,
) -> Result<sam3::GeometryPrompt, AppError> {
    if prompts.is_empty() {
        return Ok(sam3::GeometryPrompt::default());
    }
    let w = frame_width as f32;
    let h = frame_height as f32;
    let coords: Vec<f32> = prompts
        .iter()
        .flat_map(|p| [(p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)])
        .collect();
    let labels: Vec<u32> = vec![1; prompts.len()];
    let n = prompts.len();
    let points_xy = Tensor::from_vec(coords, (1, n, 2), device)
        .map_err(|e| AppError::internal(e.to_string()))?;
    let point_labels = Tensor::from_vec(labels, (1, n), device)
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(sam3::GeometryPrompt {
        points_xy: Some(points_xy),
        point_labels: Some(point_labels),
        ..Default::default()
    })
}

/// Threshold raw mask logits (shape [batch, query, H, W]) to a binary `FrameMask`.
///
/// The best query (index 0) is selected, upsampled via bilinear interpolation to
/// `model_size`×`model_size`, sigmoid-activated, then resampled to the target output
/// dimensions. Pixels where probability > 0.5 are set to 255, others to 0.
pub fn threshold_mask_logits_to_frame(
    mask_logits: &Tensor,
    out_width: usize,
    out_height: usize,
    model_size: usize,
) -> Result<FrameMask, AppError> {
    let logits_2d = mask_logits
        .i((0, 0))
        .map_err(|e| AppError::internal(e.to_string()))?;
    let probs = logits_2d
        .unsqueeze(0)
        .and_then(|t| t.unsqueeze(0))
        .and_then(|t| t.upsample_bilinear2d(model_size, model_size, false))
        .and_then(|t| ops::sigmoid(&t))
        .and_then(|t| t.upsample_bilinear2d(out_height, out_width, false))
        .and_then(|t| t.i((0, 0)))
        .map_err(|e| AppError::internal(e.to_string()))?;
    let rows = probs
        .to_vec2::<f32>()
        .map_err(|e| AppError::internal(e.to_string()))?;
    let pixels: Vec<u8> = rows
        .iter()
        .flat_map(|row| row.iter().map(|&v| if v > 0.5 { 255 } else { 0 }))
        .collect();
    Ok(FrameMask {
        width: out_width,
        height: out_height,
        pixels,
    })
}

/// Read the pixel dimensions of the first sorted image file in `frames_dir`.
/// Used by the video segmenter to normalise pixel-coord annotation points.
pub fn first_frame_dimensions(frames_dir: &std::path::Path) -> Result<(usize, usize), AppError> {
    let mut entries: Vec<_> = std::fs::read_dir(frames_dir)
        .map_err(|e| AppError::internal(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext, "jpg" | "jpeg" | "png" | "tif" | "tiff"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());
    let first = entries
        .first()
        .ok_or_else(|| AppError::bad_request("frames_dir contains no image files"))?;
    let img = image::ImageReader::open(first.path())
        .map_err(|e| AppError::internal(e.to_string()))?
        .decode()
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok((img.width() as usize, img.height() as usize))
}

/// Extract a binary `FrameMask` from the union of all tracked objects on a
/// single video frame.  The `masks` tensor on each `ObjectFrameOutput` holds
/// probability values; pixels where any object's probability > 0.5 become 255.
pub fn video_frame_to_mask(
    objects: &[candle_transformers::models::sam3::ObjectFrameOutput],
    out_width: usize,
    out_height: usize,
) -> Result<FrameMask, AppError> {
    if objects.is_empty() {
        return Ok(FrameMask {
            width: out_width,
            height: out_height,
            pixels: vec![0u8; out_width * out_height],
        });
    }
    // Merge masks from all tracked objects (union).
    let n = out_width * out_height;
    let mut merged = vec![0.0f32; n];
    for obj in objects {
        let probs_2d = obj
            .masks
            .squeeze(0)
            .or_else(|_| obj.masks.i(0))
            .and_then(|t| t.squeeze(0).or_else(|_| Ok(t)))
            .and_then(|t| {
                t.upsample_bilinear2d(out_height, out_width, false)
                    .or_else(|_| Ok(t))
            })
            .map_err(|e: candle_core::Error| AppError::internal(e.to_string()))?;
        let flat = probs_2d
            .flatten_all()
            .map_err(|e| AppError::internal(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| AppError::internal(e.to_string()))?;
        for (dst, src) in merged.iter_mut().zip(flat.iter()) {
            if *src > *dst {
                *dst = *src;
            }
        }
    }
    let pixels: Vec<u8> = merged
        .iter()
        .map(|&v| if v > 0.5 { 255 } else { 0 })
        .collect();
    Ok(FrameMask {
        width: out_width,
        height: out_height,
        pixels,
    })
}

#[cfg(test)]
mod tests;
