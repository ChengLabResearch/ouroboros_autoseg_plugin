use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::sam3::{normalize_rgb_frame_for_sam3, FrameSource, ImageSize};
use image::ImageReader;

#[derive(Debug)]
struct FrameBlob {
    data: Vec<f32>,
    frame_size: ImageSize,
}

impl FrameBlob {
    fn to_tensor(&self, target_device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            self.data.clone(),
            (3, self.frame_size.height, self.frame_size.width),
            &Device::Cpu,
        )?
        .to_device(target_device)
    }

    fn memory_bytes(&self) -> usize {
        self.data.len().saturating_mul(std::mem::size_of::<f32>())
    }
}

/// Lazy adapter for the plugin's staged JPEG frame directory.
///
/// JPEG decode and bicubic resize remain in the plugin's existing `image`
/// dependency. Only normalized tensors cross the Candle model boundary.
#[derive(Debug)]
pub struct StagedJpegFrameSource {
    frame_paths: Vec<PathBuf>,
    image_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    video_size: ImageSize,
    cache: HashMap<usize, FrameBlob>,
}

impl StagedJpegFrameSource {
    pub fn new(
        frames_dir: &Path,
        image_size: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    ) -> Result<Self> {
        let frame_paths = sorted_jpeg_paths(frames_dir)?;
        let first = frame_paths.first().expect("validated non-empty frame list");
        let first_image = ImageReader::open(first)?
            .decode()
            .map_err(candle_core::Error::wrap)?;
        let video_size =
            ImageSize::new(first_image.height() as usize, first_image.width() as usize);
        Ok(Self {
            frame_paths,
            image_size,
            image_mean,
            image_std,
            video_size,
            cache: HashMap::new(),
        })
    }

    pub fn source_size(&self) -> ImageSize {
        self.video_size
    }

    fn ensure_loaded(&mut self, frame_idx: usize) -> Result<()> {
        if self.cache.contains_key(&frame_idx) {
            return Ok(());
        }
        let path = self.frame_paths.get(frame_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("frame_idx {frame_idx} out of bounds"))
        })?;
        let image = ImageReader::open(path)?
            .decode()
            .map_err(candle_core::Error::wrap)?
            .to_rgb8();
        let current_size = ImageSize::new(image.height() as usize, image.width() as usize);
        if current_size != self.video_size {
            candle_core::bail!(
                "frame {} has size {}x{} but the session expects {}x{}",
                path.display(),
                current_size.height,
                current_size.width,
                self.video_size.height,
                self.video_size.width
            )
        }
        let resized = if current_size == ImageSize::square(self.image_size) {
            image
        } else {
            image::imageops::resize(
                &image,
                self.image_size as u32,
                self.image_size as u32,
                image::imageops::FilterType::CatmullRom,
            )
        };
        let image = Tensor::from_vec(
            resized.into_raw(),
            (self.image_size, self.image_size, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;
        let image = (image / 255.)?;
        let normalized = normalize_rgb_frame_for_sam3(&image, self.image_mean, self.image_std)?;
        self.cache.insert(
            frame_idx,
            FrameBlob {
                data: normalized.flatten_all()?.to_vec1::<f32>()?,
                frame_size: ImageSize::square(self.image_size),
            },
        );
        Ok(())
    }
}

impl FrameSource for StagedJpegFrameSource {
    fn frame_count(&self) -> usize {
        self.frame_paths.len()
    }

    fn video_size(&self) -> ImageSize {
        self.video_size
    }

    fn get_frame(&mut self, frame_idx: usize, target_device: &Device) -> Result<Tensor> {
        self.ensure_loaded(frame_idx)?;
        self.cache
            .get(&frame_idx)
            .expect("frame inserted by ensure_loaded")
            .to_tensor(target_device)
    }

    fn prefetch(&mut self, frame_indices: &[usize]) -> Result<()> {
        for frame_idx in frame_indices {
            self.ensure_loaded(*frame_idx)?;
        }
        Ok(())
    }

    fn evict_except(&mut self, keep_frame_indices: &BTreeSet<usize>) {
        self.cache
            .retain(|frame_idx, _| keep_frame_indices.contains(frame_idx));
    }

    fn loaded_frame_count(&self) -> usize {
        self.cache.len()
    }

    fn memory_bytes(&self) -> (usize, usize) {
        (self.cache.values().map(FrameBlob::memory_bytes).sum(), 0)
    }

    fn close(&mut self) {
        self.cache.clear();
    }
}

fn sorted_jpeg_paths(frames_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(frames_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            matches!(
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("jpg" | "jpeg")
            )
        })
        .collect::<Vec<_>>();
    if paths.is_empty() {
        candle_core::bail!(
            "no JPEG frames found in staged frame directory {}",
            frames_dir.display()
        )
    }
    if paths.iter().all(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<usize>().ok())
            .is_some()
    }) {
        paths.sort_by_key(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
    } else {
        paths.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn write_jpeg(path: &Path, color: [u8; 3], width: u32, height: u32) {
        let image = ImageBuffer::from_pixel(width, height, Rgb(color));
        image.save(path).unwrap();
    }

    #[test]
    fn staged_source_is_lazy_bounded_and_numerically_ordered() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        write_jpeg(&temp.path().join("10.jpg"), [10, 20, 30], 3, 2);
        write_jpeg(&temp.path().join("2.jpg"), [40, 50, 60], 3, 2);
        write_jpeg(&temp.path().join("1.jpg"), [70, 80, 90], 3, 2);
        fs::write(temp.path().join("ignored.png"), b"not a jpeg").unwrap();
        let mut source = StagedJpegFrameSource::new(temp.path(), 4, [0.0; 3], [1.0; 3])?;

        assert_eq!(source.frame_count(), 3);
        assert_eq!(source.video_size(), ImageSize::new(2, 3));
        assert_eq!(source.loaded_frame_count(), 0);
        let first = source.get_frame(0, &Device::Cpu)?;
        assert_eq!(first.dims(), &[3, 4, 4]);
        assert_eq!(source.loaded_frame_count(), 1);
        source.prefetch(&[1])?;
        assert_eq!(source.loaded_frame_count(), 2);
        assert_eq!(source.memory_bytes(), (2 * 3 * 4 * 4 * 4, 0));
        source.evict_except(&BTreeSet::from([1]));
        assert_eq!(source.loaded_frame_count(), 1);
        source.close();
        assert_eq!(source.loaded_frame_count(), 0);
        Ok(())
    }

    #[test]
    fn staged_source_rejects_geometry_drift_when_frame_is_requested() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        write_jpeg(&temp.path().join("0.jpg"), [1, 2, 3], 3, 2);
        write_jpeg(&temp.path().join("1.jpg"), [4, 5, 6], 4, 2);
        let mut source = StagedJpegFrameSource::new(temp.path(), 4, [0.0; 3], [1.0; 3])?;

        let error = source.get_frame(1, &Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("session expects 2x3"));
        assert_eq!(source.loaded_frame_count(), 0);
        source.close();
        Ok(())
    }

    #[test]
    fn representative_staged_jpeg_matches_the_pre_backbone_tensor_golden() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let rgb = [60, 120, 180];
        let path = temp.path().join("0.jpg");
        write_jpeg(&path, rgb, 3, 2);
        let decoded_rgb = ImageReader::open(&path)?
            .decode()
            .map_err(candle_core::Error::wrap)?
            .to_rgb8()
            .get_pixel(0, 0)
            .0;
        let mut source = StagedJpegFrameSource::new(temp.path(), 4, [0.0; 3], [1.0; 3])?;

        let values = source
            .get_frame(0, &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (channel, expected) in decoded_rgb.into_iter().enumerate() {
            let expected = expected as f32 / 255.0;
            for value in &values[channel * 16..(channel + 1) * 16] {
                assert!((value - expected).abs() <= 1e-6, "{value} != {expected}");
            }
        }
        Ok(())
    }
}
