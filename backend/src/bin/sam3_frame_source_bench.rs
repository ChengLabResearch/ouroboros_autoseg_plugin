use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::sam3::{normalize_rgb_frame_for_sam3, Config, FrameSource};
use image::ImageReader;
use ouroboros_autoseg_plugin_backend::inference::candle_sam3_frame_source::StagedJpegFrameSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frames_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: sam3_frame_source_bench <staged-jpeg-directory> [passes]")?;
    let passes = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    if passes == 0 {
        return Err("passes must be greater than zero".into());
    }

    let config = Config::default();
    let mut paths = std::fs::read_dir(&frames_dir)?
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
    paths.sort();
    let mut source = StagedJpegFrameSource::new(
        &frames_dir,
        config.image.image_size,
        config.image.image_mean,
        config.image.image_std,
    )?;
    let frame_count = source.frame_count();
    let mut peak_loaded_frames = source.loaded_frame_count();
    let mut peak_cpu_bytes = source.memory_bytes().0;
    let started = Instant::now();
    for _ in 0..passes {
        for frame_idx in 0..frame_count {
            let _ = source.get_frame(frame_idx, &Device::Cpu)?;
            peak_loaded_frames = peak_loaded_frames.max(source.loaded_frame_count());
            peak_cpu_bytes = peak_cpu_bytes.max(source.memory_bytes().0);
            source.evict_except(&BTreeSet::new());
        }
    }
    let elapsed = started.elapsed();
    let decoded_frames = frame_count.saturating_mul(passes);
    source.close();

    let legacy_started = Instant::now();
    let mut legacy_cache = HashMap::<usize, Vec<f32>>::new();
    for _ in 0..passes {
        for (frame_idx, path) in paths.iter().enumerate() {
            let image = ImageReader::open(path)?.decode()?.to_rgb8();
            let resized = image::imageops::resize(
                &image,
                config.image.image_size as u32,
                config.image.image_size as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let image = Tensor::from_vec(
                resized.into_raw(),
                (config.image.image_size, config.image.image_size, 3),
                &Device::Cpu,
            )?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?;
            let image = (image / 255.)?;
            let normalized = normalize_rgb_frame_for_sam3(
                &image,
                config.image.image_mean,
                config.image.image_std,
            )?;
            let data = normalized.flatten_all()?.to_vec1::<f32>()?;
            legacy_cache.insert(frame_idx, data);
            let _ = Tensor::from_vec(
                legacy_cache
                    .get(&frame_idx)
                    .expect("inserted legacy frame")
                    .clone(),
                (3, config.image.image_size, config.image.image_size),
                &Device::Cpu,
            )?;
            legacy_cache.clear();
        }
    }
    let legacy_elapsed = legacy_started.elapsed();
    let frames_per_second = decoded_frames as f64 / elapsed.as_secs_f64();
    let legacy_frames_per_second = decoded_frames as f64 / legacy_elapsed.as_secs_f64();
    let regression_percent =
        (legacy_frames_per_second - frames_per_second) / legacy_frames_per_second * 100.0;

    println!(
        "{{\"frame_count\":{frame_count},\"passes\":{passes},\"decoded_frames\":{decoded_frames},\"elapsed_seconds\":{:.6},\"frames_per_second\":{frames_per_second:.6},\"legacy_equivalent_elapsed_seconds\":{:.6},\"legacy_equivalent_frames_per_second\":{legacy_frames_per_second:.6},\"regression_percent\":{regression_percent:.3},\"peak_loaded_frames\":{peak_loaded_frames},\"peak_cpu_bytes\":{peak_cpu_bytes},\"final_loaded_frames\":{}}}",
        elapsed.as_secs_f64(),
        legacy_elapsed.as_secs_f64(),
        source.loaded_frame_count()
    );
    Ok(())
}
