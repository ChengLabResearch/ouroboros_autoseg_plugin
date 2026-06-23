use std::{path::PathBuf, sync::Arc};

use ouroboros_autoseg_plugin_backend::{
    imaging::tiff_io::ImageFrame,
    inference::{
        candle_sam3::{load_sam3_handle, CandleSam3ImageSegmenter},
        image::{ImageSegmenter, PositivePointPrompt},
    },
};

#[tokio::test]
#[ignore = "requires OUROBOROS_SAM3_CHECKPOINT pointing to a staged SAM3 checkpoint"]
async fn sam3_medical_checkpoint_load() {
    let Some(checkpoint_path) = checkpoint_from_env() else {
        eprintln!("skipping: OUROBOROS_SAM3_CHECKPOINT is not set");
        return;
    };

    let handle = load_sam3_handle(
        "medical_sam3".to_string(),
        &checkpoint_path,
        candle_core::Device::Cpu,
    )
    .expect("checkpoint should load through the production Candle SAM3 source");

    let frame = synthetic_frame();
    let segmenter = CandleSam3ImageSegmenter {
        handle: Arc::new(handle),
    };
    let mask = segmenter
        .segment(
            &frame,
            &[PositivePointPrompt {
                x: (frame.width / 2) as f32,
                y: (frame.height / 2) as f32,
            }],
        )
        .await
        .expect("image path should produce a mask");

    assert_eq!(mask.width, frame.width);
    assert_eq!(mask.height, frame.height);
    assert_eq!(mask.pixels.len(), frame.width * frame.height);
    assert!(
        mask.pixels.iter().all(|&value| value == 0 || value == 255),
        "mask should use the established 0/255 convention"
    );
}

fn checkpoint_from_env() -> Option<PathBuf> {
    let path = std::env::var_os("OUROBOROS_SAM3_CHECKPOINT").map(PathBuf::from)?;
    assert!(
        path.is_file(),
        "OUROBOROS_SAM3_CHECKPOINT must point to a checkpoint file: {}",
        path.display()
    );
    Some(path)
}

fn synthetic_frame() -> ImageFrame {
    let width = 32usize;
    let height = 32usize;
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let inside = (8..24).contains(&x) && (8..24).contains(&y);
            let v = if inside { 220 } else { 32 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    ImageFrame {
        width,
        height,
        channels: 3,
        pixels,
    }
}
