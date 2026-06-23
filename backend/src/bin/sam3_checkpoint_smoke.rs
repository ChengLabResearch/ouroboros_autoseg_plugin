use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use ouroboros_autoseg_plugin_backend::{
    imaging::{output::write_mask_stack, tiff_io::ImageFrame},
    inference::{
        candle_sam3::{load_sam3_handle, CandleSam3ImageSegmenter},
        image::{ImageSegmenter, PositivePointPrompt},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args().skip(1))?;
    let frame = match args.image_path.as_deref() {
        Some(path) => load_image_frame(path)?,
        None => synthetic_frame(),
    };

    let handle = load_sam3_handle(
        args.model_name,
        &args.checkpoint_path,
        candle_core::Device::Cpu,
    )?;
    let segmenter = CandleSam3ImageSegmenter {
        handle: Arc::new(handle),
    };
    let prompt = PositivePointPrompt {
        x: (frame.width / 2) as f32,
        y: (frame.height / 2) as f32,
    };
    let mask = segmenter.segment(&frame, &[prompt]).await?;

    if mask.width != frame.width || mask.height != frame.height {
        return Err(format!(
            "mask geometry {}x{} does not match input {}x{}",
            mask.width, mask.height, frame.width, frame.height
        )
        .into());
    }
    if mask.pixels.len() != frame.width * frame.height {
        return Err("mask pixel count does not match frame geometry".into());
    }

    if let Some(mask_out) = args.mask_out {
        write_mask_stack(&mask_out, &[mask]).await?;
        println!("wrote mask {}", mask_out.display());
    }

    Ok(())
}

#[derive(Debug)]
struct Args {
    checkpoint_path: PathBuf,
    image_path: Option<PathBuf>,
    mask_out: Option<PathBuf>,
    model_name: String,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut checkpoint_path = None;
        let mut image_path = None;
        let mut mask_out = None;
        let mut model_name = "medical_sam3".to_string();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--checkpoint" => checkpoint_path = Some(next_path(&mut args, "--checkpoint")?),
                "--image" => image_path = Some(next_path(&mut args, "--image")?),
                "--mask-out" => mask_out = Some(next_path(&mut args, "--mask-out")?),
                "--model-name" => {
                    model_name = args
                        .next()
                        .ok_or_else(|| "--model-name requires a value".to_string())?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                value => return Err(format!("unknown argument: {value}").into()),
            }
        }

        let checkpoint_path =
            checkpoint_path.ok_or_else(|| "--checkpoint is required".to_string())?;

        Ok(Self {
            checkpoint_path,
            image_path,
            mask_out,
            model_name,
        })
    }
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a path").into())
}

fn print_help() {
    println!(
        "Usage: sam3_checkpoint_smoke --checkpoint PATH [--image PATH] [--mask-out PATH] [--model-name NAME]"
    );
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

fn load_image_frame(path: &Path) -> Result<ImageFrame, Box<dyn Error>> {
    let image = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (width, height) = image.dimensions();
    Ok(ImageFrame {
        width: width as usize,
        height: height as usize,
        channels: 3,
        pixels: image.into_raw(),
    })
}
