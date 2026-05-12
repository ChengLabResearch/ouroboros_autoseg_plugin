use std::path::{Path, PathBuf};

use super::{plan, prepare, run};
use crate::{
    app_state::AppState,
    config::AppConfig,
    domain::requests::ProcessRequest,
    imaging::{
        preprocess::PreparedFrameEncoding,
        tiff_io::{write_tiff_pages, WritableTiffPage},
    },
};
use uuid::Uuid;

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase3-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn test_state(root: PathBuf) -> AppState {
    let plugin_name = "sam3-segmentation".to_string();
    AppState::new(AppConfig {
        plugin_name: plugin_name.clone(),
        volume_name: "ouroboros-volume".to_string(),
        volume_server_url: "http://127.0.0.1:3001".to_string(),
        huggingface_base_url: "https://huggingface.co".to_string(),
        checkpoint_dir: root.join(&plugin_name).join("chkpts"),
        internal_volume_path: root,
        fallback_annotation_interval: 200,
        bind_address: "127.0.0.1:8686".parse().expect("valid socket"),
    })
    .expect("state")
}

fn sample_request(predictor_type: &str) -> ProcessRequest {
    ProcessRequest {
        file_path: r#"C:\data\input-stack.tif"#.to_string(),
        output_file: r#"/host/output/segmented.tif"#.to_string(),
        model_type: "sam3".to_string(),
        predictor_type: predictor_type.to_string(),
        overlay_annotation_points: false,
        annotation_overlay_intensity: 127,
    }
}

fn write_input_volume(path: &Path, description: Option<&str>) {
    write_tiff_pages(
        path,
        &[
            WritableTiffPage {
                width: 4,
                height: 2,
                channels: 1,
                pixels: vec![0, 1, 2, 3, 4, 5, 6, 7],
                description: description.map(str::to_string),
            },
            WritableTiffPage {
                width: 4,
                height: 2,
                channels: 1,
                pixels: vec![8, 9, 10, 11, 12, 13, 14, 15],
                description: None,
            },
        ],
    )
    .expect("write input volume");
}

#[tokio::test]
async fn plan_matches_python_volume_mapping_semantics() {
    let state = test_state(PathBuf::from("/tmp/ouroboros-volume"));
    let request = sample_request("ImagePredictor");

    let plan = plan(&state, &request).expect("plan");

    assert_eq!(
        plan.volume_source,
        PathBuf::from("/tmp/ouroboros-volume/sam3-segmentation/input-stack.tif")
    );
    assert_eq!(
        plan.temp_volume_dir,
        PathBuf::from("/tmp/ouroboros-volume/sam3-segmentation/input-stack_temp")
    );
    assert_eq!(
        plan.volume_output,
        PathBuf::from("/tmp/ouroboros-volume/sam3-segmentation/Segmentation/segmented.tif")
    );
    assert_eq!(
        plan.copy_to_volume_mappings()[0].source_path,
        r#"C:\data\input-stack.tif"#
    );
    assert_eq!(plan.copy_to_volume_mappings()[0].target_path, "");
    assert_eq!(
        plan.copy_to_host_mappings()[0].source_path,
        "/host/output/segmented.tif"
    );
    assert_eq!(plan.copy_to_host_mappings()[0].target_path, "Segmentation");
}

#[tokio::test]
async fn prepare_uses_metadata_annotations_and_video_jpeg_targets() {
    let root = unique_temp_dir();
    let state = test_state(root.clone());
    let plugin_root = state.config().plugin_root();
    std::fs::create_dir_all(&plugin_root).expect("create plugin root");
    write_input_volume(
        &plugin_root.join("input-stack.tif"),
        Some(r#"{"annotation_points": [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]}"#),
    );

    let prepared = prepare(&state, &sample_request("VideoPredictor"))
        .await
        .expect("prepare pipeline input");

    assert_eq!(prepared.volume.geometry.frames, 2);
    assert_eq!(prepared.annotation_points.len(), 2);
    assert_eq!(prepared.annotation_points[0].x, 1.0);
    assert_eq!(prepared.annotation_points[1].y, 4.0);
    assert_eq!(prepared.staged_frames.len(), 2);
    assert_eq!(
        prepared.staged_frames[0].encoding,
        PreparedFrameEncoding::Jpeg
    );
    assert_eq!(
        prepared.staged_frames[0].target,
        plugin_root.join("input-stack_temp").join("0")
    );
}

#[tokio::test]
async fn prepare_generates_default_annotations_for_missing_metadata() {
    let root = unique_temp_dir();
    let state = test_state(root.clone());
    let plugin_root = state.config().plugin_root();
    std::fs::create_dir_all(&plugin_root).expect("create plugin root");
    write_input_volume(&plugin_root.join("input-stack.tif"), None);

    let prepared = prepare(&state, &sample_request("ImagePredictor"))
        .await
        .expect("prepare pipeline input");

    assert_eq!(prepared.annotation_points.len(), 2);
    assert_eq!(prepared.annotation_points[0].x, 2.0);
    assert_eq!(prepared.annotation_points[0].y, 1.0);
    assert_eq!(prepared.annotation_points[0].z, 0.0);
    assert_eq!(prepared.annotation_points[1].z, 1.0);
    assert_eq!(
        prepared.staged_frames[0].encoding,
        PreparedFrameEncoding::Tiff
    );
}

#[tokio::test]
async fn run_stages_frames_before_returning_inference_stub() {
    let root = unique_temp_dir();
    let state = test_state(root.clone());
    let plugin_root = state.config().plugin_root();
    std::fs::create_dir_all(&plugin_root).expect("create plugin root");
    write_input_volume(&plugin_root.join("input-stack.tif"), None);

    let error = run(&state, "job-1", &sample_request("ImagePredictor"))
        .await
        .expect_err("pipeline should still be inference stubbed");

    assert_eq!(
        error.to_string(),
        "Rust TIFF staging and prompt plumbing are wired, but Candle inference is not wired yet"
    );
    assert!(plugin_root.join("input-stack_temp").join("0.tif").exists());
    assert!(plugin_root.join("input-stack_temp").join("1.tif").exists());
}
