use std::path::PathBuf;

use super::plan;
use crate::{app_state::AppState, config::AppConfig, domain::requests::ProcessRequest};

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

#[tokio::test]
async fn plan_matches_python_volume_mapping_semantics() {
    let state = test_state(PathBuf::from("/tmp/ouroboros-volume"));
    let request = ProcessRequest {
        file_path: r#"C:\data\input-stack.tif"#.to_string(),
        output_file: r#"/host/output/segmented.tif"#.to_string(),
        model_type: "sam3".to_string(),
        predictor_type: "ImagePredictor".to_string(),
        overlay_annotation_points: false,
        annotation_overlay_intensity: 127,
    };

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
