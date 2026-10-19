use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde_json::{json, Value};

use super::{
    annotation_overlay_path, inference_device, plan, prepare, run, save_masks, validate_video_masks,
};
use crate::{
    app_state::AppState,
    config::AppConfig,
    domain::requests::ProcessRequest,
    imaging::{
        preprocess::PreparedFrameEncoding,
        tiff_io::{read_volume_frames, write_tiff_pages, WritableTiffPage},
    },
    inference::image::FrameMask,
};
use uuid::Uuid;

fn unique_temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("ouroboros-rust-phase3-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn test_state(root: PathBuf) -> AppState {
    test_state_with_volume_server(root, "http://127.0.0.1:3001")
}

fn test_state_with_volume_server(root: PathBuf, volume_server_url: &str) -> AppState {
    let plugin_name = "sam3-segmentation".to_string();
    AppState::new(AppConfig {
        plugin_name: plugin_name.clone(),
        volume_name: "ouroboros-volume".to_string(),
        volume_server_url: volume_server_url.to_string(),
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

// Exercise the same HTTP boundary as the packaged host. The virtual Windows
// host paths map to real fixture files; a successful response requires a copy.
struct VolumeFixture {
    state: AppState,
    host_input: PathBuf,
    host_outputs: PathBuf,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    server: tokio::task::JoinHandle<()>,
    _root: tempfile::TempDir,
}

#[derive(Clone)]
struct VolumeFixtureState {
    plugin_root: PathBuf,
    host_input: PathBuf,
    host_outputs: PathBuf,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    fail_operation: Option<&'static str>,
}

impl VolumeFixture {
    fn start(description: Option<&str>, fail_operation: Option<&'static str>) -> Self {
        let root = tempfile::tempdir().expect("fixture directory");
        let host_input = root.path().join("host-input.tif");
        write_input_volume(&host_input, description);
        let host_outputs = root.path().join("host-outputs");
        let volume_root = root.path().join("volume");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/copy-to-volume",
                post(|State(state), Json(body)| async move {
                    fixture_copy(state, "copy-to-volume", body)
                }),
            )
            .route(
                "/copy-to-host",
                post(|State(state), Json(body)| async move {
                    fixture_copy(state, "copy-to-host", body)
                }),
            )
            .with_state(VolumeFixtureState {
                plugin_root: volume_root.join("sam3-segmentation"),
                host_input: host_input.clone(),
                host_outputs: host_outputs.clone(),
                requests: requests.clone(),
                fail_operation,
            });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind fixture server");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let address = listener.local_addr().expect("fixture address");
        let server = tokio::spawn(async move {
            axum::Server::from_tcp(listener)
                .expect("fixture server")
                .serve(app.into_make_service())
                .await
                .expect("serve volume operations");
        });
        Self {
            state: test_state_with_volume_server(volume_root, &format!("http://{address}")),
            host_input,
            host_outputs,
            requests,
            server,
            _root: root,
        }
    }

    fn requests(&self) -> Vec<(String, Value)> {
        self.requests.lock().expect("recorded requests").clone()
    }
}

impl Drop for VolumeFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn fixture_copy(state: VolumeFixtureState, operation: &str, body: Value) -> (StatusCode, String) {
    state
        .requests
        .lock()
        .expect("record request")
        .push((operation.into(), body.clone()));
    if state.fail_operation == Some(operation) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("fixture {operation} failed"),
        );
    }
    assert_eq!(body["volumeName"], "ouroboros-volume");
    assert_eq!(body["pluginFolderName"], "sam3-segmentation");
    for mapping in body["files"].as_array().expect("file mappings") {
        let host_path = mapping["sourcePath"].as_str().expect("host path");
        let target = mapping["targetPath"].as_str().expect("volume subdirectory");
        // Simulate Node's native Windows basename at the host boundary without
        // borrowing the Rust implementation under test.
        let basename = host_path.rsplit(['\\', '/']).next().expect("basename");
        let volume_path = state.plugin_root.join(target).join(basename);
        let (source, destination) = if operation == "copy-to-volume" {
            assert_eq!(host_path, r"C:\data\input-stack.tif");
            assert_eq!(target, "");
            (state.host_input.clone(), volume_path)
        } else {
            assert_eq!(target, "Segmentation");
            (volume_path, state.host_outputs.join(basename))
        };
        std::fs::create_dir_all(destination.parent().expect("destination parent"))
            .expect("create copy destination");
        if let Err(error) = std::fs::copy(&source, &destination) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("fixture copy failed: {error}"),
            );
        }
    }
    (StatusCode::OK, "Files copied successfully.".into())
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
        plan.copy_to_host_mappings(false)[0].source_path,
        "/host/output/segmented.tif"
    );
    assert_eq!(
        plan.copy_to_host_mappings(false)[0].target_path,
        "Segmentation"
    );
}

#[tokio::test]
async fn prepare_uses_metadata_annotations_and_video_jpeg_targets() {
    let fixture = VolumeFixture::start(
        Some(r#"{"annotation_points": [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]}"#),
        None,
    );
    let state = &fixture.state;
    let plugin_root = state.config().plugin_root();
    assert!(!plugin_root.join("input-stack.tif").exists());

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
async fn prepare_copies_windows_host_input_before_reading_missing_or_stale_volume() {
    for stale in [false, true] {
        let fixture = VolumeFixture::start(None, None);
        let volume_source = fixture.state.config().plugin_root().join("input-stack.tif");
        if stale {
            std::fs::create_dir_all(volume_source.parent().unwrap()).expect("volume directory");
            std::fs::write(&volume_source, b"stale invalid TIFF").expect("stale input");
        } else {
            assert!(!volume_source.exists());
        }

        let prepared = prepare(&fixture.state, &sample_request("ImagePredictor"))
            .await
            .expect("transfer supplies current input before TIFF inspection");

        assert_eq!(
            fixture.requests(),
            vec![(
                "copy-to-volume".into(),
                json!({
                    "volumeName": "ouroboros-volume",
                    "pluginFolderName": "sam3-segmentation",
                    "files": [{ "sourcePath": r"C:\data\input-stack.tif", "targetPath": "" }]
                })
            )]
        );
        assert_eq!(prepared.volume.source, volume_source);
        assert_eq!(
            prepared.staged_frames[0].frame.pixels,
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            std::fs::read(&volume_source).expect("copied input"),
            std::fs::read(&fixture.host_input).expect("host input"),
        );
    }
}

#[tokio::test]
async fn prepare_propagates_copy_failure_even_when_a_volume_input_exists() {
    for stale in [false, true] {
        let fixture = VolumeFixture::start(None, Some("copy-to-volume"));
        let volume_source = fixture.state.config().plugin_root().join("input-stack.tif");
        if stale {
            std::fs::create_dir_all(volume_source.parent().unwrap()).expect("volume directory");
            write_input_volume(&volume_source, None);
        }
        let before = std::fs::read(&volume_source).ok();

        let error = prepare(&fixture.state, &sample_request("ImagePredictor"))
            .await
            .expect_err("failed host transfer must not fall back to cached volume input");

        assert!(
            error.to_string().contains("fixture copy-to-volume failed"),
            "{error}"
        );
        assert_eq!(fixture.requests().len(), 1);
        assert_eq!(fixture.requests()[0].0, "copy-to-volume");
        assert_eq!(std::fs::read(&volume_source).ok(), before);
    }
}

#[tokio::test]
async fn save_masks_delivers_primary_and_requested_overlay_to_windows_host() {
    for overlay in [false, true] {
        let fixture = VolumeFixture::start(None, None);
        let mut request = sample_request("ImagePredictor");
        request.output_file = r"C:\Data Set\output\segmented.tif".into();
        request.overlay_annotation_points = overlay;
        let prepared = prepare(&fixture.state, &request)
            .await
            .expect("prepare input");
        let masks = vec![
            FrameMask {
                width: 4,
                height: 2,
                pixels: vec![0; 8],
            },
            FrameMask {
                width: 4,
                height: 2,
                pixels: vec![255; 8],
            },
        ];
        let (progress, _receiver) = crate::services::progress::channel();
        assert!(!fixture.host_outputs.exists());
        assert!(!prepared.plan.volume_output.exists());

        save_masks(&fixture.state, &prepared, &request, &masks, &progress)
            .await
            .expect("write output before requesting host delivery");

        let mut files = vec![json!({
            "sourcePath": r"C:\Data Set\output\segmented.tif", "targetPath": "Segmentation"
        })];
        if overlay {
            files.push(json!({
                "sourcePath": r"C:\Data Set\output\segmented_annotation_overlay.tif",
                "targetPath": "Segmentation"
            }));
        }
        let requests = fixture.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].0, "copy-to-volume");
        assert_eq!(
            requests[1],
            (
                "copy-to-host".into(),
                json!({
                    "volumeName": "ouroboros-volume", "pluginFolderName": "sam3-segmentation",
                    "files": files
                })
            )
        );
        let host_output = fixture.host_outputs.join("segmented.tif");
        assert_eq!(
            std::fs::read(&host_output).expect("delivered primary mask"),
            std::fs::read(&prepared.plan.volume_output).expect("volume primary mask"),
        );
        let frames = read_volume_frames(&host_output)
            .await
            .expect("delivered valid TIFF");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pixels, masks[0].pixels);
        assert_eq!(frames[1].pixels, masks[1].pixels);
        let host_overlay = fixture
            .host_outputs
            .join("segmented_annotation_overlay.tif");
        assert_eq!(host_overlay.exists(), overlay);
        if overlay {
            assert_eq!(
                std::fs::read(&host_overlay).expect("delivered overlay"),
                std::fs::read(annotation_overlay_path(&prepared.plan.volume_output))
                    .expect("volume overlay"),
            );
            let overlay_frames = read_volume_frames(&host_overlay)
                .await
                .expect("overlay TIFF");
            assert!(overlay_frames[0]
                .pixels
                .contains(&request.annotation_overlay_intensity));
        }
    }
}

#[tokio::test]
async fn save_masks_propagates_host_delivery_failure_and_retains_volume_output() {
    let fixture = VolumeFixture::start(None, Some("copy-to-host"));
    let request = sample_request("ImagePredictor");
    let prepared = prepare(&fixture.state, &request)
        .await
        .expect("prepare input");
    let masks = vec![
        FrameMask {
            width: 4,
            height: 2,
            pixels: vec![255; 8]
        };
        2
    ];
    let (progress, _receiver) = crate::services::progress::channel();

    let error = save_masks(&fixture.state, &prepared, &request, &masks, &progress)
        .await
        .expect_err("saving must fail when host delivery fails");

    assert!(
        error.to_string().contains("fixture copy-to-host failed"),
        "{error}"
    );
    assert!(prepared.plan.volume_output.is_file());
    assert!(!fixture.host_outputs.join("segmented.tif").exists());
    let requests = fixture.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].0, "copy-to-volume");
    assert_eq!(requests[1].0, "copy-to-host");
}

#[tokio::test]
async fn prepare_generates_default_annotations_for_missing_metadata() {
    let fixture = VolumeFixture::start(None, None);
    let state = &fixture.state;

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
async fn validate_video_masks_accepts_expected_frame_count_and_geometry() {
    let fixture = VolumeFixture::start(None, None);
    let state = &fixture.state;
    let prepared = prepare(&state, &sample_request("VideoPredictor"))
        .await
        .expect("prepare pipeline input");
    let masks = vec![
        FrameMask {
            width: 4,
            height: 2,
            pixels: vec![0; 8],
        },
        FrameMask {
            width: 4,
            height: 2,
            pixels: vec![255; 8],
        },
    ];

    validate_video_masks(&masks, &prepared).expect("valid video masks");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_build_selects_cpu_without_cuda_initialization() {
    assert!(inference_device().expect("CPU device").is_cpu());
}

#[tokio::test]
async fn validate_video_masks_rejects_frame_count_geometry_and_pixels() {
    let fixture = VolumeFixture::start(None, None);
    let state = &fixture.state;
    let prepared = prepare(&state, &sample_request("VideoPredictor"))
        .await
        .expect("prepare pipeline input");

    let frame_count_error = validate_video_masks(
        &[FrameMask {
            width: 4,
            height: 2,
            pixels: vec![0; 8],
        }],
        &prepared,
    )
    .expect_err("frame count mismatch");
    assert!(
        frame_count_error
            .to_string()
            .contains("returned 1 masks for 2 staged frames"),
        "{frame_count_error}"
    );

    let geometry_error = validate_video_masks(
        &[
            FrameMask {
                width: 3,
                height: 2,
                pixels: vec![0; 6],
            },
            FrameMask {
                width: 4,
                height: 2,
                pixels: vec![0; 8],
            },
        ],
        &prepared,
    )
    .expect_err("geometry mismatch");
    assert!(
        geometry_error.to_string().contains("expected 4x2"),
        "{geometry_error}"
    );

    let pixel_error = validate_video_masks(
        &[
            FrameMask {
                width: 4,
                height: 2,
                pixels: vec![0; 7],
            },
            FrameMask {
                width: 4,
                height: 2,
                pixels: vec![0; 8],
            },
        ],
        &prepared,
    )
    .expect_err("pixel length mismatch");
    assert!(
        pixel_error.to_string().contains("has 7 pixels, expected 8"),
        "{pixel_error}"
    );
}

#[tokio::test]
async fn run_requires_downloaded_sam3_checkpoint_before_staging() {
    let root = unique_temp_dir();
    let state = test_state(root.clone());
    let plugin_root = state.config().plugin_root();
    std::fs::create_dir_all(&plugin_root).expect("create plugin root");
    write_input_volume(&plugin_root.join("input-stack.tif"), None);

    let (progress, _receiver) = crate::services::progress::channel();
    let error = run(&state, &sample_request("ImagePredictor"), &progress)
        .await
        .expect_err("run without downloaded model should fail");

    assert!(
        error.to_string().contains("SAM3 checkpoint missing"),
        "unexpected error: {error}"
    );
    // Staging should not have happened since the model availability check is first.
    assert!(!plugin_root.join("input-stack_temp").exists());
}

#[test]
fn annotation_overlay_path_keeps_primary_output_clean() {
    assert_eq!(
        annotation_overlay_path(Path::new("/tmp/Segmentation/segmented.tif")),
        PathBuf::from("/tmp/Segmentation/segmented_annotation_overlay.tif")
    );
}
