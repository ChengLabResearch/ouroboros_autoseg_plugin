use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

use crate::error::AppError;

#[derive(Debug, Clone, Copy)]
pub enum DownloadSource {
    PublicUrl(&'static str),
    HuggingFace {
        repo: &'static str,
        filename: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ModelDescriptor {
    pub model_name: &'static str,
    pub checkpoint_file: &'static str,
    pub download_source: DownloadSource,
}

const MODEL_CATALOG: &[ModelDescriptor] = &[
    ModelDescriptor {
        model_name: "sam2_hiera_tiny",
        checkpoint_file: "sam2_hiera_tiny.pt",
        download_source: DownloadSource::PublicUrl(
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        ),
    },
    ModelDescriptor {
        model_name: "sam2_hiera_small",
        checkpoint_file: "sam2_hiera_small.pt",
        download_source: DownloadSource::PublicUrl(
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        ),
    },
    ModelDescriptor {
        model_name: "sam2_hiera_base_plus",
        checkpoint_file: "sam2_hiera_base_plus.pt",
        download_source: DownloadSource::PublicUrl(
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        ),
    },
    ModelDescriptor {
        model_name: "sam2_hiera_large",
        checkpoint_file: "sam2_hiera_large.pt",
        download_source: DownloadSource::PublicUrl(
            "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        ),
    },
    ModelDescriptor {
        model_name: "sam3",
        checkpoint_file: "sam3.pt",
        download_source: DownloadSource::HuggingFace {
            repo: "facebook/sam3",
            filename: "sam3.pt",
        },
    },
];

const TRACKED_MODEL_STATUS: &[&str] = &["sam2_hiera_base_plus", "sam3"];

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub plugin_name: String,
    pub volume_server_url: String,
    pub internal_volume_path: PathBuf,
    pub checkpoint_dir: PathBuf,
    pub fallback_annotation_interval: usize,
    pub bind_address: SocketAddr,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, AppError> {
        let plugin_name = "sam3-segmentation".to_string();
        let volume_server_url = std::env::var("VOLUME_SERVER_URL")
            .unwrap_or_else(|_| "http://host.docker.internal:3001".to_string());
        let internal_volume_path = if running_in_docker() {
            PathBuf::from("/ouroboros-volume")
        } else {
            PathBuf::from(
                std::env::var("VOLUME_MOUNT_PATH")
                    .unwrap_or_else(|_| "/tmp/ouroboros-volume".to_string()),
            )
        };
        let checkpoint_dir = internal_volume_path.join(&plugin_name).join("chkpts");
        let fallback_annotation_interval = std::env::var("FALLBACK_ANNOTATION_INTERVAL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(200);
        let bind_address = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            std::env::var("PORT")
                .ok()
                .and_then(|value| value.parse::<u16>().ok())
                .unwrap_or(8686),
        );

        Ok(Self {
            plugin_name,
            volume_server_url,
            internal_volume_path,
            checkpoint_dir,
            fallback_annotation_interval,
            bind_address,
        })
    }

    pub fn model_descriptor(&self, model_name: &str) -> Option<&'static ModelDescriptor> {
        MODEL_CATALOG
            .iter()
            .find(|descriptor| descriptor.model_name == model_name)
    }

    pub fn tracked_model_status(&self) -> &'static [&'static str] {
        TRACKED_MODEL_STATUS
    }

    pub fn checkpoint_path(&self, model_name: &str) -> PathBuf {
        self.checkpoint_dir.join(format!("{model_name}.pt"))
    }

    pub fn ensure_checkpoint_dir(&self) -> Result<(), AppError> {
        std::fs::create_dir_all(&self.checkpoint_dir)?;
        Ok(())
    }
}

fn running_in_docker() -> bool {
    Path::new("/.dockerenv").exists() || std::env::var("RUNNING_IN_DOCKER").as_deref() == Ok("1")
}
