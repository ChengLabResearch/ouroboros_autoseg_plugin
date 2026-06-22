use std::{collections::HashMap, sync::Arc, time::Duration};

use candle_core::Device;
use candle_transformers::models::sam3;
use tokio::sync::{Mutex, RwLock, Semaphore};

use crate::{
    config::AppConfig,
    domain::{jobs::JobRecord, responses::StartupStatus},
    error::AppError,
    inference::registry::ModelRegistry,
};

pub struct Sam3ModelHandle {
    pub model_name: String,
    pub image_model: Arc<sam3::Sam3ImageModel>,
    pub tracker: Arc<sam3::Sam3TrackerModel>,
    pub device: Device,
}

#[derive(Clone)]
pub struct AppState {
    config: Arc<AppConfig>,
    http_client: reqwest::Client,
    jobs: Arc<RwLock<HashMap<String, JobRecord>>>,
    startup_status: Arc<RwLock<StartupStatus>>,
    model_registry: Arc<Mutex<ModelRegistry>>,
    sam3_handle: Arc<RwLock<Option<Arc<Sam3ModelHandle>>>>,
    inference_limit: Arc<Semaphore>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Result<Self, AppError> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            config: Arc::new(config),
            http_client,
            jobs: Arc::new(RwLock::new(HashMap::new())),
            startup_status: Arc::new(RwLock::new(StartupStatus::pending())),
            model_registry: Arc::new(Mutex::new(ModelRegistry::new())),
            sam3_handle: Arc::new(RwLock::new(None)),
            inference_limit: Arc::new(Semaphore::new(1)),
        })
    }

    pub fn config(&self) -> &AppConfig {
        self.config.as_ref()
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    pub async fn startup_status(&self) -> StartupStatus {
        self.startup_status.read().await.clone()
    }

    pub async fn set_startup_status(&self, status: StartupStatus) {
        *self.startup_status.write().await = status;
    }

    pub async fn ml_runtime_ready(&self) -> bool {
        self.sam3_handle.read().await.is_some()
    }

    pub async fn sam3_handle(&self, model_name: &str) -> Option<Arc<Sam3ModelHandle>> {
        self.sam3_handle
            .read()
            .await
            .clone()
            .filter(|handle| handle.model_name == model_name)
    }

    pub async fn set_sam3_handle(&self, handle: Sam3ModelHandle) -> Arc<Sam3ModelHandle> {
        let handle = Arc::new(handle);
        *self.sam3_handle.write().await = Some(handle.clone());
        self.model_registry.lock().await.mark_ready();
        handle
    }

    pub async fn create_job(&self, job_id: String) -> JobRecord {
        let record = JobRecord::new_running();
        self.jobs.write().await.insert(job_id, record.clone());
        record
    }

    pub async fn get_job(&self, job_id: &str) -> Option<JobRecord> {
        self.jobs.read().await.get(job_id).cloned()
    }

    pub async fn latest_running_job(&self) -> Option<(String, JobRecord)> {
        self.jobs
            .read()
            .await
            .iter()
            .filter(|(_, record)| record.status == "running")
            .max_by(|left, right| {
                left.1
                    .created_at
                    .partial_cmp(&right.1.created_at)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(job_id, record)| (job_id.clone(), record.clone()))
    }

    pub async fn update_job_step(&self, job_id: &str, step_index: usize, progress: u8) {
        if let Some(record) = self.jobs.write().await.get_mut(job_id) {
            record.update_step(step_index, progress);
        }
    }

    pub async fn set_job_status(&self, job_id: &str, status: &str) {
        if let Some(record) = self.jobs.write().await.get_mut(job_id) {
            record.set_status(status);
        }
    }

    pub async fn with_inference_slot<F, Fut, T>(&self, func: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let _permit = self.inference_limit.acquire().await.ok();
        func().await
    }
}
