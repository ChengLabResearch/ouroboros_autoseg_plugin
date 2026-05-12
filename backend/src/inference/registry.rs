use std::collections::HashSet;

#[derive(Debug)]
pub struct ModelRegistry {
    supported_models: HashSet<&'static str>,
    ready: bool,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let supported_models = HashSet::from([
            "sam2_hiera_tiny",
            "sam2_hiera_small",
            "sam2_hiera_base_plus",
            "sam2_hiera_large",
            "sam3",
            "candle_sam3",
        ]);

        Self {
            supported_models,
            ready: false,
        }
    }

    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    pub fn runtime_ready(&self) -> bool {
        self.ready
    }

    pub fn supports_model(&self, model_name: &str) -> bool {
        self.supported_models.contains(model_name)
    }
}
