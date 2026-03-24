use std::collections::HashSet;

#[derive(Debug)]
pub struct ModelRegistry {
    supported_models: HashSet<&'static str>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let supported_models = HashSet::from([
            "sam2_hiera_tiny",
            "sam2_hiera_small",
            "sam2_hiera_base_plus",
            "sam2_hiera_large",
            "sam3",
        ]);

        Self { supported_models }
    }

    pub fn runtime_ready(&self) -> bool {
        false
    }

    pub fn supports_model(&self, model_name: &str) -> bool {
        self.supported_models.contains(model_name)
    }
}
