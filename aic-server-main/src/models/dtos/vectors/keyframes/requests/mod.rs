#[derive(Debug, Clone, serde::Deserialize, utoipa::ToSchema)]
pub struct VectorizedKeyframeRequestDto {
    prompt: String,
    top_k: u64,
    // object_threshold: f64,
}

impl VectorizedKeyframeRequestDto {
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn top_k(&self) -> u64 {
        self.top_k
    }
}
