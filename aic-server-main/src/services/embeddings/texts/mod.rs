use tonic::Request;

use crate::models::protos::text_embed::{
    TextRequest, text_embedding_service_client::TextEmbeddingServiceClient,
};

pub struct TextEmbeddingService {
    client: TextEmbeddingServiceClient<tonic::transport::Channel>,
}

impl TextEmbeddingService {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            client: TextEmbeddingServiceClient::connect(std::env::var("EMBEDDING_SERVER_URL")?)
                .await?,
        })
    }

    pub async fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let request = Request::new(TextRequest { text: text.into() });
        let response = self.client.get_embedding(request).await?;
        Ok(response.into_inner().embeddings)
    }
}
