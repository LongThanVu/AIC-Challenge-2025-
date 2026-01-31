use qdrant_client::{
    Qdrant,
    qdrant::{Query, QueryPointsBuilder},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use translators::{GoogleTranslator, Translator};

use crate::{
    constants::QDRANT_KEYFRAME_COLLECTION_NAME,
    models::{dtos::vectors::keyframes::VectorizedKeyframeDto, states::AppState},
    services::embeddings::texts::TextEmbeddingService,
};

#[derive(Clone, Copy)]
pub struct VectorizedKeyframeService<'a> {
    client: &'a Qdrant,
    translator: &'a GoogleTranslator,
}

impl<'a> From<&'a AppState> for VectorizedKeyframeService<'a> {
    fn from(value: &'a AppState) -> Self {
        Self {
            client: value.qdrant_client(),
            translator: value.translator(),
        }
    }
}

impl<'a> VectorizedKeyframeService<'a> {
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut embedding_service = TextEmbeddingService::new().await?;
        embedding_service.embed_text(text).await
    }

    pub async fn find_nearest_top_k_by_text(
        &self,
        text: &str,
        top_k: u64,
    ) -> anyhow::Result<Vec<VectorizedKeyframeDto>> {
        let translated_text = self.translator.translate_async(text, "vi", "en").await?;
        let embeddings = self.embed_text(&translated_text).await?;

        let query_result = self
            .client
            .query(
                QueryPointsBuilder::new(QDRANT_KEYFRAME_COLLECTION_NAME)
                    .query(Query::new_nearest(embeddings))
                    .using("images")
                    .limit(top_k),
            )
            .await?
            .result;

        query_result
            .into_par_iter()
            .map(VectorizedKeyframeDto::try_from)
            .collect::<Result<_, _>>()
    }
}
