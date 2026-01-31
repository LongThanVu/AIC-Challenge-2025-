use deadpool_diesel::postgres::Pool;
use diesel::{ExpressionMethods, QueryDsl, RunQueryDsl, SelectableHelper};

use crate::{
    models::entities::{keyframes::KeyframeEntity, videos::VideoEntity},
    services::keyframes::KeyframeService,
};

#[derive(Clone)]
pub struct KeyframeRepository<'a> {
    pool: &'a Pool,
}

impl<'a> From<&'a KeyframeService<'_>> for KeyframeRepository<'a> {
    fn from(value: &'a KeyframeService) -> Self {
        Self { pool: value.pool() }
    }
}

impl<'a> KeyframeRepository<'a> {
    pub async fn find_by_id_and_join_video(
        &self,
        id: i64,
    ) -> anyhow::Result<Option<(KeyframeEntity, VideoEntity)>> {
        use crate::schema::keyframes::dsl as keyframe_dsl;
        use crate::schema::videos::dsl as video_dsl;

        let query = keyframe_dsl::keyframes
            .inner_join(video_dsl::videos)
            .filter(keyframe_dsl::id.eq(id))
            .select((KeyframeEntity::as_select(), VideoEntity::as_select()));

        let conn = self.pool.get().await?;

        let results = conn
            .interact(move |conn| query.load::<(KeyframeEntity, VideoEntity)>(conn))
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))??;

        Ok(results.into_iter().next())
    }
}
