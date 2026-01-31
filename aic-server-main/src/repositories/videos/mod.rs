use deadpool_diesel::postgres::Pool;
use diesel::{ExpressionMethods, QueryDsl, RunQueryDsl, SelectableHelper};

use crate::{models::entities::videos::VideoEntity, services::videos::VideoService};

#[derive(Clone)]
pub struct VideoRepository<'a> {
    pool: &'a Pool,
}

impl<'a> From<&'a VideoService<'_>> for VideoRepository<'a> {
    fn from(value: &'a VideoService) -> Self {
        Self { pool: value.pool() }
    }
}

impl<'a> VideoRepository<'a> {
    pub async fn find_by_id(&self, id: i64) -> anyhow::Result<Option<VideoEntity>> {
        use crate::schema::videos::dsl;

        let query = dsl::videos
            .filter(dsl::id.eq(id))
            .select(VideoEntity::as_select());

        let conn = self.pool.get().await?;

        let results = conn
            .interact(move |conn| query.load::<VideoEntity>(conn))
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))??;

        Ok(results.into_iter().next())
    }
}
