use deadpool_diesel::postgres::Pool;

use crate::{
    models::{dtos::videos::VideoDto, states::AppState},
    repositories::videos::VideoRepository,
};

#[derive(Clone, Copy)]
pub struct VideoService<'a> {
    pool: &'a Pool,
}

impl<'a> From<&'a AppState> for VideoService<'a> {
    fn from(value: &'a AppState) -> Self {
        Self {
            pool: value.diesel_pool(),
        }
    }
}

impl<'a> VideoService<'a> {
    pub const fn pool(&self) -> &'a Pool {
        self.pool
    }
}

impl<'a> VideoService<'a> {
    pub async fn find_by_id(&self, id: i64) -> anyhow::Result<Option<VideoDto>> {
        let repository = VideoRepository::from(self);
        Ok(repository.find_by_id(id).await?.map(VideoDto::from))
    }
}
