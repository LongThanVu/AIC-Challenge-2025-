use crate::models::entities::{keyframes::KeyframeEntity, videos::VideoEntity};

#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct KeyframeDto {
    id: i64,
    name: String,
    video_id: i64,
    video_name: String,
    watch_url: String,
    frame_index: i64,
    frame_timestamp: f32,
}

impl From<(KeyframeEntity, VideoEntity)> for KeyframeDto {
    fn from((mut keyframe, mut video): (KeyframeEntity, VideoEntity)) -> Self {
        let id = keyframe.id();
        let name = std::mem::take(keyframe.name_mut());
        let video_id = video.id();
        let video_name = std::mem::take(video.name_mut());
        let watch_url = std::mem::take(video.watch_url_mut());
        let frame_index = keyframe.frame_index();
        let frame_timestamp = keyframe.frame_timestamp();
        Self {
            id,
            name,
            video_id,
            video_name,
            watch_url,
            frame_index,
            frame_timestamp,
        }
    }
}

impl From<(VideoEntity, KeyframeEntity)> for KeyframeDto {
    fn from((video, keyframe): (VideoEntity, KeyframeEntity)) -> Self {
        Self::from((keyframe, video))
    }
}
