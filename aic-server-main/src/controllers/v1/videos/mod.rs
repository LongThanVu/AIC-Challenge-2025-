use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use utoipa_axum::{router::OpenApiRouter, routes};

use crate::{
    constants::OPENAPI_TAG,
    models::{dtos::videos::VideoDto, states::AppState},
    services::videos::VideoService,
};

pub fn router() -> OpenApiRouter<AppState> {
    OpenApiRouter::new().routes(routes!(find_by_id))
}

#[utoipa::path(
    get,
    path = "/{id}",
    tag = OPENAPI_TAG,
    params(
        ("id" = i64, Path, description = "Video database id")
    ),
    responses(
        (status = 200, description = "Video matched by id", body = VideoDto)
    )
)]
async fn find_by_id(
    Path(id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<VideoDto>, StatusCode> {
    let service = VideoService::from(&state);
    match service.find_by_id(id).await {
        Ok(result) => result.map(Json).ok_or(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::debug!("{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
