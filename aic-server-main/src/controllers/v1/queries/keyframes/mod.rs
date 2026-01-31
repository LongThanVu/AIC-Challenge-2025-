use axum::{Json, extract::State, http::StatusCode};
use utoipa_axum::{router::OpenApiRouter, routes};

use crate::{
    constants::OPENAPI_TAG,
    models::{
        dtos::vectors::keyframes::{VectorizedKeyframeDto, VectorizedKeyframeRequestDto},
        states::AppState,
    },
    services::queries::keyframes::VectorizedKeyframeService,
};

pub fn router() -> OpenApiRouter<AppState> {
    OpenApiRouter::new().routes(routes!(find_nearest_top_k_by_text))
}

#[utoipa::path(
    post,
    path = "/searches",
    tag = OPENAPI_TAG,
    request_body = VectorizedKeyframeRequestDto,
    responses(
        (status = 200, description = "Top k documents nearest to prompt", body = Vec<VectorizedKeyframeDto>)
    )
)]
async fn find_nearest_top_k_by_text(
    State(state): State<AppState>,
    Json(dto): Json<VectorizedKeyframeRequestDto>,
) -> Result<Json<Vec<VectorizedKeyframeDto>>, StatusCode> {
    let service = VectorizedKeyframeService::from(&state);
    match service
        .find_nearest_top_k_by_text(dto.prompt(), dto.top_k())
        .await
        .map(Json)
    {
        Ok(value) => Ok(value),
        Err(e) => {
            tracing::error!("{:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
