use utoipa_axum::router::OpenApiRouter;

use crate::models::states::AppState;

mod keyframes;

pub fn router() -> OpenApiRouter<AppState> {
    OpenApiRouter::new().nest("/keyframes", keyframes::router())
}
