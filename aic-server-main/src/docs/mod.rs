use utoipa::OpenApi;

use crate::constants::OPENAPI_TAG;

#[derive(OpenApi)]
#[openapi(
    tags(
        (name = OPENAPI_TAG, description = "AIC data server API")
    )
)]
pub struct ApiDoc;
