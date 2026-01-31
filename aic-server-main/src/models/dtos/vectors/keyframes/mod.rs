use qdrant_client::qdrant::{ScoredPoint, point_id::PointIdOptions};

mod requests;

pub use requests::VectorizedKeyframeRequestDto;

#[derive(Debug, Clone, Copy, serde::Serialize, utoipa::ToSchema)]
pub struct VectorizedKeyframeDto {
    id: u64,
    score: f32,
}

impl TryFrom<ScoredPoint> for VectorizedKeyframeDto {
    type Error = anyhow::Error;
    fn try_from(value: ScoredPoint) -> Result<Self, Self::Error> {
        Ok(Self {
            id: value
                .id
                .and_then(|x| x.point_id_options)
                .and_then(|x| {
                    if let PointIdOptions::Num(x) = x {
                        Some(x)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| anyhow::anyhow!("scored point should have id"))?,
            score: value.score,
        })
    }
}
