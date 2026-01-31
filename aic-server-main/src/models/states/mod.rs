use std::sync::Arc;

use deadpool_diesel::{
    Runtime,
    postgres::{Manager, Pool},
};
use qdrant_client::Qdrant;
use translators::GoogleTranslator;

#[derive(Clone)]
pub struct AppState {
    diesel_pool: Pool,
    qdrant_client: Arc<Qdrant>,
    translator: GoogleTranslator,
}

impl AppState {
    pub async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            diesel_pool: Self::diesel_pool_helper()?,
            qdrant_client: Self::qdrant_client_helper().await,
            translator: GoogleTranslator::default(),
        })
    }

    fn diesel_pool_helper() -> anyhow::Result<Pool> {
        tracing::debug!("initializing database connection pool");
        Ok(Pool::builder(Manager::new(
            std::env::var("DATABASE_URL").expect("`DATABASE_URL` environment variable must be set"),
            Runtime::Tokio1,
        ))
        .build()?)
    }

    async fn qdrant_client_helper() -> Arc<Qdrant> {
        tracing::debug!("initializing qdrant client");
        let client = Qdrant::from_url(
            &std::env::var("QDRANT_URL").expect("`QDRANT_URL` environment variable must be set"),
        )
        .api_key(
            std::env::var("QDRANT_API_KEY")
                .expect("`QDRANT_API_KEY` environment variable must be set"),
        )
        .build()
        .expect("Qdrant client initialization failed");

        Arc::new(client)
    }

    pub fn diesel_pool(&self) -> &Pool {
        &self.diesel_pool
    }

    pub fn qdrant_client(&self) -> &Qdrant {
        &self.qdrant_client
    }

    pub fn translator(&self) -> &GoogleTranslator {
        &self.translator
    }
}
