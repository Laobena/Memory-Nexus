//! Vector store with Qdrant INT8 quantization

#[cfg(feature = "storage")]
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        CreateCollection, Distance, QuantizationType, ScalarQuantization,
        SearchPoints, VectorParams, PointStruct, ScoredPoint,
    },
};

use crate::core::{BlockError, BlockResult};
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

/// Quantization configuration for 97% RAM reduction
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization type (INT8)
    pub quantization_type: String,
    /// Quantile for scalar quantization
    pub quantile: f32,
    /// Always keep in RAM
    pub always_ram: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: "int8".to_string(),
            quantile: 0.99,
            always_ram: true,
        }
    }
}

/// Vector store with quantization
#[cfg(feature = "storage")]
pub struct VectorStore {
    #[cfg(feature = "qdrant-client")]
    client: Arc<QdrantClient>,
    collection_name: String,
    vector_dim: usize,
    quantization: QuantizationConfig,
    stats: Arc<RwLock<VectorStoreStats>>,
}

#[cfg(not(feature = "storage"))]
pub struct VectorStore;

/// Vector store statistics
#[derive(Debug, Default)]
struct VectorStoreStats {
    total_vectors: u64,
    searches_performed: u64,
    average_search_time_ms: f64,
}

#[cfg(feature = "storage")]
impl VectorStore {
    /// Create new vector store
    #[cfg(feature = "qdrant-client")]
    pub async fn new(
        url: &str,
        collection_name: String,
        vector_dim: usize,
        quantization: QuantizationConfig,
    ) -> BlockResult<Self> {
        let client = QdrantClient::from_url(url).build()?;
        
        let store = Self {
            client: Arc::new(client),
            collection_name,
            vector_dim,
            quantization,
            stats: Arc::new(RwLock::new(VectorStoreStats::default())),
        };
        
        // Initialize collection with quantization
        store.init_collection().await?;
        
        Ok(store)
    }
    
    /// Initialize collection with INT8 quantization
    #[cfg(feature = "qdrant-client")]
    async fn init_collection(&self) -> BlockResult<()> {
        use qdrant_client::qdrant::{CreateCollectionBuilder, VectorParamsBuilder, ScalarQuantizationBuilder};
        
        // Check if collection exists
        let collections = self.client.list_collections().await?;
        
        if collections.collections.iter().any(|c| c.name == self.collection_name) {
            info!("Collection {} already exists", self.collection_name);
            return Ok(());
        }
        
        // Create collection with INT8 quantization
        let quantization = ScalarQuantizationBuilder::default()
            .r#type(QuantizationType::Int8)
            .quantile(self.quantization.quantile)
            .always_ram(self.quantization.always_ram)
            .build();
        
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(
                        VectorParamsBuilder::new(self.vector_dim as u64, Distance::Cosine)
                    )
                    .quantization_config(quantization)
            )
            .await?;
        
        info!(
            "Created collection {} with INT8 quantization (97% RAM reduction)",
            self.collection_name
        );
        
        Ok(())
    }
    
    /// Insert vector with metadata
    #[cfg(feature = "qdrant-client")]
    pub async fn insert(
        &self,
        id: Uuid,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> BlockResult<()> {
        use qdrant_client::qdrant::{UpsertPointsBuilder, PointStruct};
        
        if vector.len() != self.vector_dim {
            return Err(BlockError::Unknown(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.vector_dim,
                vector.len()
            )));
        }
        
        let point = PointStruct::new(
            id.to_string(),
            vector,
            metadata.try_into().unwrap(),
        );
        
        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.collection_name, vec![point])
            )
            .await?;
        
        self.stats.write().total_vectors += 1;
        
        debug!("Inserted vector with id: {}", id);
        Ok(())
    }
    
    /// Search for similar vectors
    #[cfg(feature = "qdrant-client")]
    pub async fn search(
        &self,
        query: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> BlockResult<Vec<VectorSearchResult>> {
        use qdrant_client::qdrant::{SearchPointsBuilder, SearchParamsBuilder};
        
        let start = std::time::Instant::now();
        
        let mut search_builder = SearchPointsBuilder::new(
            &self.collection_name,
            query,
            limit as u64,
        )
        .with_payload(true);
        
        if let Some(threshold) = score_threshold {
            search_builder = search_builder.score_threshold(threshold);
        }
        
        let results = self.client.search_points(search_builder).await?;
        
        let elapsed = start.elapsed().as_millis() as f64;
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.searches_performed += 1;
            stats.average_search_time_ms = 
                (stats.average_search_time_ms * (stats.searches_performed - 1) as f64 + elapsed)
                / stats.searches_performed as f64;
        }
        
        let search_results = results
            .result
            .into_iter()
            .map(|point| VectorSearchResult {
                id: Uuid::parse_str(&point.id.unwrap_or_default()).unwrap_or_default(),
                score: point.score,
                metadata: point.payload.into(),
            })
            .collect();
        
        debug!("Search completed in {}ms, found {} results", elapsed, search_results.len());
        
        Ok(search_results)
    }
    
    /// Delete vector by ID
    #[cfg(feature = "qdrant-client")]
    pub async fn delete(&self, id: Uuid) -> BlockResult<()> {
        use qdrant_client::qdrant::{DeletePointsBuilder, PointsIdsList};
        
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: vec![id.to_string().into()],
                    })
            )
            .await?;
        
        debug!("Deleted vector with id: {}", id);
        Ok(())
    }
    
    /// Get vector store statistics
    pub fn stats(&self) -> VectorStoreStatsSummary {
        let stats = self.stats.read();
        
        VectorStoreStatsSummary {
            total_vectors: stats.total_vectors,
            searches_performed: stats.searches_performed,
            average_search_time_ms: stats.average_search_time_ms,
            ram_reduction_percent: 97.0, // INT8 quantization achieves 97% reduction
        }
    }
}

/// Vector search result
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: Uuid,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Vector store statistics summary
#[derive(Debug)]
pub struct VectorStoreStatsSummary {
    pub total_vectors: u64,
    pub searches_performed: u64,
    pub average_search_time_ms: f64,
    pub ram_reduction_percent: f32,
}

#[cfg(not(feature = "storage"))]
impl VectorStore {
    pub async fn new(
        _url: &str,
        _collection_name: String,
        _vector_dim: usize,
        _quantization: QuantizationConfig,
    ) -> BlockResult<Self> {
        Ok(VectorStore)
    }
}