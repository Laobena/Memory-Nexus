use crate::core::{Config, Result, NexusError};
use crate::core::zero_copy::{FastSerializer, ZeroCopyAccessor};
use super::preprocessor::PreprocessedData;
use async_trait::async_trait;
use dashmap::DashMap;
use memmap2::{MmapOptions, Mmap};
use std::sync::Arc;
use std::fs::File;
use parking_lot::RwLock;
use bytes::Bytes;

/// High-performance storage engine with multiple backends
pub struct StorageEngine {
    backends: DashMap<String, Arc<dyn StorageBackend>>,
    primary_backend: Arc<RwLock<String>>,
    cache: Arc<moka::future::Cache<String, Bytes>>,
    compression_enabled: bool,
    serializer: Arc<FastSerializer>,
    mmap_cache: Arc<DashMap<String, Arc<Mmap>>>,
}

#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn store(&self, key: &str, data: &[u8]) -> Result<()>;
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    fn name(&self) -> &str;
}

impl StorageEngine {
    pub fn new() -> Self {
        let cache = moka::future::Cache::builder()
            .max_capacity(10_000)
            .time_to_live(std::time::Duration::from_secs(3600))
            .build();
            
        Self {
            backends: DashMap::new(),
            primary_backend: Arc::new(RwLock::new("memory".to_string())),
            cache: Arc::new(cache),
            compression_enabled: true,
            serializer: Arc::new(FastSerializer::with_capacity(4096)),
            mmap_cache: Arc::new(DashMap::new()),
        }
    }
    
    pub async fn initialize(&self, _config: &Config) -> Result<()> {
        // Register default backends
        self.register_backend("memory", Arc::new(MemoryBackend::new()));
        
        tracing::debug!("Storage engine initialized with memory backend");
        Ok(())
    }
    
    pub async fn store(&self, data: &PreprocessedData) -> Result<()> {
        let key = generate_storage_key(&data.original);
        
        // Use zero-copy serialization instead of JSON
        let serialized = self.serializer.serialize(data)
            .map_err(|e| NexusError::Serialization(e.to_string()))?;
        
        let compressed = if self.compression_enabled {
            compress_data(&serialized)?
        } else {
            serialized
        };
        
        // Store in cache
        self.cache.insert(key.clone(), Bytes::from(compressed.clone())).await;
        
        // Store in primary backend
        let backend_name = self.primary_backend.read().clone();
        if let Some(backend) = self.backends.get(&backend_name) {
            backend.store(&key, &compressed).await?;
        }
        
        Ok(())
    }
    
    pub async fn retrieve(&self, key: &str) -> Result<Option<PreprocessedData>> {
        // Check cache first
        if let Some(cached) = self.cache.get(key).await {
            let decompressed = if self.compression_enabled {
                decompress_data(&cached)?
            } else {
                cached.to_vec()
            };
            
            // Use zero-copy deserialization
            let archived = ZeroCopyAccessor::access::<PreprocessedData>(&decompressed)
                .map_err(|e| NexusError::Deserialization(e.to_string()))?;
            let data = ZeroCopyAccessor::deserialize(archived)
                .map_err(|e| NexusError::Deserialization(e.to_string()))?;
            return Ok(Some(data));
        }
        
        // Check primary backend
        let backend_name = self.primary_backend.read().clone();
        if let Some(backend) = self.backends.get(&backend_name) {
            if let Some(stored) = backend.retrieve(key).await? {
                let decompressed = if self.compression_enabled {
                    decompress_data(&stored)?
                } else {
                    stored
                };
                
                let data: PreprocessedData = serde_json::from_slice(&decompressed)?;
                
                // Update cache
                self.cache.insert(key.to_string(), Bytes::from(stored)).await;
                
                return Ok(Some(data));
            }
        }
        
        Ok(None)
    }
    
    pub fn register_backend(&self, name: &str, backend: Arc<dyn StorageBackend>) {
        self.backends.insert(name.to_string(), backend);
    }
    
    pub fn set_primary_backend(&self, name: &str) -> Result<()> {
        if self.backends.contains_key(name) {
            *self.primary_backend.write() = name.to_string();
            Ok(())
        } else {
            Err(NexusError::NotFound(format!("Backend '{}' not found", name)))
        }
    }
    
    /// Memory-map a file for zero-copy access
    pub async fn mmap_file(&self, path: &str) -> Result<Arc<Mmap>> {
        // Check cache first
        if let Some(cached) = self.mmap_cache.get(path) {
            return Ok(cached.clone());
        }
        
        // Open file and create memory map
        let file = File::open(path)
            .map_err(|e| NexusError::Io(e.to_string()))?;
        let mmap = unsafe { 
            MmapOptions::new()
                .map(&file)
                .map_err(|e| NexusError::Io(e.to_string()))?
        };
        
        let mmap_arc = Arc::new(mmap);
        self.mmap_cache.insert(path.to_string(), mmap_arc.clone());
        
        Ok(mmap_arc)
    }
}

// ===== BACKENDS =====

struct MemoryBackend {
    storage: DashMap<String, Vec<u8>>,
}

impl MemoryBackend {
    fn new() -> Self {
        Self {
            storage: DashMap::new(),
        }
    }
}

#[async_trait]
impl StorageBackend for MemoryBackend {
    async fn store(&self, key: &str, data: &[u8]) -> Result<()> {
        self.storage.insert(key.to_string(), data.to_vec());
        Ok(())
    }
    
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.storage.get(key).map(|v| v.clone()))
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        self.storage.remove(key);
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.storage.contains_key(key))
    }
    
    fn name(&self) -> &str {
        "memory"
    }
}

// ===== HELPERS =====

fn generate_storage_key(request: &crate::core::types::PipelineRequest) -> String {
    format!("storage:{}:{}", request.id, request.timestamp.timestamp())
}

fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    Ok(lz4::block::compress(data, None, false)?)
}

fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    lz4::block::decompress(data, None)
        .map_err(|e| NexusError::Serialization(serde_json::Error::custom(e)))
}