//! Embedding service for vector generation using Ollama
//!
//! This module provides high-performance embedding generation with:
//! - Ollama HTTP API integration
//! - Batch processing for efficiency
//! - Caching to reduce API calls
//! - Connection pooling and retries

use crate::core::{ConstVector, Result as CoreResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Embedding dimension for mxbai-embed-large model
pub const EMBEDDING_DIM: usize = 1024;

/// Configuration for embedding service
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// Ollama API endpoint
    pub ollama_url: String,
    /// Model name (e.g., "mxbai-embed-large")
    pub model: String,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Max cache size
    pub max_cache_size: usize,
    /// Connection timeout
    pub timeout_secs: u64,
    /// Max retries on failure
    pub max_retries: u32,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            model: "mxbai-embed-large".to_string(),
            cache_ttl_secs: 3600,  // 1 hour
            max_cache_size: 10000,
            timeout_secs: 30,
            max_retries: 3,
            batch_size: 32,
        }
    }
}

/// Ollama API request structure
#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

/// Ollama API response structure
#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f64>,  // Ollama returns f64
}

/// Cached embedding entry
struct CachedEmbedding {
    embedding: Vec<f32>,
    timestamp: Instant,
}

/// High-performance embedding service
pub struct EmbeddingService {
    config: EmbeddingConfig,
    client: reqwest::Client,
    cache: Arc<DashMap<String, CachedEmbedding>>,
    stats: Arc<RwLock<EmbeddingStats>>,
}

/// Statistics for monitoring
#[derive(Default)]
struct EmbeddingStats {
    total_requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    api_calls: u64,
    api_errors: u64,
    total_latency_ms: u64,
}

impl EmbeddingService {
    /// Create new embedding service
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        }
    }

    /// Initialize and verify connection to Ollama
    pub async fn initialize(&self) -> CoreResult<()> {
        info!("Initializing embedding service with model: {}", self.config.model);
        
        // Test connection to Ollama
        self.verify_connection().await?;
        
        // Test embedding generation
        let test_embedding = self.generate_embedding("test").await?;
        if test_embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "Model {} returned wrong dimension: {} (expected {})",
                self.config.model,
                test_embedding.len(),
                EMBEDDING_DIM
            ).into());
        }
        
        info!("Embedding service initialized successfully");
        Ok(())
    }

    /// Verify Ollama connection and model availability
    async fn verify_connection(&self) -> CoreResult<()> {
        let url = format!("{}/api/tags", self.config.ollama_url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to connect to Ollama: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Ollama responded with status: {}", response.status()).into());
        }
        
        let text = response.text().await
            .map_err(|e| format!("Failed to read response: {}", e))?;
        
        if !text.contains(&self.config.model) {
            warn!("Model {} not found. Please run: ollama pull {}", 
                  self.config.model, self.config.model);
        }
        
        Ok(())
    }

    /// Generate embedding for a single text
    pub async fn generate_embedding(&self, text: &str) -> CoreResult<Vec<f32>> {
        let start = Instant::now();
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }
        
        // Check cache first
        let cache_key = self.cache_key(text);
        if let Some(cached) = self.get_cached(&cache_key).await {
            let mut stats = self.stats.write().await;
            stats.cache_hits += 1;
            stats.total_latency_ms += start.elapsed().as_millis() as u64;
            return Ok(cached);
        }
        
        // Cache miss - generate new embedding
        {
            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        }
        
        // Call Ollama API with retries
        let embedding = self.call_ollama_with_retry(text).await?;
        
        // Cache the result
        self.cache_embedding(cache_key, embedding.clone()).await;
        
        // Update latency stats
        {
            let mut stats = self.stats.write().await;
            stats.total_latency_ms += start.elapsed().as_millis() as u64;
        }
        
        Ok(embedding)
    }

    /// Generate embeddings for a batch of texts
    pub async fn generate_batch(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        
        // Process in parallel batches
        for chunk in texts.chunks(self.config.batch_size) {
            let mut handles = Vec::new();
            
            for text in chunk {
                let service = self.clone();
                let text = text.clone();
                
                let handle = tokio::spawn(async move {
                    service.generate_embedding(&text).await
                });
                
                handles.push(handle);
            }
            
            // Collect results
            for handle in handles {
                let embedding = handle.await
                    .map_err(|e| format!("Task failed: {}", e))?
                    .map_err(|e| format!("Embedding failed: {}", e))?;
                embeddings.push(embedding);
            }
        }
        
        Ok(embeddings)
    }

    /// Generate embedding as ConstVector for SIMD operations
    pub async fn generate_const_vector(&self, text: &str) -> CoreResult<ConstVector<EMBEDDING_DIM>> {
        let embedding = self.generate_embedding(text).await?;
        
        let mut array = [0.0f32; EMBEDDING_DIM];
        array.copy_from_slice(&embedding);
        
        Ok(ConstVector::new(array))
    }

    /// Call Ollama API with retry logic
    async fn call_ollama_with_retry(&self, text: &str) -> CoreResult<Vec<f32>> {
        let mut last_error = None;
        
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(100 * (1 << attempt))).await;
            }
            
            match self.call_ollama_api(text).await {
                Ok(embedding) => {
                    let mut stats = self.stats.write().await;
                    stats.api_calls += 1;
                    return Ok(embedding);
                }
                Err(e) => {
                    warn!("Ollama API call failed (attempt {}/{}): {}", 
                          attempt + 1, self.config.max_retries, e);
                    last_error = Some(e);
                }
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.api_errors += 1;
        
        Err(last_error.unwrap_or_else(|| "All retry attempts failed".into()))
    }

    /// Direct Ollama API call
    async fn call_ollama_api(&self, text: &str) -> CoreResult<Vec<f32>> {
        let request = EmbeddingRequest {
            model: self.config.model.clone(),
            prompt: text.to_string(),
        };
        
        let url = format!("{}/api/embeddings", self.config.ollama_url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Ollama API error {}: {}", status, error_text).into());
        }
        
        let embedding_response: EmbeddingResponse = response.json().await
            .map_err(|e| format!("Failed to parse response: {}", e))?;
        
        // Convert f64 to f32
        let embedding: Vec<f32> = embedding_response.embedding
            .into_iter()
            .map(|x| x as f32)
            .collect();
        
        // Verify dimension
        if embedding.len() != EMBEDDING_DIM {
            return Err(format!(
                "Unexpected embedding dimension: {} (expected {})",
                embedding.len(),
                EMBEDDING_DIM
            ).into());
        }
        
        Ok(embedding)
    }

    /// Generate cache key for text
    fn cache_key(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.config.model.hash(&mut hasher);
        
        format!("embed_{}_{}", self.config.model, hasher.finish())
    }

    /// Get cached embedding if available and not expired
    async fn get_cached(&self, key: &str) -> Option<Vec<f32>> {
        let entry = self.cache.get(key)?;
        
        // Check if expired
        if entry.timestamp.elapsed().as_secs() > self.config.cache_ttl_secs {
            drop(entry);  // Release lock before removing
            self.cache.remove(key);
            return None;
        }
        
        Some(entry.embedding.clone())
    }

    /// Cache an embedding
    async fn cache_embedding(&self, key: String, embedding: Vec<f32>) {
        // Evict old entries if cache is full
        if self.cache.len() >= self.config.max_cache_size {
            self.evict_old_entries().await;
        }
        
        self.cache.insert(key, CachedEmbedding {
            embedding,
            timestamp: Instant::now(),
        });
    }

    /// Evict old cache entries
    async fn evict_old_entries(&self) {
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        
        // Collect keys to remove
        let keys_to_remove: Vec<String> = self.cache
            .iter()
            .filter(|entry| now.duration_since(entry.value().timestamp) > ttl)
            .map(|entry| entry.key().clone())
            .collect();
        
        // Remove expired entries
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
        
        // If still over capacity, remove oldest entries
        if self.cache.len() >= self.config.max_cache_size {
            let mut entries: Vec<(String, Instant)> = self.cache
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().timestamp))
                .collect();
            
            entries.sort_by_key(|e| e.1);
            
            let to_remove = entries.len() - (self.config.max_cache_size * 9 / 10);  // Keep 90%
            for (key, _) in entries.into_iter().take(to_remove) {
                self.cache.remove(&key);
            }
        }
    }

    /// Get service statistics
    pub async fn stats(&self) -> String {
        let stats = self.stats.read().await;
        
        let hit_rate = if stats.total_requests > 0 {
            (stats.cache_hits as f64 / stats.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        let avg_latency = if stats.total_requests > 0 {
            stats.total_latency_ms as f64 / stats.total_requests as f64
        } else {
            0.0
        };
        
        format!(
            "Embedding Service Stats:\n\
             Total Requests: {}\n\
             Cache Hits: {} ({:.1}%)\n\
             Cache Misses: {}\n\
             API Calls: {}\n\
             API Errors: {}\n\
             Avg Latency: {:.2}ms\n\
             Cache Size: {}",
            stats.total_requests,
            stats.cache_hits,
            hit_rate,
            stats.cache_misses,
            stats.api_calls,
            stats.api_errors,
            avg_latency,
            self.cache.len()
        )
    }

    /// Clear the cache
    pub async fn clear_cache(&self) {
        self.cache.clear();
        info!("Embedding cache cleared");
    }
}

impl Clone for EmbeddingService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            cache: self.cache.clone(),
            stats: self.stats.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let config = EmbeddingConfig::default();
        let service = EmbeddingService::new(config);
        
        let key1 = service.cache_key("test text");
        let key2 = service.cache_key("test text");
        let key3 = service.cache_key("different text");
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = EmbeddingConfig {
            cache_ttl_secs: 1,
            ..Default::default()
        };
        let service = EmbeddingService::new(config);
        
        let key = "test_key".to_string();
        let embedding = vec![0.1, 0.2, 0.3];
        
        // Cache embedding
        service.cache_embedding(key.clone(), embedding.clone()).await;
        
        // Should be in cache
        assert!(service.get_cached(&key).await.is_some());
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Should be expired
        assert!(service.get_cached(&key).await.is_none());
    }
}