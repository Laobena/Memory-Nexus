//! Local AI Engine using Ollama for embedding generation
//!
//! Provides 100% local AI processing with aggressive caching for <50ms performance.

use super::{AIError, AIResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Request structure for Ollama embedding API
#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// Response structure from Ollama embedding API
#[derive(Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Local AI Engine for embedding generation using Ollama
pub struct LocalAIEngine {
    /// Model name for embeddings
    model: String,
    /// Ollama API base URL
    api_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Embedding cache: content_hash -> embedding
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    /// Cache hit counter
    cache_hits: Arc<Mutex<u64>>,
    /// Cache miss counter
    cache_misses: Arc<Mutex<u64>>,
}

impl LocalAIEngine {
    /// Default embedding model
    const DEFAULT_MODEL: &'static str = "mxbai-embed-large";

    /// Expected embedding dimension for mxbai-embed-large
    const EXPECTED_EMBEDDING_DIM: usize = 1024;

    /// Create new LocalAIEngine with connection testing
    pub async fn new() -> AIResult<Self> {
        let model = std::env::var("OLLAMA_EMBEDDING_MODEL")
            .unwrap_or_else(|_| Self::DEFAULT_MODEL.to_string());
        
        let api_url = std::env::var("OLLAMA_API_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| AIError::ConnectionError(format!("Failed to create HTTP client: {}", e)))?;

        let engine = Self {
            model: model.clone(),
            api_url,
            client,
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
        };

        // Test actual Ollama connection
        engine.test_connection().await?;
        println!("✅ LocalAIEngine initialized - Ollama connection verified");
        println!("🤖 Model: {}", model);

        Ok(engine)
    }

    /// Test connection to Ollama service
    async fn test_connection(&self) -> AIResult<()> {
        println!("🔗 Testing Ollama connection at {}...", self.api_url);
        
        // Check if Ollama is running by hitting the version endpoint
        let version_url = format!("{}/api/version", self.api_url);
        let response = self.client.get(&version_url)
            .send()
            .await
            .map_err(|e| AIError::ConnectionError(format!("Failed to connect to Ollama: {}", e)))?;
        
        if !response.status().is_success() {
            return Err(AIError::ConnectionError(format!(
                "Ollama returned status: {}",
                response.status()
            )));
        }
        
        // Verify model is available by attempting a small embedding
        println!("🤖 Verifying model '{}' availability...", self.model);
        let test_embedding = self.generate_embedding_from_ollama("test").await?;
        
        if test_embedding.len() != Self::EXPECTED_EMBEDDING_DIM {
            return Err(AIError::ModelNotAvailable(format!(
                "Model {} returned unexpected embedding dimension: {} (expected {})",
                self.model, test_embedding.len(), Self::EXPECTED_EMBEDDING_DIM
            )));
        }
        
        Ok(())
    }

    /// Generate embedding for given content with caching
    pub async fn generate_embedding(&self, content: &str) -> AIResult<Vec<f32>> {
        let start_time = Instant::now();

        // Generate content hash for caching
        let content_hash = self.hash_content(content);

        // Check cache first
        if let Some(embedding) = self.get_cached_embedding(&content_hash) {
            *self.cache_hits.lock().unwrap() += 1;
            let elapsed = start_time.elapsed();
            println!(
                "🚀 Cache HIT - embedding retrieved in {}μs",
                elapsed.as_micros()
            );
            return Ok(embedding);
        }

        // Generate new embedding
        *self.cache_misses.lock().unwrap() += 1;
        let embedding = self.generate_embedding_from_ollama(content).await?;

        // Cache the result
        self.cache_embedding(content_hash, embedding.clone());

        let elapsed = start_time.elapsed();
        println!(
            "🔸 Cache MISS - embedding generated in {}ms",
            elapsed.as_millis()
        );

        // Verify performance target
        if elapsed.as_millis() > 50 {
            eprintln!(
                "⚠️  Performance target missed: {}ms > 50ms",
                elapsed.as_millis()
            );
        }

        // Verify embedding dimension
        if embedding.len() != Self::EXPECTED_EMBEDDING_DIM {
            return Err(AIError::EmbeddingError(format!(
                "Unexpected embedding dimension: {} (expected {})",
                embedding.len(),
                Self::EXPECTED_EMBEDDING_DIM
            )));
        }

        Ok(embedding)
    }

    /// Generate embedding from Ollama API
    async fn generate_embedding_from_ollama(&self, content: &str) -> AIResult<Vec<f32>> {
        let url = format!("{}/api/embed", self.api_url);
        
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: vec![content.to_string()],
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AIError::HttpError(format!("Failed to send request: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AIError::HttpError(format!(
                "Ollama API error ({}): {}",
                status, error_text
            )));
        }
        
        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| AIError::InvalidResponse(format!("Failed to parse response: {}", e)))?;
        
        // Get the first (and only) embedding from the response
        let embedding = embedding_response.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| AIError::InvalidResponse("No embeddings in response".to_string()))?;
        
        Ok(embedding)
    }

    /// Generate hash for content (simple but effective for caching)
    fn hash_content(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get cached embedding if available
    fn get_cached_embedding(&self, content_hash: &str) -> Option<Vec<f32>> {
        self.embedding_cache
            .lock()
            .unwrap()
            .get(content_hash)
            .cloned()
    }

    /// Cache embedding for future use
    fn cache_embedding(&self, content_hash: String, embedding: Vec<f32>) {
        let mut cache = self.embedding_cache.lock().unwrap();

        // Simple eviction: if cache gets too large, clear it
        if cache.len() > 1000 {
            cache.clear();
            println!("🧹 Embedding cache cleared (exceeded 1000 entries)");
        }

        cache.insert(content_hash, embedding);
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (u64, u64, f64) {
        let hits = *self.cache_hits.lock().unwrap();
        let misses = *self.cache_misses.lock().unwrap();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        (hits, misses, hit_rate)
    }

    /// Get cache size
    pub fn get_cache_size(&self) -> usize {
        self.embedding_cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_connection() {
        let engine = LocalAIEngine::new().await;
        match engine {
            Ok(_) => println!("✅ Ollama connection test passed"),
            Err(e) => {
                println!("⚠️ Ollama connection test skipped: {}", e);
                println!("   Make sure Ollama is running: ollama serve");
            }
        }
    }

    #[tokio::test]
    async fn test_embedding_generation() {
        let engine = match LocalAIEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("⚠️ Skipping test - Ollama not available");
                return;
            }
        };

        let content = "Hello, AI world!";
        let embedding = engine
            .generate_embedding(content)
            .await
            .expect("Failed to generate embedding");

        assert_eq!(embedding.len(), LocalAIEngine::EXPECTED_EMBEDDING_DIM);
        println!("✅ Generated embedding with {} dimensions", embedding.len());
    }

    #[tokio::test]
    async fn test_embedding_caching() {
        let engine = match LocalAIEngine::new().await {
            Ok(e) => e,
            Err(_) => {
                println!("⚠️ Skipping test - Ollama not available");
                return;
            }
        };

        let content = "Test caching functionality";

        // First call - should be cache miss
        let start = Instant::now();
        let embedding1 = engine
            .generate_embedding(content)
            .await
            .expect("Failed to generate embedding");
        let first_duration = start.elapsed();

        // Second call - should be cache hit
        let start = Instant::now();
        let embedding2 = engine
            .generate_embedding(content)
            .await
            .expect("Failed to generate embedding");
        let second_duration = start.elapsed();

        // Embeddings should be identical
        assert_eq!(embedding1, embedding2);

        // Second call should be much faster
        assert!(second_duration < first_duration);

        let (hits, misses, hit_rate) = engine.get_cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 50.0);

        println!("✅ Cache test passed - hit rate: {:.1}%", hit_rate);
    }
}

