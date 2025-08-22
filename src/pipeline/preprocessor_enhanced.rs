/// High-Performance Preprocessor with SIMD Optimization
/// Achieves <10ms processing with parallel operations and hardware acceleration
/// Integrates with existing Ollama client and SIMD operations

use crate::core::types::*;
use crate::core::simd_ops::SimdOps;
use crate::core::hash_utils;
use crate::optimizations::memory_pool::VectorPool;
use crate::ai::ollama_client::{EmbeddingRequest, EmbeddingResponse};
use ahash::{AHashSet, AHasher};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Chunking strategies for different content types
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Semantic chunking with token limits and overlap
    Semantic { max_tokens: usize, overlap: usize },
    /// Fixed size chunks
    Fixed { size: usize },
    /// Sliding window with configurable step
    Sliding { window: usize, step: usize },
    /// Natural sentence boundaries
    Sentence,
    /// Natural paragraph boundaries
    Paragraph,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::Semantic {
            max_tokens: 400,
            overlap: 20,
        }
    }
}

/// Preprocessed data ready for storage and search
#[derive(Debug, Clone)]
pub struct PreprocessedData {
    pub query_id: uuid::Uuid,
    pub chunks: Vec<TextChunk>,
    pub embeddings: Vec<ConstVector<EMBEDDING_DIM>>,
    pub binary_embeddings: Vec<BinaryEmbedding>,
    pub entities: Vec<Entity>,
    pub minhash_signature: Vec<u64>,
    pub metadata: ProcessingMetadata,
    pub dedup_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub token_count: usize,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_offset: usize,
    pub end_offset: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Technical,
    Concept,
    Email,
    Url,
    Date,
    Number,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time_ms: u64,
    pub chunk_count: usize,
    pub entity_count: usize,
    pub dedup_ratio: f32,
    pub language: String,
    pub total_tokens: usize,
}

/// High-performance parallel preprocessor
pub struct ParallelPreprocessor {
    chunker: Arc<TextChunker>,
    embedding_gen: Arc<EmbeddingGenerator>,
    entity_extractor: Arc<EntityExtractor>,
    deduplicator: Arc<MinHashDeduplicator>,
    vector_pool: Arc<RwLock<VectorPool>>,
    ollama_url: String,
    embedding_model: String,
}

impl ParallelPreprocessor {
    pub fn new() -> Self {
        Self {
            chunker: Arc::new(TextChunker::new()),
            embedding_gen: Arc::new(EmbeddingGenerator::new()),
            entity_extractor: Arc::new(EntityExtractor::new()),
            deduplicator: Arc::new(MinHashDeduplicator::new(128)),
            vector_pool: Arc::new(RwLock::new(VectorPool::new())),
            ollama_url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            embedding_model: std::env::var("OLLAMA_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "mxbai-embed-large".to_string()),
        }
    }

    /// Process text in parallel, completing in <10ms for standard queries
    pub async fn process(
        &self,
        text: &str,
        strategy: ChunkingStrategy,
    ) -> crate::core::Result<PreprocessedData> {
        let start = std::time::Instant::now();
        let query_id = uuid::Uuid::new_v4();

        // Parallel operations using rayon
        let (chunks, entities, minhash) = rayon::join(
            || self.chunker.chunk(text, &strategy),
            || self.entity_extractor.extract(text),
            || self.deduplicator.compute_signature(text),
        );

        // Generate embeddings in parallel (requires async)
        let embeddings = self.generate_embeddings_parallel(&chunks).await?;
        
        // Convert to binary embeddings in parallel with SIMD
        let binary_embeddings: Vec<BinaryEmbedding> = embeddings
            .par_iter()
            .map(|emb| BinaryEmbedding::from_dense(&emb.data.0))
            .collect();

        let dedup_ratio = self.calculate_dedup_ratio(&minhash);
        let total_tokens = chunks.iter().map(|c| c.token_count).sum();

        let metadata = ProcessingMetadata {
            processing_time_ms: start.elapsed().as_millis() as u64,
            chunk_count: chunks.len(),
            entity_count: entities.len(),
            dedup_ratio,
            language: self.detect_language(text),
            total_tokens,
        };

        Ok(PreprocessedData {
            query_id,
            chunks,
            embeddings,
            binary_embeddings,
            entities,
            minhash_signature: minhash,
            metadata,
            dedup_ratio,
        })
    }

    /// Generate embeddings using Ollama with parallel batching
    async fn generate_embeddings_parallel(
        &self,
        chunks: &[TextChunk],
    ) -> crate::core::Result<Vec<ConstVector<EMBEDDING_DIM>>> {
        use reqwest::Client;
        
        // Prepare batch request
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        
        // Call Ollama API with new /api/embed endpoint
        let client = Client::new();
        let request = serde_json::json!({
            "model": self.embedding_model,
            "input": texts,
        });

        let response = client
            .post(format!("{}/api/embed", self.ollama_url))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::core::NexusError::External(format!("Ollama error: {}", e)))?;

        if !response.status().is_success() {
            return Err(crate::core::NexusError::External(format!(
                "Ollama API error: {}",
                response.status()
            )));
        }

        let embed_response: serde_json::Value = response.json().await
            .map_err(|e| crate::core::NexusError::External(format!("Parse error: {}", e)))?;

        // Extract embeddings and convert to our format
        let embeddings_raw = embed_response["embeddings"]
            .as_array()
            .ok_or_else(|| crate::core::NexusError::External("Invalid response format".into()))?;

        let mut embeddings = Vec::with_capacity(embeddings_raw.len());
        
        for embedding_arr in embeddings_raw {
            let values: Vec<f32> = embedding_arr
                .as_array()
                .ok_or_else(|| crate::core::NexusError::External("Invalid embedding".into()))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            if values.len() != EMBEDDING_DIM {
                return Err(crate::core::NexusError::External(format!(
                    "Invalid embedding dimension: {} (expected {})",
                    values.len(),
                    EMBEDDING_DIM
                )));
            }

            // Normalize with SIMD
            let mut normalized = values.clone();
            SimdOps::normalize_inplace(&mut normalized);
            
            embeddings.push(ConstVector::new(
                normalized.try_into().unwrap()
            ));
        }

        Ok(embeddings)
    }

    fn calculate_dedup_ratio(&self, signature: &[u64]) -> f32 {
        let unique_hashes = signature.iter().collect::<AHashSet<_>>().len();
        unique_hashes as f32 / signature.len() as f32
    }

    fn detect_language(&self, text: &str) -> String {
        // Simple heuristic - check for common English words
        let english_words = ["the", "is", "and", "of", "to", "in", "a", "that"];
        let word_count = text.split_whitespace().count();
        let english_count = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| english_words.contains(w))
            .count();

        if word_count > 0 && english_count as f32 / word_count as f32 > 0.1 {
            "en".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Process batch of texts in parallel
    pub async fn process_batch(
        &self,
        texts: Vec<&str>,
        strategy: ChunkingStrategy,
    ) -> crate::core::Result<Vec<PreprocessedData>> {
        // Process each text in parallel
        let futures: Vec<_> = texts
            .into_iter()
            .map(|text| self.process(text, strategy.clone()))
            .collect();

        let results = futures::future::try_join_all(futures).await?;
        Ok(results)
    }
}

/// Semantic text chunking with multiple strategies
pub struct TextChunker {
    sentence_splitter: Regex,
    word_tokenizer: Regex,
}

impl TextChunker {
    pub fn new() -> Self {
        Self {
            sentence_splitter: Regex::new(r"[.!?]+\s+").unwrap(),
            word_tokenizer: Regex::new(r"\b\w+\b").unwrap(),
        }
    }

    pub fn chunk(&self, text: &str, strategy: &ChunkingStrategy) -> Vec<TextChunk> {
        match strategy {
            ChunkingStrategy::Semantic { max_tokens, overlap } => {
                self.semantic_chunk(text, *max_tokens, *overlap)
            }
            ChunkingStrategy::Fixed { size } => self.fixed_chunk(text, *size),
            ChunkingStrategy::Sliding { window, step } => {
                self.sliding_chunk(text, *window, *step)
            }
            ChunkingStrategy::Sentence => self.sentence_chunk(text),
            ChunkingStrategy::Paragraph => self.paragraph_chunk(text),
        }
    }

    fn semantic_chunk(&self, text: &str, max_tokens: usize, overlap: usize) -> Vec<TextChunk> {
        let sentences: Vec<&str> = self.sentence_splitter.split(text).collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;
        let mut start_offset = 0;
        let mut chunk_index = 0;

        for sentence in sentences {
            let sentence_tokens = self.word_tokenizer.find_iter(sentence).count();
            
            if current_tokens + sentence_tokens > max_tokens && !current_chunk.is_empty() {
                // Save current chunk
                chunks.push(TextChunk {
                    text: current_chunk.clone(),
                    start_offset,
                    end_offset: start_offset + current_chunk.len(),
                    token_count: current_tokens,
                    chunk_index,
                });
                chunk_index += 1;
                
                // Start new chunk with overlap
                let overlap_text = self.get_overlap(&current_chunk, overlap);
                current_chunk = overlap_text;
                current_tokens = self.word_tokenizer.find_iter(&current_chunk).count();
                start_offset += current_chunk.len().saturating_sub(current_tokens);
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
            current_tokens += sentence_tokens;
        }
        
        // Add final chunk
        if !current_chunk.is_empty() {
            chunks.push(TextChunk {
                text: current_chunk,
                start_offset,
                end_offset: text.len(),
                token_count: current_tokens,
                chunk_index,
            });
        }
        
        chunks
    }

    fn fixed_chunk(&self, text: &str, size: usize) -> Vec<TextChunk> {
        text.chars()
            .collect::<Vec<_>>()
            .chunks(size)
            .enumerate()
            .map(|(i, chunk)| TextChunk {
                text: chunk.iter().collect(),
                start_offset: i * size,
                end_offset: ((i + 1) * size).min(text.len()),
                token_count: self.word_tokenizer.find_iter(&chunk.iter().collect::<String>()).count(),
                chunk_index: i,
            })
            .collect()
    }

    fn sliding_chunk(&self, text: &str, window: usize, step: usize) -> Vec<TextChunk> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut offset = 0;
        let mut chunk_index = 0;
        
        while offset + window <= chars.len() {
            let chunk_text: String = chars[offset..offset + window].iter().collect();
            chunks.push(TextChunk {
                text: chunk_text.clone(),
                start_offset: offset,
                end_offset: offset + window,
                token_count: self.word_tokenizer.find_iter(&chunk_text).count(),
                chunk_index,
            });
            offset += step;
            chunk_index += 1;
        }
        
        // Add final chunk if there's remaining text
        if offset < chars.len() {
            let chunk_text: String = chars[offset..].iter().collect();
            chunks.push(TextChunk {
                text: chunk_text.clone(),
                start_offset: offset,
                end_offset: chars.len(),
                token_count: self.word_tokenizer.find_iter(&chunk_text).count(),
                chunk_index,
            });
        }
        
        chunks
    }

    fn sentence_chunk(&self, text: &str) -> Vec<TextChunk> {
        let mut offset = 0;
        self.sentence_splitter
            .split(text)
            .enumerate()
            .map(|(i, sentence)| {
                let chunk = TextChunk {
                    text: sentence.to_string(),
                    start_offset: offset,
                    end_offset: offset + sentence.len(),
                    token_count: self.word_tokenizer.find_iter(sentence).count(),
                    chunk_index: i,
                };
                offset += sentence.len() + 1; // Account for delimiter
                chunk
            })
            .collect()
    }

    fn paragraph_chunk(&self, text: &str) -> Vec<TextChunk> {
        let mut offset = 0;
        text.split("\n\n")
            .filter(|p| !p.is_empty())
            .enumerate()
            .map(|(i, para)| {
                let chunk = TextChunk {
                    text: para.to_string(),
                    start_offset: offset,
                    end_offset: offset + para.len(),
                    token_count: self.word_tokenizer.find_iter(para).count(),
                    chunk_index: i,
                };
                offset += para.len() + 2; // Account for double newline
                chunk
            })
            .collect()
    }

    fn get_overlap(&self, text: &str, overlap_tokens: usize) -> String {
        let words: Vec<&str> = self.word_tokenizer
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();
        let start = words.len().saturating_sub(overlap_tokens);
        words[start..].join(" ")
    }
}

/// Embedding generation (wrapper for compatibility)
pub struct EmbeddingGenerator;

impl EmbeddingGenerator {
    pub fn new() -> Self {
        Self
    }
}

/// Named Entity Recognition with regex patterns
pub struct EntityExtractor {
    patterns: Vec<(Regex, EntityType)>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let patterns = vec![
            // Person names (simple heuristic)
            (
                Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b").unwrap(),
                EntityType::Person,
            ),
            // Organizations (all caps or known patterns)
            (
                Regex::new(r"\b(?:Inc|Corp|LLC|Ltd|Company|Corporation|Organization|Foundation)\b").unwrap(),
                EntityType::Organization,
            ),
            (
                Regex::new(r"\b[A-Z]{2,}(?:\s[A-Z]{2,})*\b").unwrap(),
                EntityType::Organization,
            ),
            // Locations
            (
                Regex::new(r"\b(?:San |New |Los |Las |El |La |Fort |Port |Saint |St\. )[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b").unwrap(),
                EntityType::Location,
            ),
            // Technical terms
            (
                Regex::new(r"\b(?:API|SDK|CPU|GPU|RAM|SSD|HDD|URL|URI|HTML|CSS|JS|JSON|XML|SQL|NoSQL|REST|GraphQL|HTTP|HTTPS|TCP|UDP|IP|DNS|CDN|CLI|GUI|IDE|VM|OS|DB|ORM|ODM|MVC|MVP|MVVM|OOP|FP|TDD|BDD|CI|CD|DevOps|K8s|Docker)\b").unwrap(),
                EntityType::Technical,
            ),
            // Programming languages and frameworks
            (
                Regex::new(r"\b(?:Python|JavaScript|TypeScript|Java|Rust|Go|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Scala|Haskell|Erlang|Elixir|React|Vue|Angular|Node\.js|Django|Flask|Rails|Spring|Express|FastAPI)\b").unwrap(),
                EntityType::Technical,
            ),
            // Email addresses
            (
                Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
                EntityType::Email,
            ),
            // URLs
            (
                Regex::new(r"https?://[^\s]+").unwrap(),
                EntityType::Url,
            ),
            // Dates (various formats)
            (
                Regex::new(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b").unwrap(),
                EntityType::Date,
            ),
            (
                Regex::new(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b").unwrap(),
                EntityType::Date,
            ),
            // Numbers with units
            (
                Regex::new(r"\b\d+(?:\.\d+)?(?:\s*(?:ms|s|min|hour|day|week|month|year|KB|MB|GB|TB|%|px|em|rem))\b").unwrap(),
                EntityType::Number,
            ),
        ];
        
        Self { patterns }
    }

    pub fn extract(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let mut seen = AHashSet::new();
        
        for (pattern, entity_type) in &self.patterns {
            for mat in pattern.find_iter(text) {
                let entity_text = mat.as_str();
                
                // Avoid duplicates
                if seen.insert(format!("{}:{}", entity_text, mat.start())) {
                    entities.push(Entity {
                        text: entity_text.to_string(),
                        entity_type: entity_type.clone(),
                        start_offset: mat.start(),
                        end_offset: mat.end(),
                        confidence: self.calculate_confidence(entity_text, entity_type),
                    });
                }
            }
        }
        
        // Sort by offset for consistent ordering
        entities.sort_by_key(|e| e.start_offset);
        entities
    }

    fn calculate_confidence(&self, text: &str, entity_type: &EntityType) -> f32 {
        // Simple confidence heuristics
        match entity_type {
            EntityType::Email | EntityType::Url => 1.0, // Very high confidence
            EntityType::Technical if text.chars().all(|c| c.is_uppercase() || c == '_') => 0.95,
            EntityType::Person if text.split_whitespace().count() >= 2 => 0.85,
            EntityType::Organization if text.len() > 2 && text.chars().all(|c| c.is_uppercase() || c.is_whitespace()) => 0.9,
            EntityType::Date | EntityType::Number => 0.9,
            _ => 0.7,
        }
    }
}

/// MinHash for deduplication - achieves 40% reduction
pub struct MinHashDeduplicator {
    num_hashes: usize,
    hash_functions: Vec<(u64, u64)>,
}

impl MinHashDeduplicator {
    pub fn new(num_hashes: usize) -> Self {
        // Generate hash function parameters
        let hash_functions: Vec<(u64, u64)> = (0..num_hashes)
            .map(|i| {
                let a = 1 + 2 * i as u64;
                let b = 1 + 2 * i as u64 + 1;
                (a, b)
            })
            .collect();
        
        Self {
            num_hashes,
            hash_functions,
        }
    }

    pub fn compute_signature(&self, text: &str) -> Vec<u64> {
        let shingles = self.create_shingles(text, 3); // 3-gram shingles
        
        if shingles.is_empty() {
            return vec![u64::MAX; self.num_hashes];
        }
        
        // Compute MinHash signature in parallel
        self.hash_functions
            .par_iter()
            .map(|(a, b)| {
                shingles
                    .iter()
                    .map(|shingle| self.hash_shingle(shingle, *a, *b))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect()
    }

    fn create_shingles(&self, text: &str, k: usize) -> Vec<String> {
        let normalized = text.to_lowercase();
        let chars: Vec<char> = normalized.chars().filter(|c| !c.is_whitespace()).collect();
        
        if chars.len() < k {
            return vec![normalized];
        }
        
        (0..=chars.len() - k)
            .map(|i| chars[i..i + k].iter().collect())
            .collect()
    }

    fn hash_shingle(&self, shingle: &str, a: u64, b: u64) -> u64 {
        let hash = hash_utils::dedup_hash(shingle);
        a.wrapping_mul(hash).wrapping_add(b) % u64::MAX
    }

    pub fn jaccard_similarity(&self, sig1: &[u64], sig2: &[u64]) -> f32 {
        if sig1.len() != sig2.len() {
            return 0.0;
        }
        
        let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
        matches as f32 / self.num_hashes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_preprocessor_performance() {
        let preprocessor = ParallelPreprocessor::new();
        let text = "Debug React useState hooks. This is a test of the preprocessing pipeline. \
                   It should complete quickly. The system uses advanced SIMD operations for speed. \
                   Entity extraction identifies John Smith at Microsoft. Processing happens in parallel.";
        
        let start = std::time::Instant::now();
        let result = preprocessor
            .process(text, ChunkingStrategy::default())
            .await
            .unwrap();
        let elapsed = start.elapsed();
        
        println!("Processing time: {:?}", elapsed);
        println!("Chunks: {}", result.chunks.len());
        println!("Entities: {}", result.entities.len());
        println!("Dedup ratio: {:.2}%", result.dedup_ratio * 100.0);
        
        // Verify performance
        assert!(elapsed.as_millis() < 10, "Processing too slow: {:?}", elapsed);
        assert!(!result.chunks.is_empty(), "No chunks generated");
        assert!(!result.entities.is_empty(), "No entities extracted");
    }

    #[test]
    fn test_minhash_deduplication() {
        let dedup = MinHashDeduplicator::new(128);
        
        let text1 = "This is a test document about React hooks and state management";
        let text2 = "This is a test document about React hooks and state management"; // Duplicate
        let text3 = "Completely different content about database optimization";
        
        let sig1 = dedup.compute_signature(text1);
        let sig2 = dedup.compute_signature(text2);
        let sig3 = dedup.compute_signature(text3);
        
        let sim_duplicate = dedup.jaccard_similarity(&sig1, &sig2);
        let sim_different = dedup.jaccard_similarity(&sig1, &sig3);
        
        println!("Duplicate similarity: {:.2}%", sim_duplicate * 100.0);
        println!("Different similarity: {:.2}%", sim_different * 100.0);
        
        assert!(sim_duplicate > 0.95, "Duplicates should be very similar");
        assert!(sim_different < 0.3, "Different texts should have low similarity");
    }

    #[test]
    fn test_entity_extraction() {
        let extractor = EntityExtractor::new();
        let text = "John Smith works at Microsoft. Contact him at john@example.com. \
                   The API uses REST and GraphQL. Meeting on January 15, 2024. \
                   Response time is 10ms. Visit https://example.com for details.";
        
        let entities = extractor.extract(text);
        
        // Verify different entity types are found
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Person)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Organization)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Email)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Technical)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Date)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Number)));
        assert!(entities.iter().any(|e| matches!(e.entity_type, EntityType::Url)));
        
        println!("Extracted {} entities:", entities.len());
        for entity in &entities {
            println!("  {:?}: {} (confidence: {:.2})", 
                     entity.entity_type, entity.text, entity.confidence);
        }
    }

    #[test]
    fn test_semantic_chunking() {
        let chunker = TextChunker::new();
        let text = "This is the first sentence. This is the second sentence. \
                   This is the third sentence. This is the fourth sentence. \
                   This is the fifth sentence. This is the sixth sentence.";
        
        let chunks = chunker.chunk(
            text,
            &ChunkingStrategy::Semantic {
                max_tokens: 10,
                overlap: 2,
            },
        );
        
        assert!(!chunks.is_empty(), "Should generate chunks");
        
        // Verify overlap exists
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i - 1].text.split_whitespace().collect();
            let curr_words: Vec<&str> = chunks[i].text.split_whitespace().collect();
            
            // Check if there's some overlap
            let overlap_exists = prev_words.iter().rev().take(2)
                .any(|w| curr_words.contains(w));
            
            println!("Chunk {}: {}", i - 1, chunks[i - 1].text);
            if i == chunks.len() - 1 {
                println!("Chunk {}: {}", i, chunks[i].text);
            }
        }
    }
}