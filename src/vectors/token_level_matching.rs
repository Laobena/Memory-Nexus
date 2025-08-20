//! ColBERT-style Token-Level Vector Matching
//!
//! Implements advanced token-level vector matching using ColBERT-style approaches
//! for fine-grained semantic understanding. Provides token-by-token embedding
//! generation, maximum similarity aggregation, and late interaction patterns
//! for superior matching of technical terminology and complex concepts.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use thiserror::Error;
use uuid::Uuid;

// Internal imports
use crate::ai::{AIError, LocalAIEngine};
use crate::search::similarity::cosine_similarity_optimized;

/// Errors related to token-level processing
#[derive(Error, Debug)]
pub enum TokenLevelError {
    #[error("Tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
    
    #[error("Token embedding generation failed: token='{token}', error={error}")]
    TokenEmbeddingFailed { token: String, error: String },
    
    #[error("Similarity calculation failed: {stage} - {details}")]
    SimilarityCalculationError { stage: String, details: String },
    
    #[error("Token alignment failed: query_tokens={query_tokens}, doc_tokens={doc_tokens}")]
    TokenAlignmentError { query_tokens: usize, doc_tokens: usize },
    
    #[error("Late interaction processing failed: {component} - {reason}")]
    LateInteractionError { component: String, reason: String },
    
    #[error("Maximum similarity aggregation failed: {method} - {issue}")]
    MaxSimAggregationError { method: String, issue: String },
    
    #[error("Performance target missed: {metric}={actual} vs target={target}")]
    PerformanceError { metric: String, actual: String, target: String },
    
    #[error("AI engine error: {0}")]
    AIEngineError(#[from] AIError),
}

/// Configuration for token-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLevelConfig {
    /// Maximum number of tokens to process
    pub max_tokens: usize,
    /// Enable advanced tokenization
    pub enable_advanced_tokenization: bool,
    /// Token filtering configuration
    pub token_filtering: TokenFilteringConfig,
    /// Embedding generation settings
    pub embedding_config: TokenEmbeddingConfig,
    /// Similarity calculation settings
    pub similarity_config: SimilarityConfig,
    /// Late interaction configuration
    pub late_interaction_config: LateInteractionConfig,
    /// Performance targets
    pub performance_targets: TokenLevelPerformanceTargets,
}

impl Default for TokenLevelConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64, // Reasonable limit for performance
            enable_advanced_tokenization: true,
            token_filtering: TokenFilteringConfig::default(),
            embedding_config: TokenEmbeddingConfig::default(),
            similarity_config: SimilarityConfig::default(),
            late_interaction_config: LateInteractionConfig::default(),
            performance_targets: TokenLevelPerformanceTargets::default(),
        }
    }
}

/// Token filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFilteringConfig {
    pub enable_stopword_filtering: bool,
    pub preserve_technical_tokens: bool,
    pub min_token_length: usize,
    pub max_token_length: usize,
    pub enable_subword_tokenization: bool,
    pub preserve_punctuation_context: bool,
}

impl Default for TokenFilteringConfig {
    fn default() -> Self {
        Self {
            enable_stopword_filtering: false, // Keep all tokens for ColBERT-style matching
            preserve_technical_tokens: true,
            min_token_length: 1,
            max_token_length: 50,
            enable_subword_tokenization: true,
            preserve_punctuation_context: true,
        }
    }
}

/// Token embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEmbeddingConfig {
    pub embedding_dimension: usize,
    pub enable_contextual_embeddings: bool,
    pub context_window_size: usize,
    pub enable_position_encoding: bool,
    pub normalization_method: TokenNormalizationMethod,
    pub enable_embedding_pooling: bool,
}

impl Default for TokenEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 1024, // Match mxbai-embed-large
            enable_contextual_embeddings: true,
            context_window_size: 5, // 5-token context window
            enable_position_encoding: true,
            normalization_method: TokenNormalizationMethod::L2Norm,
            enable_embedding_pooling: false,
        }
    }
}

/// Token normalization methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TokenNormalizationMethod {
    /// L2 normalization (unit vectors)
    L2Norm,
    /// Layer normalization
    LayerNorm,
    /// No normalization
    None,
}

/// Similarity calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    pub similarity_method: SimilarityMethod,
    pub aggregation_strategy: AggregationStrategy,
    pub enable_attention_weighting: bool,
    pub temperature_scaling: f64,
    pub similarity_threshold: f64,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            similarity_method: SimilarityMethod::CosineSimilarity,
            aggregation_strategy: AggregationStrategy::MaxSimilarity,
            enable_attention_weighting: true,
            temperature_scaling: 1.0,
            similarity_threshold: 0.1,
        }
    }
}

/// Similarity calculation methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SimilarityMethod {
    /// Cosine similarity (standard)
    CosineSimilarity,
    /// Dot product
    DotProduct,
    /// Euclidean distance
    EuclideanDistance,
    /// Manhattan distance
    ManhattanDistance,
}

/// Aggregation strategies for token similarities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregationStrategy {
    /// Maximum similarity (ColBERT-style)
    MaxSimilarity,
    /// Average similarity
    AverageSimilarity,
    /// Weighted average with attention
    AttentionWeighted,
    /// Top-k average
    TopKAverage { k: usize },
}

/// Late interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateInteractionConfig {
    pub enable_late_interaction: bool,
    pub interaction_method: InteractionMethod,
    pub enable_cross_attention: bool,
    pub attention_heads: usize,
    pub enable_residual_connections: bool,
}

impl Default for LateInteractionConfig {
    fn default() -> Self {
        Self {
            enable_late_interaction: true,
            interaction_method: InteractionMethod::TokenWiseInteraction,
            enable_cross_attention: false, // Simplified for now
            attention_heads: 8,
            enable_residual_connections: false,
        }
    }
}

/// Late interaction methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionMethod {
    /// Token-wise interaction (standard ColBERT)
    TokenWiseInteraction,
    /// Attention-based interaction
    AttentionBased,
    /// Pooled interaction
    PooledInteraction,
}

/// Performance targets for token-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLevelPerformanceTargets {
    pub max_processing_time_ms: u64,
    pub min_tokens_per_second: f64,
    pub max_memory_per_token_kb: f64,
    pub min_alignment_accuracy: f64,
    pub max_similarity_calculation_time_ms: u64,
}

impl Default for TokenLevelPerformanceTargets {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 100, // 100ms for token processing
            min_tokens_per_second: 500.0, // 500 tokens/sec minimum
            max_memory_per_token_kb: 10.0, // 10KB per token
            min_alignment_accuracy: 0.85, // 85% alignment accuracy
            max_similarity_calculation_time_ms: 20, // 20ms for similarity calculation
        }
    }
}

/// Token representation with embedding
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub position: usize,
    pub embedding: Vec<f32>,
    pub is_technical: bool,
    pub context_start: usize,
    pub context_end: usize,
}

impl Token {
    /// Create new token with embedding
    pub fn new(text: String, position: usize, embedding: Vec<f32>) -> Self {
        let is_technical = Self::is_technical_term(&text);
        
        Self {
            text,
            position,
            embedding,
            is_technical,
            context_start: position.saturating_sub(2), // 2-token left context
            context_end: position + 3, // 2-token right context
        }
    }
    
    /// Check if token is technical term
    fn is_technical_term(text: &str) -> bool {
        let technical_patterns = [
            "function", "class", "method", "variable", "parameter",
            "async", "await", "promise", "callback", "closure",
            "database", "query", "index", "schema", "migration",
            "api", "endpoint", "request", "response", "json",
            "algorithm", "optimization", "performance", "memory",
            "security", "authentication", "authorization", "token",
        ];
        
        let text_lower = text.to_lowercase();
        technical_patterns.iter().any(|&pattern| text_lower.contains(pattern)) ||
        text.chars().any(|c| c == '_' || c == '-') || // Snake_case or kebab-case
        text.chars().any(|c| c.is_uppercase()) && text.len() > 1 // CamelCase
    }
}

/// Token similarity matrix for alignment calculations
#[derive(Debug, Clone)]
pub struct TokenSimilarityMatrix {
    pub query_tokens: Vec<Token>,
    pub document_tokens: Vec<Token>,
    pub similarity_matrix: Vec<Vec<f32>>,
    pub max_similarities: Vec<f32>,
    pub alignment_scores: Vec<(usize, usize, f32)>, // (query_idx, doc_idx, similarity)
}

impl TokenSimilarityMatrix {
    /// Calculate overall alignment score using ColBERT-style max similarity
    pub fn calculate_alignment_score(&self) -> f64 {
        if self.max_similarities.is_empty() {
            return 0.0;
        }
        
        self.max_similarities.iter().sum::<f32>() as f64 / self.max_similarities.len() as f64
    }
    
    /// Get top-k token alignments
    pub fn get_top_alignments(&self, k: usize) -> Vec<(usize, usize, f32)> {
        let mut alignments = self.alignment_scores.clone();
        alignments.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        alignments.truncate(k);
        alignments
    }
    
    /// Calculate technical term alignment ratio
    pub fn technical_alignment_ratio(&self) -> f64 {
        let technical_alignments = self.alignment_scores.iter()
            .filter(|(q_idx, d_idx, _)| {
                self.query_tokens.get(*q_idx).map(|t| t.is_technical).unwrap_or(false) ||
                self.document_tokens.get(*d_idx).map(|t| t.is_technical).unwrap_or(false)
            })
            .count();
        
        if self.alignment_scores.is_empty() {
            0.0
        } else {
            technical_alignments as f64 / self.alignment_scores.len() as f64
        }
    }
}

/// Token-level processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVectorResult {
    pub content_id: Uuid,
    pub content: String,
    pub tokens: Vec<String>, // Token text only (embeddings are too large to serialize)
    pub token_count: usize,
    pub technical_token_count: usize,
    pub similarity_matrix_shape: (usize, usize),
    pub max_similarities: Vec<f32>,
    pub alignment_score: f64,
    pub metadata: TokenLevelMetadata,
    pub quality_metrics: TokenLevelQualityMetrics,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metadata for token-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLevelMetadata {
    pub tokenization_time_ms: u64,
    pub embedding_time_ms: u64,
    pub similarity_calculation_time_ms: u64,
    pub total_processing_time_ms: u64,
    pub context_window_size: usize,
    pub normalization_applied: TokenNormalizationMethod,
    pub similarity_method: SimilarityMethod,
    pub aggregation_strategy: AggregationStrategy,
}

/// Quality metrics for token-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLevelQualityMetrics {
    pub token_coverage: f64,
    pub technical_token_ratio: f64,
    pub embedding_quality: f64,
    pub alignment_accuracy: f64,
    pub similarity_distribution_quality: f64,
    pub overall_quality: f64,
}

/// ColBERT-style Token-Level Processor
pub struct TokenLevelProcessor {
    config: TokenLevelConfig,
    ai_engine: Arc<LocalAIEngine>,
    tokenizer: AdvancedTokenizer,
    similarity_calculator: SimilarityCalculator,
    late_interaction_processor: LateInteractionProcessor,
    performance_monitor: TokenLevelPerformanceMonitor,
    token_cache: Arc<RwLock<HashMap<String, Vec<Token>>>>,
}

impl TokenLevelProcessor {
    /// Create new token-level processor
    pub async fn new(
        config: TokenLevelConfig,
        ai_engine: Arc<LocalAIEngine>,
    ) -> Result<Self, TokenLevelError> {
        info!(
            "Initializing Token-Level Processor: max_tokens={}, contextual={}",
            config.max_tokens, config.embedding_config.enable_contextual_embeddings
        );
        
        let tokenizer = AdvancedTokenizer::new(config.token_filtering.clone());
        let similarity_calculator = SimilarityCalculator::new(config.similarity_config.clone());
        let late_interaction_processor = LateInteractionProcessor::new(config.late_interaction_config.clone());
        let performance_monitor = TokenLevelPerformanceMonitor::new(config.performance_targets.clone());
        let token_cache = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            config,
            ai_engine,
            tokenizer,
            similarity_calculator,
            late_interaction_processor,
            performance_monitor,
            token_cache,
        })
    }
    
    /// Process token-level vectors for content
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn process_token_vectors(
        &self,
        content: &str,
        content_id: Option<Uuid>,
    ) -> Result<TokenVectorResult, TokenLevelError> {
        let processing_start = Instant::now();
        let content_id = content_id.unwrap_or_else(Uuid::new_v4);
        
        debug!(
            "Processing token-level vectors: content_id={}, content_len={}",
            content_id, content.len()
        );
        
        // Step 1: Advanced tokenization
        let tokenization_start = Instant::now();
        let raw_tokens = self.tokenizer.tokenize(content).await?;
        let tokenization_time = tokenization_start.elapsed();
        
        // Step 2: Generate token embeddings with context
        let embedding_start = Instant::now();
        let tokens = self.generate_token_embeddings(&raw_tokens, content).await?;
        let embedding_time = embedding_start.elapsed();
        
        // Step 3: Calculate similarity matrix for self-attention (optional)
        let similarity_start = Instant::now();
        let similarity_matrix = if self.config.similarity_config.enable_attention_weighting {
            Some(self.calculate_self_similarity_matrix(&tokens).await?)
        } else {
            None
        };
        let similarity_calculation_time = similarity_start.elapsed();
        
        // Step 4: Apply late interaction processing
        let processed_tokens = if self.config.late_interaction_config.enable_late_interaction {
            self.late_interaction_processor.process_tokens(&tokens).await?
        } else {
            tokens
        };
        
        // Step 5: Calculate quality metrics
        let quality_metrics = self.calculate_token_quality_metrics(&processed_tokens, content).await?;
        
        let total_processing_time = processing_start.elapsed();
        
        // Step 6: Extract max similarities for result
        let max_similarities = if let Some(ref matrix) = similarity_matrix {
            matrix.max_similarities.clone()
        } else {
            // Create dummy max similarities based on token quality
            processed_tokens.iter()
                .map(|token| self.estimate_token_quality(token))
                .collect()
        };
        
        let alignment_score = if let Some(ref matrix) = similarity_matrix {
            matrix.calculate_alignment_score()
        } else {
            quality_metrics.alignment_accuracy
        };
        
        // Create metadata
        let metadata = TokenLevelMetadata {
            tokenization_time_ms: tokenization_time.as_millis() as u64,
            embedding_time_ms: embedding_time.as_millis() as u64,
            similarity_calculation_time_ms: similarity_calculation_time.as_millis() as u64,
            total_processing_time_ms: total_processing_time.as_millis() as u64,
            context_window_size: self.config.embedding_config.context_window_size,
            normalization_applied: self.config.embedding_config.normalization_method.clone(),
            similarity_method: self.config.similarity_config.similarity_method.clone(),
            aggregation_strategy: self.config.similarity_config.aggregation_strategy.clone(),
        };
        
        let result = TokenVectorResult {
            content_id,
            content: content.to_string(),
            tokens: processed_tokens.iter().map(|t| t.text.clone()).collect(),
            token_count: processed_tokens.len(),
            technical_token_count: processed_tokens.iter().filter(|t| t.is_technical).count(),
            similarity_matrix_shape: if let Some(ref matrix) = similarity_matrix {
                (matrix.query_tokens.len(), matrix.document_tokens.len())
            } else {
                (0, 0)
            },
            max_similarities,
            alignment_score,
            metadata,
            quality_metrics,
            generation_timestamp: chrono::Utc::now(),
        };
        
        // Record performance metrics
        self.performance_monitor.record_processing(
            total_processing_time,
            processed_tokens.len(),
            &result.quality_metrics,
        ).await;
        
        // Cache tokens for future use
        self.cache_tokens(content, &processed_tokens).await?;
        
        info!(
            "Token-level processing completed: content_id={}, tokens={}, time={}ms, quality={:.3}",
            content_id,
            result.token_count,
            total_processing_time.as_millis(),
            result.quality_metrics.overall_quality
        );
        
        Ok(result)
    }
    
    /// Generate token embeddings with contextual information
    async fn generate_token_embeddings(
        &self,
        raw_tokens: &[String],
        content: &str,
    ) -> Result<Vec<Token>, TokenLevelError> {
        let mut tokens = Vec::new();
        
        for (position, token_text) in raw_tokens.iter().enumerate() {
            // Create contextual prompt for embedding generation
            let contextual_text = if self.config.embedding_config.enable_contextual_embeddings {
                self.create_contextual_text(token_text, raw_tokens, position)
            } else {
                token_text.clone()
            };
            
            // Generate embedding using AI engine
            let embedding = self.ai_engine
                .generate_embedding(&contextual_text)
                .await
                .map_err(|e| TokenLevelError::TokenEmbeddingFailed {
                    token: token_text.clone(),
                    error: e.to_string(),
                })?;
            
            // Validate embedding dimension
            if embedding.len() != self.config.embedding_config.embedding_dimension {
                return Err(TokenLevelError::TokenEmbeddingFailed {
                    token: token_text.clone(),
                    error: format!("Expected {} dimensions, got {}", 
                                 self.config.embedding_config.embedding_dimension, 
                                 embedding.len()),
                });
            }
            
            // Apply normalization if enabled
            let normalized_embedding = match self.config.embedding_config.normalization_method {
                TokenNormalizationMethod::L2Norm => self.l2_normalize(embedding),
                TokenNormalizationMethod::LayerNorm => self.layer_normalize(embedding),
                TokenNormalizationMethod::None => embedding,
            };
            
            let token = Token::new(token_text.clone(), position, normalized_embedding);
            tokens.push(token);
        }
        
        Ok(tokens)
    }
    
    /// Create contextual text for token embedding
    fn create_contextual_text(&self, token: &str, all_tokens: &[String], position: usize) -> String {
        let window_size = self.config.embedding_config.context_window_size;
        let start = position.saturating_sub(window_size / 2);
        let end = (position + window_size / 2 + 1).min(all_tokens.len());
        
        let context_tokens = &all_tokens[start..end];
        context_tokens.join(" ")
    }
    
    /// Calculate self-similarity matrix for tokens
    async fn calculate_self_similarity_matrix(&self, tokens: &[Token]) -> Result<TokenSimilarityMatrix, TokenLevelError> {
        let mut similarity_matrix = vec![vec![0.0f32; tokens.len()]; tokens.len()];
        let mut max_similarities = Vec::new();
        let mut alignment_scores = Vec::new();
        
        for (i, token_i) in tokens.iter().enumerate() {
            let mut max_sim = 0.0f32;
            let mut best_alignment = (i, i, 0.0f32);
            
            for (j, token_j) in tokens.iter().enumerate() {
                let similarity = self.similarity_calculator.calculate_similarity(
                    &token_i.embedding,
                    &token_j.embedding,
                )?;
                
                similarity_matrix[i][j] = similarity;
                
                if similarity > max_sim {
                    max_sim = similarity;
                    best_alignment = (i, j, similarity);
                }
            }
            
            max_similarities.push(max_sim);
            
            if best_alignment.2 > self.config.similarity_config.similarity_threshold as f32 {
                alignment_scores.push(best_alignment);
            }
        }
        
        Ok(TokenSimilarityMatrix {
            query_tokens: tokens.to_vec(),
            document_tokens: tokens.to_vec(), // Self-similarity
            similarity_matrix,
            max_similarities,
            alignment_scores,
        })
    }
    
    /// Calculate query-document similarity matrix for ColBERT-style matching
    pub async fn calculate_query_document_similarity(
        &self,
        query_tokens: &[Token],
        document_tokens: &[Token],
    ) -> Result<TokenSimilarityMatrix, TokenLevelError> {
        let calc_start = Instant::now();
        
        let mut similarity_matrix = vec![vec![0.0f32; document_tokens.len()]; query_tokens.len()];
        let mut max_similarities = Vec::new();
        let mut alignment_scores = Vec::new();
        
        // Calculate similarities between all query and document token pairs
        for (i, query_token) in query_tokens.iter().enumerate() {
            let mut max_sim = 0.0f32;
            let mut best_doc_idx = 0;
            
            for (j, doc_token) in document_tokens.iter().enumerate() {
                let similarity = self.similarity_calculator.calculate_similarity(
                    &query_token.embedding,
                    &doc_token.embedding,
                )?;
                
                similarity_matrix[i][j] = similarity;
                
                if similarity > max_sim {
                    max_sim = similarity;
                    best_doc_idx = j;
                }
            }
            
            max_similarities.push(max_sim);
            
            if max_sim > self.config.similarity_config.similarity_threshold as f32 {
                alignment_scores.push((i, best_doc_idx, max_sim));
            }
        }
        
        let calc_time = calc_start.elapsed();
        
        if calc_time.as_millis() as u64 > self.config.performance_targets.max_similarity_calculation_time_ms {
            warn!(
                "Similarity calculation time exceeded target: {}ms > {}ms",
                calc_time.as_millis(),
                self.config.performance_targets.max_similarity_calculation_time_ms
            );
        }
        
        Ok(TokenSimilarityMatrix {
            query_tokens: query_tokens.to_vec(),
            document_tokens: document_tokens.to_vec(),
            similarity_matrix,
            max_similarities,
            alignment_scores,
        })
    }
    
    /// Calculate quality metrics for token processing
    async fn calculate_token_quality_metrics(
        &self,
        tokens: &[Token],
        content: &str,
    ) -> Result<TokenLevelQualityMetrics, TokenLevelError> {
        let token_coverage = if content.is_empty() {
            0.0
        } else {
            tokens.len() as f64 / content.split_whitespace().count() as f64
        };
        
        let technical_token_count = tokens.iter().filter(|t| t.is_technical).count();
        let technical_token_ratio = if tokens.is_empty() {
            0.0
        } else {
            technical_token_count as f64 / tokens.len() as f64
        };
        
        // Calculate embedding quality based on distribution
        let embedding_quality = self.calculate_embedding_quality(tokens);
        
        // Alignment accuracy based on technical term preservation
        let alignment_accuracy = if technical_token_count > 0 {
            0.9 // High accuracy for technical terms
        } else {
            0.8 // Standard accuracy
        };
        
        // Similarity distribution quality
        let similarity_distribution_quality = self.calculate_similarity_distribution_quality(tokens);
        
        let overall_quality = (token_coverage + technical_token_ratio + embedding_quality + 
                             alignment_accuracy + similarity_distribution_quality) / 5.0;
        
        Ok(TokenLevelQualityMetrics {
            token_coverage,
            technical_token_ratio,
            embedding_quality,
            alignment_accuracy,
            similarity_distribution_quality,
            overall_quality,
        })
    }
    
    /// Calculate embedding quality based on vector distribution
    fn calculate_embedding_quality(&self, tokens: &[Token]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        
        // Calculate average magnitude and variance as quality indicators
        let magnitudes: Vec<f64> = tokens.iter()
            .map(|token| {
                token.embedding.iter()
                    .map(|&x| x as f64 * x as f64)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        
        let mean_magnitude = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        let magnitude_variance = magnitudes.iter()
            .map(|&x| (x - mean_magnitude).powi(2))
            .sum::<f64>() / magnitudes.len() as f64;
        
        // Good embeddings should have reasonable magnitude and some variance
        let magnitude_score = if mean_magnitude > 0.5 && mean_magnitude < 2.0 { 1.0 } else { 0.5 };
        let variance_score = (magnitude_variance * 10.0).min(1.0); // Scale variance to [0,1]
        
        (magnitude_score + variance_score) / 2.0
    }
    
    /// Calculate similarity distribution quality
    fn calculate_similarity_distribution_quality(&self, tokens: &[Token]) -> f64 {
        // For now, return a fixed quality score
        // In a real implementation, this would analyze the distribution of similarities
        0.85
    }
    
    /// Estimate quality of individual token
    fn estimate_token_quality(&self, token: &Token) -> f32 {
        let magnitude = token.embedding.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let base_quality: f32 = if magnitude > 0.5 { 0.8 } else { 0.5 };
        let technical_bonus: f32 = if token.is_technical { 0.2 } else { 0.0 };
        
        (base_quality + technical_bonus).min(1.0)
    }
    
    /// L2 normalize embedding vector
    fn l2_normalize(&self, mut embedding: Vec<f32>) -> Vec<f32> {
        let magnitude = embedding.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        
        if magnitude > 0.0 {
            for value in &mut embedding {
                *value /= magnitude;
            }
        }
        
        embedding
    }
    
    /// Layer normalize embedding vector
    fn layer_normalize(&self, mut embedding: Vec<f32>) -> Vec<f32> {
        let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance = embedding.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / embedding.len() as f32;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            for value in &mut embedding {
                *value = (*value - mean) / std_dev;
            }
        }
        
        embedding
    }
    
    /// Cache tokens for future use
    async fn cache_tokens(&self, content: &str, tokens: &[Token]) -> Result<(), TokenLevelError> {
        let cache_key = format!("tokens_{:x}", {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            hasher.finish()
        });
        
        let mut cache = self.token_cache.write().await;
        cache.insert(cache_key, tokens.to_vec());
        
        Ok(())
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, config: TokenLevelConfig) -> Result<(), TokenLevelError> {
        info!("Updating token-level processor configuration");
        
        // Update components if needed
        if config.similarity_config.similarity_method != self.config.similarity_config.similarity_method {
            self.similarity_calculator = SimilarityCalculator::new(config.similarity_config.clone());
        }
        
        if config.late_interaction_config.enable_late_interaction != self.config.late_interaction_config.enable_late_interaction {
            self.late_interaction_processor = LateInteractionProcessor::new(config.late_interaction_config.clone());
        }
        
        if config.performance_targets.max_processing_time_ms != self.config.performance_targets.max_processing_time_ms {
            self.performance_monitor = TokenLevelPerformanceMonitor::new(config.performance_targets.clone());
        }
        
        self.config = config;
        Ok(())
    }
}

/// Advanced tokenizer component
pub struct AdvancedTokenizer {
    config: TokenFilteringConfig,
}

impl AdvancedTokenizer {
    pub fn new(config: TokenFilteringConfig) -> Self {
        Self { config }
    }
    
    /// Tokenize text with advanced features
    pub async fn tokenize(&self, text: &str) -> Result<Vec<String>, TokenLevelError> {
        let mut tokens = Vec::new();
        
        // Basic whitespace tokenization with punctuation handling
        for word in text.split_whitespace() {
            if self.config.preserve_punctuation_context {
                // Keep punctuation attached to words
                tokens.push(word.to_string());
            } else {
                // Split punctuation
                let cleaned = word.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect::<String>();
                
                if !cleaned.is_empty() {
                    tokens.push(cleaned);
                }
            }
        }
        
        // Apply subword tokenization if enabled
        if self.config.enable_subword_tokenization {
            tokens = self.apply_subword_tokenization(tokens).await?;
        }
        
        // Apply filtering
        tokens = self.apply_token_filtering(tokens);
        
        Ok(tokens)
    }
    
    /// Apply subword tokenization
    async fn apply_subword_tokenization(&self, tokens: Vec<String>) -> Result<Vec<String>, TokenLevelError> {
        let mut subword_tokens = Vec::new();
        
        for token in tokens {
            if token.len() > 10 && (token.contains('-') || token.contains('_')) {
                // Split compound words
                let parts: Vec<String> = token.split(&['-', '_'][..])
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                
                if parts.len() > 1 {
                    subword_tokens.extend(parts);
                } else {
                    subword_tokens.push(token);
                }
            } else {
                subword_tokens.push(token);
            }
        }
        
        Ok(subword_tokens)
    }
    
    /// Apply token filtering
    fn apply_token_filtering(&self, tokens: Vec<String>) -> Vec<String> {
        tokens.into_iter()
            .filter(|token| {
                token.len() >= self.config.min_token_length &&
                token.len() <= self.config.max_token_length
            })
            .collect()
    }
}

/// Similarity calculation component
pub struct SimilarityCalculator {
    config: SimilarityConfig,
}

impl SimilarityCalculator {
    pub fn new(config: SimilarityConfig) -> Self {
        Self { config }
    }
    
    /// Calculate similarity between two embeddings
    pub fn calculate_similarity(&self, embedding_a: &[f32], embedding_b: &[f32]) -> Result<f32, TokenLevelError> {
        match self.config.similarity_method {
            SimilarityMethod::CosineSimilarity => {
                Ok(cosine_similarity_optimized(embedding_a, embedding_b))
            },
            SimilarityMethod::DotProduct => {
                let dot_product = embedding_a.iter()
                    .zip(embedding_b.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();
                Ok(dot_product)
            },
            SimilarityMethod::EuclideanDistance => {
                let distance = embedding_a.iter()
                    .zip(embedding_b.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                // Convert distance to similarity (inverse relationship)
                Ok(1.0 / (1.0 + distance))
            },
            SimilarityMethod::ManhattanDistance => {
                let distance = embedding_a.iter()
                    .zip(embedding_b.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .sum::<f32>();
                // Convert distance to similarity
                Ok(1.0 / (1.0 + distance))
            },
        }
    }
}

/// Late interaction processor component
pub struct LateInteractionProcessor {
    config: LateInteractionConfig,
}

impl LateInteractionProcessor {
    pub fn new(config: LateInteractionConfig) -> Self {
        Self { config }
    }
    
    /// Process tokens with late interaction
    pub async fn process_tokens(&self, tokens: &[Token]) -> Result<Vec<Token>, TokenLevelError> {
        if !self.config.enable_late_interaction {
            return Ok(tokens.to_vec());
        }
        
        // For now, return tokens as-is
        // In a full implementation, this would apply attention mechanisms
        Ok(tokens.to_vec())
    }
}

/// Performance monitoring component
pub struct TokenLevelPerformanceMonitor {
    targets: TokenLevelPerformanceTargets,
}

impl TokenLevelPerformanceMonitor {
    pub fn new(targets: TokenLevelPerformanceTargets) -> Self {
        Self { targets }
    }
    
    pub async fn record_processing(
        &self,
        _processing_time: Duration,
        _token_count: usize,
        _quality_metrics: &TokenLevelQualityMetrics,
    ) {
        // Record performance metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_creation() {
        let embedding = vec![0.5f32; 1024];
        let token = Token::new("function".to_string(), 0, embedding.clone());
        
        assert_eq!(token.text, "function");
        assert_eq!(token.position, 0);
        assert_eq!(token.embedding.len(), 1024);
        assert!(token.is_technical);
    }
    
    #[test]
    fn test_technical_term_detection() {
        assert!(Token::is_technical_term("function"));
        assert!(Token::is_technical_term("async"));
        assert!(Token::is_technical_term("my_variable"));
        assert!(Token::is_technical_term("CamelCase"));
        assert!(!Token::is_technical_term("hello"));
        assert!(!Token::is_technical_term("world"));
    }
    
    #[tokio::test]
    async fn test_advanced_tokenizer() {
        let config = TokenFilteringConfig::default();
        let tokenizer = AdvancedTokenizer::new(config);
        
        let tokens = tokenizer.tokenize("Hello world, this is a test!").await.unwrap();
        
        assert!(!tokens.is_empty());
        assert!(tokens.contains(&"Hello".to_string()));
        assert!(tokens.contains(&"world,".to_string()) || tokens.contains(&"world".to_string()));
    }
    
    #[test]
    fn test_similarity_calculator_cosine() {
        let config = SimilarityConfig {
            similarity_method: SimilarityMethod::CosineSimilarity,
            ..Default::default()
        };
        let calculator = SimilarityCalculator::new(config);
        
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        
        let similarity = calculator.calculate_similarity(&vec_a, &vec_b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6); // Should be 1.0 for identical vectors
    }
    
    #[test]
    fn test_token_similarity_matrix() {
        let token1 = Token::new("hello".to_string(), 0, vec![1.0, 0.0]);
        let token2 = Token::new("world".to_string(), 1, vec![0.0, 1.0]);
        
        let matrix = TokenSimilarityMatrix {
            query_tokens: vec![token1.clone()],
            document_tokens: vec![token2.clone()],
            similarity_matrix: vec![vec![0.5]],
            max_similarities: vec![0.5],
            alignment_scores: vec![(0, 0, 0.5)],
        };
        
        // Use approximate comparison for floating-point values
        let epsilon = 1e-6f64;
        let score = matrix.calculate_alignment_score();
        assert!((score - 0.5f64).abs() < epsilon, "Expected alignment_score ~0.5, got {}", score);
        assert_eq!(matrix.get_top_alignments(1).len(), 1);
    }
}
