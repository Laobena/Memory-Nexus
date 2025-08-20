//! Sparse Vector Generator with Enhanced BM25+ Integration
//!
//! Advanced sparse vector processing system that generates high-quality keyword-based
//! vector representations using enhanced BM25+ algorithms with technical term weighting,
//! inverse document frequency optimization, and domain-specific enhancements for
//! superior keyword matching precision in multi-vector search scenarios.

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
use crate::search::enhanced_bm25::{EnhancedBM25Plus, BM25Config};

/// Errors related to sparse vector generation
#[derive(Error, Debug)]
pub enum SparseVectorError {
    #[error("Tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
    
    #[error("Term frequency calculation failed: {term} - {issue}")]
    TermFrequencyError { term: String, issue: String },
    
    #[error("IDF calculation failed: {corpus_size} documents - {error}")]
    IDFCalculationError { corpus_size: usize, error: String },
    
    #[error("Sparse vector encoding failed: {dimension} - {reason}")]
    EncodingError { dimension: usize, reason: String },
    
    #[error("Dictionary operation failed: {operation} - {details}")]
    DictionaryError { operation: String, details: String },
    
    #[error("Performance target missed: {metric}={actual} vs target={target}")]
    PerformanceError { metric: String, actual: String, target: String },
    
    #[error("Corpus statistics invalid: {statistic} - {value}")]
    CorpusStatisticsError { statistic: String, value: String },
    
    #[error("BM25 processing error: {0}")]
    BM25Error(String),
}

/// Configuration for sparse vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVectorConfig {
    /// Maximum vocabulary size
    pub max_vocabulary_size: usize,
    /// Enable enhanced BM25+ features
    pub enable_enhanced_bm25: bool,
    /// BM25 parameters
    pub bm25_config: BM25Config,
    /// Term filtering configuration
    pub term_filtering: TermFilteringConfig,
    /// Technical term enhancement
    pub technical_enhancement: TechnicalTermConfig,
    /// IDF computation settings
    pub idf_config: IDFConfig,
    /// Vector encoding configuration
    pub encoding_config: SparseEncodingConfig,
    /// Performance targets
    pub performance_targets: SparseVectorPerformanceTargets,
}

impl Default for SparseVectorConfig {
    fn default() -> Self {
        Self {
            max_vocabulary_size: 50000,
            enable_enhanced_bm25: true,
            bm25_config: BM25Config::default(),
            term_filtering: TermFilteringConfig::default(),
            technical_enhancement: TechnicalTermConfig::default(),
            idf_config: IDFConfig::default(),
            encoding_config: SparseEncodingConfig::default(),
            performance_targets: SparseVectorPerformanceTargets::default(),
        }
    }
}

/// Term filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermFilteringConfig {
    pub enable_stopword_removal: bool,
    pub custom_stopwords: Vec<String>,
    pub min_term_frequency: usize,
    pub max_term_frequency: usize,
    pub min_document_frequency: usize,
    pub min_term_length: usize,
    pub max_term_length: usize,
    pub enable_stemming: bool,
    pub enable_lemmatization: bool,
}

impl Default for TermFilteringConfig {
    fn default() -> Self {
        Self {
            enable_stopword_removal: true,
            custom_stopwords: vec![
                "the".to_string(), "a".to_string(), "an".to_string(),
                "and".to_string(), "or".to_string(), "but".to_string(),
                "in".to_string(), "on".to_string(), "at".to_string(),
                "to".to_string(), "for".to_string(), "of".to_string(),
                "with".to_string(), "by".to_string(), "is".to_string(),
                "are".to_string(), "was".to_string(), "were".to_string(),
            ],
            min_term_frequency: 2,
            max_term_frequency: 1000,
            min_document_frequency: 2,
            min_term_length: 2,
            max_term_length: 50,
            enable_stemming: false,
            enable_lemmatization: false,
        }
    }
}

/// Technical term enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalTermConfig {
    pub enable_technical_boost: bool,
    pub technical_term_multiplier: f64,
    pub programming_terms: HashSet<String>,
    pub domain_specific_terms: HashMap<String, f64>, // Domain -> weight multiplier
    pub compound_term_handling: CompoundTermHandling,
}

impl Default for TechnicalTermConfig {
    fn default() -> Self {
        let mut programming_terms = HashSet::new();
        for term in &[
            "function", "class", "method", "variable", "parameter", "return",
            "async", "await", "promise", "callback", "closure", "lambda",
            "database", "query", "index", "transaction", "schema", "migration",
            "api", "endpoint", "request", "response", "header", "payload",
            "algorithm", "optimization", "performance", "scalability",
            "security", "authentication", "authorization", "encryption",
            "docker", "kubernetes", "microservice", "container", "deployment",
            "react", "component", "props", "state", "hooks", "redux",
            "javascript", "typescript", "python", "rust", "golang", "java",
            "framework", "library", "package", "module", "import", "export",
        ] {
            programming_terms.insert(term.to_string());
        }
        
        let mut domain_terms = HashMap::new();
        domain_terms.insert("programming".to_string(), 1.5);
        domain_terms.insert("database".to_string(), 1.4);
        domain_terms.insert("security".to_string(), 1.3);
        domain_terms.insert("devops".to_string(), 1.3);
        
        Self {
            enable_technical_boost: true,
            technical_term_multiplier: 1.5,
            programming_terms,
            domain_specific_terms: domain_terms,
            compound_term_handling: CompoundTermHandling::SplitAndBoost,
        }
    }
}

/// Compound term handling strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompoundTermHandling {
    /// Treat as single term
    Single,
    /// Split and include both compound and parts
    SplitAndBoost,
    /// Split and only include parts
    SplitOnly,
}

/// Inverse Document Frequency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IDFConfig {
    pub enable_smooth_idf: bool,
    pub smoothing_factor: f64,
    pub enable_probabilistic_idf: bool,
    pub corpus_size_estimate: usize,
    pub enable_dynamic_idf: bool,
    pub idf_update_interval: Duration,
}

impl Default for IDFConfig {
    fn default() -> Self {
        Self {
            enable_smooth_idf: true,
            smoothing_factor: 1.0,
            enable_probabilistic_idf: true,
            corpus_size_estimate: 100000,
            enable_dynamic_idf: true,
            idf_update_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Sparse vector encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEncodingConfig {
    pub encoding_method: SparseEncodingMethod,
    pub normalization: SparseNormalization,
    pub compression_threshold: f64,
    pub max_non_zero_elements: usize,
    pub enable_quantization: bool,
    pub quantization_levels: usize,
}

impl Default for SparseEncodingConfig {
    fn default() -> Self {
        Self {
            encoding_method: SparseEncodingMethod::IndexValue,
            normalization: SparseNormalization::L2Norm,
            compression_threshold: 1e-6,
            max_non_zero_elements: 1000,
            enable_quantization: false,
            quantization_levels: 256,
        }
    }
}

/// Sparse vector encoding methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SparseEncodingMethod {
    /// Index-value pairs (standard)
    IndexValue,
    /// Compressed sparse row
    CSR,
    /// Dictionary-based encoding
    Dictionary,
    /// Hybrid encoding
    Hybrid,
}

/// Sparse vector normalization methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SparseNormalization {
    /// L1 normalization (Manhattan distance)
    L1Norm,
    /// L2 normalization (Euclidean distance)
    L2Norm,
    /// Max normalization
    MaxNorm,
    /// No normalization
    None,
}

/// Performance targets for sparse vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVectorPerformanceTargets {
    pub max_generation_time_ms: u64,
    pub min_throughput_vectors_per_sec: f64,
    pub max_memory_per_vector_kb: f64,
    pub max_non_zero_ratio: f64,
    pub min_distinctiveness_score: f64,
}

impl Default for SparseVectorPerformanceTargets {
    fn default() -> Self {
        Self {
            max_generation_time_ms: 20, // 20ms per vector
            min_throughput_vectors_per_sec: 50.0, // 50 vectors/sec
            max_memory_per_vector_kb: 2.0, // 2KB per vector
            max_non_zero_ratio: 0.1, // 10% non-zero elements max
            min_distinctiveness_score: 0.7, // 70% distinctiveness
        }
    }
}

/// Sparse vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero indices
    pub indices: Vec<usize>,
    /// Corresponding values
    pub values: Vec<f32>,
    /// Vector dimension (vocabulary size)
    pub dimension: usize,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl SparseVector {
    /// Create new sparse vector
    pub fn new(indices: Vec<usize>, values: Vec<f32>, dimension: usize) -> Self {
        assert_eq!(indices.len(), values.len());
        let nnz = indices.len();
        
        Self {
            indices,
            values,
            dimension,
            nnz,
        }
    }
    
    /// Get sparsity ratio (percentage of zero elements)
    pub fn sparsity_ratio(&self) -> f64 {
        if self.dimension == 0 {
            0.0
        } else {
            (self.dimension - self.nnz) as f64 / self.dimension as f64
        }
    }
    
    /// Get density ratio (percentage of non-zero elements)
    pub fn density_ratio(&self) -> f64 {
        1.0 - self.sparsity_ratio()
    }
    
    /// Calculate L2 norm
    pub fn l2_norm(&self) -> f64 {
        self.values.iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt()
    }
    
    /// Calculate L1 norm
    pub fn l1_norm(&self) -> f64 {
        self.values.iter()
            .map(|&x| x.abs() as f64)
            .sum::<f64>()
    }
}

/// Sparse vector generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVectorResult {
    pub content_id: Uuid,
    pub content: String,
    pub sparse_vector: SparseVector,
    pub metadata: SparseVectorMetadata,
    pub quality_metrics: SparseQualityMetrics,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metadata for sparse vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVectorMetadata {
    pub total_terms: usize,
    pub unique_terms: usize,
    pub filtered_terms: usize,
    pub technical_terms: usize,
    pub generation_time_ms: u64,
    pub bm25_parameters: BM25Parameters,
    pub encoding_method: SparseEncodingMethod,
    pub normalization_applied: SparseNormalization,
}

/// BM25 parameters used in generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Parameters {
    pub k1: f64,
    pub b: f64,
    pub avg_doc_length: f64,
    pub corpus_size: usize,
}

/// Quality metrics for sparse vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQualityMetrics {
    pub sparsity_ratio: f64,
    pub term_coverage: f64,
    pub technical_term_ratio: f64,
    pub distinctiveness_score: f64,
    pub bm25_coherence: f64,
    pub overall_quality: f64,
}

/// Term statistics for corpus
#[derive(Debug, Clone)]
pub struct TermStatistics {
    pub term_frequencies: HashMap<String, usize>,
    pub document_frequencies: HashMap<String, usize>,
    pub total_documents: usize,
    pub total_terms: usize,
    pub average_document_length: f64,
}

/// Vocabulary manager for term indexing
pub struct VocabularyManager {
    term_to_index: HashMap<String, usize>,
    index_to_term: Vec<String>,
    next_index: usize,
    max_size: usize,
}

impl VocabularyManager {
    pub fn new(max_size: usize) -> Self {
        Self {
            term_to_index: HashMap::new(),
            index_to_term: Vec::new(),
            next_index: 0,
            max_size,
        }
    }
    
    /// Get or create index for term
    pub fn get_or_create_index(&mut self, term: &str) -> Option<usize> {
        if let Some(&index) = self.term_to_index.get(term) {
            Some(index)
        } else if self.next_index < self.max_size {
            let index = self.next_index;
            self.term_to_index.insert(term.to_string(), index);
            self.index_to_term.push(term.to_string());
            self.next_index += 1;
            Some(index)
        } else {
            None // Vocabulary full
        }
    }
    
    /// Get term for index
    pub fn get_term(&self, index: usize) -> Option<&str> {
        self.index_to_term.get(index).map(|s| s.as_str())
    }
    
    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.next_index
    }
}

/// Enhanced Sparse Vector Generator
pub struct SparseVectorGenerator {
    config: SparseVectorConfig,
    bm25_engine: EnhancedBM25Plus,
    vocabulary: Arc<RwLock<VocabularyManager>>,
    term_statistics: Arc<RwLock<TermStatistics>>,
    technical_terms: HashSet<String>,
    performance_monitor: SparseVectorPerformanceMonitor,
}

impl SparseVectorGenerator {
    /// Create new sparse vector generator
    pub async fn new(config: SparseVectorConfig) -> Result<Self, SparseVectorError> {
        info!(
            "Initializing Sparse Vector Generator: vocab_size={}, enhanced_bm25={}",
            config.max_vocabulary_size, config.enable_enhanced_bm25
        );
        
        let bm25_engine = EnhancedBM25Plus::new(config.bm25_config.clone());
        let vocabulary = Arc::new(RwLock::new(VocabularyManager::new(config.max_vocabulary_size)));
        let term_statistics = Arc::new(RwLock::new(TermStatistics {
            term_frequencies: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            total_terms: 0,
            average_document_length: 0.0,
        }));
        
        let technical_terms = config.technical_enhancement.programming_terms.clone();
        let performance_monitor = SparseVectorPerformanceMonitor::new(config.performance_targets.clone());
        
        Ok(Self {
            config,
            bm25_engine,
            vocabulary,
            term_statistics,
            technical_terms,
            performance_monitor,
        })
    }
    
    /// Generate sparse vector for single text content
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn generate_sparse_vector(
        &self,
        content: &str,
        content_id: Option<Uuid>,
    ) -> Result<SparseVectorResult, SparseVectorError> {
        let generation_start = Instant::now();
        let content_id = content_id.unwrap_or_else(Uuid::new_v4);
        
        debug!(
            "Generating sparse vector: content_id={}, content_len={}",
            content_id, content.len()
        );
        
        // Step 1: Tokenization and preprocessing
        let tokens = self.tokenize_and_preprocess(content).await?;
        
        // Step 2: Calculate term frequencies and BM25 scores
        let term_scores = self.calculate_bm25_scores(&tokens, content).await?;
        
        // Step 3: Apply technical term enhancement
        let enhanced_scores = if self.config.technical_enhancement.enable_technical_boost {
            self.apply_technical_enhancement(term_scores).await?
        } else {
            term_scores
        };
        
        // Step 4: Convert to sparse vector format
        let enhanced_scores_len = enhanced_scores.len();
        let sparse_vector = self.encode_sparse_vector(enhanced_scores).await?;
        
        // Step 5: Apply normalization
        let normalized_vector = self.apply_normalization(sparse_vector).await?;
        
        // Step 6: Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&normalized_vector, &tokens).await?;
        
        let generation_time = generation_start.elapsed();
        
        // Create metadata
        let metadata = SparseVectorMetadata {
            total_terms: tokens.len(),
            unique_terms: enhanced_scores_len,
            filtered_terms: tokens.len() - enhanced_scores_len,
            technical_terms: self.count_technical_terms(&tokens),
            generation_time_ms: generation_time.as_millis() as u64,
            bm25_parameters: BM25Parameters {
                k1: self.config.bm25_config.k1 as f64,
                b: self.config.bm25_config.b as f64,
                avg_doc_length: self.get_average_document_length().await,
                corpus_size: self.get_corpus_size().await,
            },
            encoding_method: self.config.encoding_config.encoding_method.clone(),
            normalization_applied: self.config.encoding_config.normalization.clone(),
        };
        
        let result = SparseVectorResult {
            content_id,
            content: content.to_string(),
            sparse_vector: normalized_vector,
            metadata,
            quality_metrics,
            generation_timestamp: chrono::Utc::now(),
        };
        
        // Record performance metrics
        self.performance_monitor.record_generation(
            generation_time,
            content.len(),
            &result.sparse_vector,
            &result.quality_metrics,
        ).await;
        
        // Validate performance targets
        if generation_time.as_millis() as u64 > self.config.performance_targets.max_generation_time_ms {
            warn!(
                "Sparse vector generation time exceeded target: {}ms > {}ms",
                generation_time.as_millis(),
                self.config.performance_targets.max_generation_time_ms
            );
        }
        
        info!(
            "Sparse vector generated: content_id={}, time={}ms, nnz={}, quality={:.3}",
            content_id,
            generation_time.as_millis(),
            result.sparse_vector.nnz,
            result.quality_metrics.overall_quality
        );
        
        Ok(result)
    }
    
    /// Tokenize and preprocess content
    async fn tokenize_and_preprocess(&self, content: &str) -> Result<Vec<String>, SparseVectorError> {
        let mut tokens = Vec::new();
        
        // Basic tokenization (split on whitespace and punctuation)
        for word in content.split_whitespace() {
            let cleaned = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>()
                .to_lowercase();
            
            if !cleaned.is_empty() {
                tokens.push(cleaned);
            }
        }
        
        // Apply filtering
        let filtered_tokens = self.apply_term_filtering(tokens).await?;
        
        Ok(filtered_tokens)
    }
    
    /// Apply term filtering based on configuration
    async fn apply_term_filtering(&self, tokens: Vec<String>) -> Result<Vec<String>, SparseVectorError> {
        let mut filtered = Vec::new();
        
        for token in tokens {
            // Length filtering
            if token.len() < self.config.term_filtering.min_term_length ||
               token.len() > self.config.term_filtering.max_term_length {
                continue;
            }
            
            // Stopword filtering
            if self.config.term_filtering.enable_stopword_removal &&
               self.config.term_filtering.custom_stopwords.contains(&token) {
                continue;
            }
            
            // Handle compound terms
            let processed_tokens = match self.config.technical_enhancement.compound_term_handling {
                CompoundTermHandling::Single => vec![token],
                CompoundTermHandling::SplitAndBoost => {
                    let mut result = vec![token.clone()];
                    if token.contains('-') || token.contains('_') {
                        let parts: Vec<String> = token.split(&['-', '_'][..])
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect();
                        result.extend(parts);
                    }
                    result
                },
                CompoundTermHandling::SplitOnly => {
                    if token.contains('-') || token.contains('_') {
                        token.split(&['-', '_'][..])
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect()
                    } else {
                        vec![token]
                    }
                },
            };
            
            filtered.extend(processed_tokens);
        }
        
        Ok(filtered)
    }
    
    /// Calculate BM25 scores for terms
    async fn calculate_bm25_scores(&self, tokens: &[String], content: &str) -> Result<HashMap<String, f64>, SparseVectorError> {
        // Calculate term frequencies
        let mut term_frequencies = HashMap::new();
        for token in tokens {
            *term_frequencies.entry(token.clone()).or_insert(0) += 1;
        }
        
        let doc_length = tokens.len();
        let avg_doc_length = self.get_average_document_length().await;
        let corpus_size = self.get_corpus_size().await;
        
        let mut bm25_scores = HashMap::new();
        
        for (term, tf) in term_frequencies {
            // Get document frequency for the term
            let df = self.get_document_frequency(&term).await;
            
            // Calculate BM25 score
            let k1 = self.config.bm25_config.k1;
            let b = self.config.bm25_config.b;
            
            let idf = if df > 0 {
                ((corpus_size as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln()
            } else {
                (corpus_size as f64).ln() // Maximum IDF for unseen terms
            };
            
            let tf_component = (tf as f64 * (k1 as f64 + 1.0)) / 
                              (tf as f64 + k1 as f64 * (1.0 - b as f64 + b as f64 * (doc_length as f64 / avg_doc_length)));
            
            let bm25_score = idf * tf_component;
            bm25_scores.insert(term, bm25_score);
        }
        
        Ok(bm25_scores)
    }
    
    /// Apply technical term enhancement
    async fn apply_technical_enhancement(&self, mut term_scores: HashMap<String, f64>) -> Result<HashMap<String, f64>, SparseVectorError> {
        let multiplier = self.config.technical_enhancement.technical_term_multiplier;
        
        for (term, score) in &mut term_scores {
            if self.technical_terms.contains(term) {
                *score *= multiplier;
            }
            
            // Apply domain-specific boosts
            for (domain, domain_multiplier) in &self.config.technical_enhancement.domain_specific_terms {
                if term.contains(domain) {
                    *score *= domain_multiplier;
                    break;
                }
            }
        }
        
        Ok(term_scores)
    }
    
    /// Encode term scores as sparse vector
    async fn encode_sparse_vector(&self, term_scores: HashMap<String, f64>) -> Result<SparseVector, SparseVectorError> {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        // Sort terms by score (descending) and take top terms
        let mut sorted_terms: Vec<_> = term_scores.into_iter().collect();
        sorted_terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to max non-zero elements
        let limit = self.config.encoding_config.max_non_zero_elements.min(sorted_terms.len());
        
        let mut vocab = self.vocabulary.write().await;
        
        for (term, score) in sorted_terms.iter().take(limit) {
            // Filter out very small values
            if score.abs() < self.config.encoding_config.compression_threshold {
                continue;
            }
            
            if let Some(index) = vocab.get_or_create_index(term) {
                indices.push(index);
                values.push(*score as f32);
            }
        }
        
        let dimension = vocab.size().max(indices.iter().max().copied().unwrap_or(0) + 1);
        
        Ok(SparseVector::new(indices, values, dimension))
    }
    
    /// Apply normalization to sparse vector
    async fn apply_normalization(&self, mut sparse_vector: SparseVector) -> Result<SparseVector, SparseVectorError> {
        match self.config.encoding_config.normalization {
            SparseNormalization::L1Norm => {
                let l1_norm = sparse_vector.l1_norm();
                if l1_norm > 0.0 {
                    for value in &mut sparse_vector.values {
                        *value /= l1_norm as f32;
                    }
                }
            },
            SparseNormalization::L2Norm => {
                let l2_norm = sparse_vector.l2_norm();
                if l2_norm > 0.0 {
                    for value in &mut sparse_vector.values {
                        *value /= l2_norm as f32;
                    }
                }
            },
            SparseNormalization::MaxNorm => {
                let max_val = sparse_vector.values.iter()
                    .map(|&x| x.abs())
                    .fold(0.0f32, f32::max);
                if max_val > 0.0 {
                    for value in &mut sparse_vector.values {
                        *value /= max_val;
                    }
                }
            },
            SparseNormalization::None => {
                // No normalization applied
            },
        }
        
        Ok(sparse_vector)
    }
    
    /// Calculate quality metrics for sparse vector
    async fn calculate_quality_metrics(
        &self,
        sparse_vector: &SparseVector,
        tokens: &[String],
    ) -> Result<SparseQualityMetrics, SparseVectorError> {
        let sparsity_ratio = sparse_vector.sparsity_ratio();
        let term_coverage = sparse_vector.nnz as f64 / tokens.len() as f64;
        let technical_terms = self.count_technical_terms(tokens);
        let technical_term_ratio = technical_terms as f64 / tokens.len() as f64;
        
        // Calculate distinctiveness based on value distribution
        let mean_value = sparse_vector.values.iter().sum::<f32>() / sparse_vector.values.len() as f32;
        let variance = sparse_vector.values.iter()
            .map(|&x| (x - mean_value) * (x - mean_value))
            .sum::<f32>() / sparse_vector.values.len() as f32;
        let distinctiveness_score = variance.sqrt() as f64;
        
        // BM25 coherence based on score distribution
        let bm25_coherence = 1.0 - sparsity_ratio; // Simple heuristic
        
        let overall_quality = (term_coverage + technical_term_ratio + distinctiveness_score + bm25_coherence) / 4.0;
        
        Ok(SparseQualityMetrics {
            sparsity_ratio,
            term_coverage,
            technical_term_ratio,
            distinctiveness_score,
            bm25_coherence,
            overall_quality,
        })
    }
    
    /// Count technical terms in tokens
    fn count_technical_terms(&self, tokens: &[String]) -> usize {
        tokens.iter()
            .filter(|token| self.technical_terms.contains(*token))
            .count()
    }
    
    /// Get average document length
    async fn get_average_document_length(&self) -> f64 {
        let stats = self.term_statistics.read().await;
        stats.average_document_length
    }
    
    /// Get corpus size
    async fn get_corpus_size(&self) -> usize {
        let stats = self.term_statistics.read().await;
        stats.total_documents.max(self.config.idf_config.corpus_size_estimate)
    }
    
    /// Get document frequency for term
    async fn get_document_frequency(&self, term: &str) -> usize {
        let stats = self.term_statistics.read().await;
        stats.document_frequencies.get(term).copied().unwrap_or(0)
    }
    
    /// Update corpus statistics with new document
    pub async fn update_corpus_statistics(&self, tokens: &[String]) -> Result<(), SparseVectorError> {
        let mut stats = self.term_statistics.write().await;
        
        stats.total_documents += 1;
        stats.total_terms += tokens.len();
        stats.average_document_length = stats.total_terms as f64 / stats.total_documents as f64;
        
        // Update term frequencies
        let mut unique_terms = HashSet::new();
        for token in tokens {
            *stats.term_frequencies.entry(token.clone()).or_insert(0) += 1;
            unique_terms.insert(token.clone());
        }
        
        // Update document frequencies
        for term in unique_terms {
            *stats.document_frequencies.entry(term).or_insert(0) += 1;
        }
        
        Ok(())
    }
    
    /// Get vocabulary statistics
    pub async fn get_vocabulary_stats(&self) -> VocabularyStatistics {
        let vocab = self.vocabulary.read().await;
        let stats = self.term_statistics.read().await;
        
        VocabularyStatistics {
            total_terms: vocab.size(),
            max_terms: self.config.max_vocabulary_size,
            usage_ratio: vocab.size() as f64 / self.config.max_vocabulary_size as f64,
            total_documents: stats.total_documents,
            average_document_length: stats.average_document_length,
        }
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, config: SparseVectorConfig) -> Result<(), SparseVectorError> {
        info!("Updating sparse vector generator configuration");
        
        // Update BM25 engine if parameters changed
        if config.bm25_config.k1 != self.config.bm25_config.k1 || 
           config.bm25_config.b != self.config.bm25_config.b {
            self.bm25_engine = EnhancedBM25Plus::new(config.bm25_config.clone());
        }
        
        // Update technical terms if changed
        if config.technical_enhancement.programming_terms != self.config.technical_enhancement.programming_terms {
            self.technical_terms = config.technical_enhancement.programming_terms.clone();
        }
        
        // Update performance monitor if targets changed
        if config.performance_targets.max_generation_time_ms != self.config.performance_targets.max_generation_time_ms {
            self.performance_monitor = SparseVectorPerformanceMonitor::new(config.performance_targets.clone());
        }
        
        self.config = config;
        Ok(())
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyStatistics {
    pub total_terms: usize,
    pub max_terms: usize,
    pub usage_ratio: f64,
    pub total_documents: usize,
    pub average_document_length: f64,
}

/// Performance monitoring for sparse vectors
pub struct SparseVectorPerformanceMonitor {
    targets: SparseVectorPerformanceTargets,
}

impl SparseVectorPerformanceMonitor {
    pub fn new(targets: SparseVectorPerformanceTargets) -> Self {
        Self { targets }
    }
    
    pub async fn record_generation(
        &self,
        _generation_time: Duration,
        _content_length: usize,
        _sparse_vector: &SparseVector,
        _quality_metrics: &SparseQualityMetrics,
    ) {
        // Record performance metrics for monitoring
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_vector_creation() {
        let indices = vec![0, 5, 10];
        let values = vec![0.8, 0.6, 0.4];
        let dimension = 20;
        
        let sparse_vec = SparseVector::new(indices, values, dimension);
        
        assert_eq!(sparse_vec.nnz, 3);
        assert_eq!(sparse_vec.dimension, 20);
        // Use approximate comparison for floating-point values
        let epsilon = 1e-6f64;
        let sparsity = sparse_vec.sparsity_ratio();
        let density = sparse_vec.density_ratio();
        assert!((sparsity - 0.85f64).abs() < epsilon, "Expected sparsity ~0.85, got {}", sparsity);
        assert!((density - 0.15f64).abs() < epsilon, "Expected density ~0.15, got {}", density);
    }
    
    #[test]
    fn test_sparse_vector_norms() {
        let indices = vec![0, 1, 2];
        let values = vec![3.0, 4.0, 0.0];
        let sparse_vec = SparseVector::new(indices, values, 10);
        
        assert!((sparse_vec.l2_norm() - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
        assert!((sparse_vec.l1_norm() - 7.0).abs() < 1e-6); // |3| + |4| + |0| = 7
    }
    
    #[tokio::test]
    async fn test_vocabulary_manager() {
        let mut vocab = VocabularyManager::new(5);
        
        assert_eq!(vocab.get_or_create_index("hello"), Some(0));
        assert_eq!(vocab.get_or_create_index("world"), Some(1));
        assert_eq!(vocab.get_or_create_index("hello"), Some(0)); // Should return existing
        
        assert_eq!(vocab.size(), 2);
        assert_eq!(vocab.get_term(0), Some("hello"));
        assert_eq!(vocab.get_term(1), Some("world"));
    }
    
    #[test]
    fn test_term_filtering_config() {
        let config = TermFilteringConfig::default();
        
        assert!(config.enable_stopword_removal);
        assert!(config.custom_stopwords.contains(&"the".to_string()));
        assert_eq!(config.min_term_length, 2);
        assert_eq!(config.max_term_length, 50);
    }
    
    #[test]
    fn test_technical_term_config() {
        let config = TechnicalTermConfig::default();
        
        assert!(config.enable_technical_boost);
        // Use approximate comparison for floating-point values
        let epsilon = 1e-6f64;
        assert!((config.technical_term_multiplier - 1.5f64).abs() < epsilon, 
            "Expected technical_term_multiplier ~1.5, got {}", config.technical_term_multiplier);
        assert!(config.programming_terms.contains("function"));
        assert!(config.programming_terms.contains("async"));
        assert!(config.domain_specific_terms.contains_key("programming"));
    }
}