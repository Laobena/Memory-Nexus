use crate::core::{Config, Result};
use crate::core::types::ProcessedResult;
use crate::core::hash_utils::{dedup_hash, xxhash3_64};
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{HashSet, HashMap};
use std::time::Instant;
use ahash::AHashSet;
use tracing::{debug, instrument};

/// Advanced fusion engine with intelligent deduplication and cross-validation
pub struct FusionEngine {
    strategies: DashMap<String, Box<dyn FusionStrategy>>,
    current_strategy: Arc<RwLock<String>>,
    scoring_matrix: Arc<RwLock<ScoringMatrix>>,
    minhash_params: MinHashParams,
    cross_validation_enabled: bool,
    stats: Arc<RwLock<CrossValidationStats>>,
}

/// MinHash parameters for deduplication
#[derive(Clone, Debug)]
struct MinHashParams {
    num_hashes: usize,
    similarity_threshold: f32,
}

#[async_trait]
trait FusionStrategy: Send + Sync {
    async fn fuse(&self, results: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>>;
    fn name(&self) -> &str;
}

/// 6-factor scoring matrix for intelligent fusion
#[derive(Clone, Debug)]
pub struct ScoringMatrix {
    pub relevance: f32,      // Semantic match quality (0.35)
    pub freshness: f32,      // Temporal relevance (0.15)
    pub diversity: f32,      // Content uniqueness (0.15)
    pub authority: f32,      // Source credibility (0.15)
    pub coherence: f32,      // Cross-reference validation (0.10)
    pub confidence: f32,     // Prediction confidence (0.10)
}

/// MinHash signature for fast similarity comparison
#[derive(Clone, Debug)]
struct MinHashSignature {
    hashes: Vec<u64>,
    content_hash: u64,
}

/// Cross-validation statistics
#[derive(Clone, Debug, Default)]
pub struct CrossValidationStats {
    pub total_results: usize,
    pub unique_results: usize,
    pub deduplicated: usize,
    pub cross_validated: usize,
    pub fusion_time_ms: u128,
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        Self {
            relevance: 0.35,
            freshness: 0.15,
            diversity: 0.15,
            authority: 0.15,
            coherence: 0.10,
            confidence: 0.10,
        }
    }
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            strategies: DashMap::new(),
            current_strategy: Arc::new(RwLock::new("intelligent".to_string())),
            scoring_matrix: Arc::new(RwLock::new(ScoringMatrix::default())),
            minhash_params: MinHashParams {
                num_hashes: 128,
                similarity_threshold: 0.7,
            },
            cross_validation_enabled: true,
            stats: Arc::new(RwLock::new(CrossValidationStats::default())),
        }
    }
    
    pub async fn initialize(&self, _config: &Config) -> Result<()> {
        // Register enhanced strategies
        self.register_strategy("intelligent", Box::new(IntelligentFusion::new()));
        self.register_strategy("weighted", Box::new(WeightedFusion));
        self.register_strategy("reciprocal_rank", Box::new(ReciprocalRankFusion));
        self.register_strategy("borda_count", Box::new(BordaCountFusion));
        self.register_strategy("hybrid_rrf", Box::new(HybridRRFFusion::new()));
        
        debug!("Fusion engine initialized with intelligent deduplication and cross-validation");
        Ok(())
    }
    
    #[instrument(skip(self, results))]
    pub async fn fuse(&self, results: Vec<ProcessedResult>) -> Result<Vec<ProcessedResult>> {
        let start = Instant::now();
        let total_results = results.len();
        
        // Apply intelligent deduplication
        let deduplicated = self.intelligent_deduplication(results).await?;
        let unique_count = deduplicated.len();
        
        // Apply cross-validation and reranking
        let final_results = if self.cross_validation_enabled {
            self.cross_validate_and_rerank(deduplicated).await?
        } else {
            self.rerank(deduplicated).await?
        };
        
        // Update stats
        let mut stats = self.stats.write();
        stats.total_results = total_results;
        stats.unique_results = unique_count;
        stats.deduplicated = total_results - unique_count;
        stats.fusion_time_ms = start.elapsed().as_millis();
        
        debug!("Fusion completed in {}ms: {} -> {} results", 
               stats.fusion_time_ms, total_results, final_results.len());
        
        Ok(final_results)
    }
    
    /// Fuse SearchResults into ProcessedResults with component scoring
    pub async fn fuse_search_results(
        &self,
        search_results: Vec<crate::pipeline::search_orchestrator::SearchResult>,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<ProcessedResult>> {
        let start = Instant::now();
        
        // Convert SearchResults to ProcessedResults
        let processed: Vec<ProcessedResult> = search_results
            .into_par_iter()
            .map(|sr| ProcessedResult {
                score: sr.score,
                content: sr.content,
                source: crate::core::types::DataSource::Database,
                metadata: sr.metadata,
            })
            .collect();
        
        // Apply intelligent fusion
        let fused = self.fuse(processed).await?;
        
        // Select top-k using efficient selection
        let top_k = self.select_top_k_efficient(fused, 8);
        
        debug!("Search fusion completed in {:?}", start.elapsed());
        Ok(top_k)
    }
    
    /// Efficient top-k selection using BinaryHeap
    pub fn select_top_k_efficient(&self, results: Vec<ProcessedResult>, k: usize) -> Vec<ProcessedResult> {
        if results.len() <= k {
            return results;
        }
        
        let mut heap: BinaryHeap<OrderedResult> = BinaryHeap::with_capacity(k);
        
        for result in results {
            heap.push(OrderedResult(result));
            if heap.len() > k {
                heap.pop(); // Remove smallest
            }
        }
        
        let mut top_k: Vec<ProcessedResult> = heap.into_sorted_vec()
            .into_iter()
            .map(|ordered| ordered.0)
            .collect();
        
        // Sort in descending order
        top_k.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        top_k
    }
    
    pub async fn fuse_multiple(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let strategy_name = self.current_strategy.read().clone();
        
        if let Some(strategy) = self.strategies.get(&strategy_name) {
            strategy.fuse(result_sets).await
        } else {
            // Fallback to simple concatenation
            Ok(result_sets.into_iter().flatten().collect())
        }
    }
    
    async fn rerank(&self, mut results: Vec<ProcessedResult>) -> Result<Vec<ProcessedResult>> {
        let matrix = self.scoring_matrix.read().clone();
        
        // Compute 6-factor composite scores in parallel
        results.par_iter_mut().for_each(|result| {
            let relevance = result.score;
            let freshness = compute_freshness_score(&result.metadata);
            let diversity = compute_diversity_score(&result.content);
            let authority = compute_authority_score(&result.metadata);
            let coherence = compute_coherence_score(&result.metadata);
            let confidence = compute_confidence_score(result.score, &result.metadata);
            
            result.score = 
                relevance * matrix.relevance +
                freshness * matrix.freshness +
                diversity * matrix.diversity +
                authority * matrix.authority +
                coherence * matrix.coherence +
                confidence * matrix.confidence;
        });
        
        // Sort by composite score
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply diversity filtering
        let diverse_results = apply_diversity_filter(results);
        
        Ok(diverse_results)
    }
    
    /// Intelligent deduplication using MinHash signatures
    async fn intelligent_deduplication(&self, results: Vec<ProcessedResult>) -> Result<Vec<ProcessedResult>> {
        if results.is_empty() {
            return Ok(results);
        }
        
        // Generate MinHash signatures in parallel
        let signatures: Vec<MinHashSignature> = results
            .par_iter()
            .map(|r| self.generate_minhash_signature(&r.content))
            .collect();
        
        // Group similar results
        let mut unique_results = Vec::new();
        let mut seen_groups = AHashSet::new();
        
        for (idx, result) in results.into_iter().enumerate() {
            let sig = &signatures[idx];
            let mut is_duplicate = false;
            
            // Check similarity with existing groups
            for seen_idx in &seen_groups {
                let seen_sig = &signatures[*seen_idx];
                if self.compute_jaccard_similarity(sig, seen_sig) > self.minhash_params.similarity_threshold {
                    is_duplicate = true;
                    break;
                }
            }
            
            if !is_duplicate {
                seen_groups.insert(idx);
                unique_results.push(result);
            }
        }
        
        Ok(unique_results)
    }
    
    /// Cross-validate results from multiple sources
    async fn cross_validate_and_rerank(&self, mut results: Vec<ProcessedResult>) -> Result<Vec<ProcessedResult>> {
        // Group results by source
        let mut source_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, result) in results.iter().enumerate() {
            let source = result.metadata.get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            source_groups.entry(source).or_default().push(idx);
        }
        
        // Boost scores for cross-validated results
        for (idx, result) in results.iter_mut().enumerate() {
            let content_hash = xxhash3_64(&result.content);
            let mut validation_count = 0;
            
            for (source, indices) in &source_groups {
                if indices.contains(&idx) {
                    continue;
                }
                
                // Check if similar content exists in other sources
                for &other_idx in indices {
                    if other_idx != idx {
                        let other = &results[other_idx];
                        let other_hash = xxhash3_64(&other.content);
                        
                        // Simple similarity check
                        if content_hash == other_hash || 
                           self.text_similarity(&result.content, &other.content) > 0.8 {
                            validation_count += 1;
                        }
                    }
                }
            }
            
            // Boost score based on cross-validation
            if validation_count > 0 {
                result.score *= 1.0 + (validation_count as f32 * 0.1).min(0.5);
                result.metadata.insert(
                    "cross_validated".to_string(),
                    serde_json::json!(validation_count)
                );
            }
        }
        
        // Update stats
        let cross_validated = results.iter()
            .filter(|r| r.metadata.contains_key("cross_validated"))
            .count();
        self.stats.write().cross_validated = cross_validated;
        
        // Final reranking
        self.rerank(results).await
    }
    
    /// Generate MinHash signature for content
    fn generate_minhash_signature(&self, content: &str) -> MinHashSignature {
        let shingles = self.generate_shingles(content, 3);
        let mut min_hashes = vec![u64::MAX; self.minhash_params.num_hashes];
        
        for shingle in shingles {
            for i in 0..self.minhash_params.num_hashes {
                let hash = xxhash3_64(&format!("{}{}", shingle, i));
                min_hashes[i] = min_hashes[i].min(hash);
            }
        }
        
        MinHashSignature {
            hashes: min_hashes,
            content_hash: xxhash3_64(content),
        }
    }
    
    /// Generate character-level shingles
    fn generate_shingles(&self, text: &str, k: usize) -> HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut shingles = HashSet::new();
        
        if chars.len() >= k {
            for i in 0..=chars.len() - k {
                let shingle: String = chars[i..i + k].iter().collect();
                shingles.insert(shingle);
            }
        }
        
        shingles
    }
    
    /// Compute Jaccard similarity between MinHash signatures
    fn compute_jaccard_similarity(&self, sig1: &MinHashSignature, sig2: &MinHashSignature) -> f32 {
        let matches = sig1.hashes.iter()
            .zip(sig2.hashes.iter())
            .filter(|(h1, h2)| h1 == h2)
            .count();
        
        matches as f32 / self.minhash_params.num_hashes as f32
    }
    
    /// Simple text similarity using token overlap
    fn text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let tokens1: HashSet<&str> = text1.split_whitespace().collect();
        let tokens2: HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = tokens1.intersection(&tokens2).count();
        let union = tokens1.union(&tokens2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
    
    pub fn register_strategy(&self, name: &str, strategy: Box<dyn FusionStrategy>) {
        self.strategies.insert(name.to_string(), strategy);
    }
    
    pub fn set_strategy(&self, name: &str) -> Result<()> {
        if self.strategies.contains_key(name) {
            *self.current_strategy.write() = name.to_string();
            Ok(())
        } else {
            Err(crate::core::NexusError::NotFound(format!("Strategy '{}' not found", name)))
        }
    }
    
    pub fn set_scoring_matrix(&self, matrix: ScoringMatrix) {
        *self.scoring_matrix.write() = matrix;
    }
    
    pub fn get_stats(&self) -> CrossValidationStats {
        self.stats.read().clone()
    }
    
    pub fn enable_cross_validation(&self, enabled: bool) {
        self.cross_validation_enabled = enabled;
    }
}

// ===== FUSION STRATEGIES =====

/// Intelligent fusion with advanced scoring and deduplication
struct IntelligentFusion {
    k_param: f32,  // RRF constant
}

impl IntelligentFusion {
    fn new() -> Self {
        Self { k_param: 60.0 }
    }
}

#[async_trait]
impl FusionStrategy for IntelligentFusion {
    async fn fuse(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let mut fusion_scores: DashMap<u64, (ProcessedResult, f32, Vec<usize>)> = DashMap::new();
        
        // Process each result set with RRF + weighted combination
        for (set_idx, results) in result_sets.into_iter().enumerate() {
            let set_weight = 1.0 / (set_idx + 1) as f32;
            
            for (rank, result) in results.into_iter().enumerate() {
                // Use content hash for deduplication
                let key = xxhash3_64(&result.content);
                
                // RRF score combined with original score
                let rrf_score = 1.0 / (self.k_param + rank as f32 + 1.0);
                let combined_score = (result.score * 0.7 + rrf_score * 0.3) * set_weight;
                
                fusion_scores
                    .entry(key)
                    .and_modify(|(stored, score, sources)| {
                        // Keep the result with better metadata
                        if result.metadata.len() > stored.metadata.len() {
                            *stored = result.clone();
                        }
                        *score += combined_score;
                        sources.push(set_idx);
                    })
                    .or_insert((result, combined_score, vec![set_idx]));
            }
        }
        
        // Convert to final results with source diversity bonus
        let mut final_results: Vec<_> = fusion_scores
            .into_iter()
            .map(|(_, (mut result, score, sources))| {
                // Bonus for appearing in multiple sources
                let diversity_bonus = (sources.len() as f32).sqrt() / 2.0;
                result.score = score * (1.0 + diversity_bonus);
                result.metadata.insert(
                    "fusion_sources".to_string(),
                    serde_json::json!(sources.len())
                );
                result
            })
            .collect();
        
        // Sort by final score
        final_results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to top results for performance
        final_results.truncate(200);
        
        Ok(final_results)
    }
    
    fn name(&self) -> &str {
        "intelligent"
    }
}

/// Hybrid RRF fusion with MinHash deduplication
struct HybridRRFFusion {
    k_param: f32,
    similarity_threshold: f32,
}

impl HybridRRFFusion {
    fn new() -> Self {
        Self {
            k_param: 60.0,
            similarity_threshold: 0.85,
        }
    }
}

#[async_trait]
impl FusionStrategy for HybridRRFFusion {
    async fn fuse(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let mut rrf_scores: DashMap<u64, (ProcessedResult, f32, f32)> = DashMap::new();
        
        for results in result_sets {
            for (rank, result) in results.into_iter().enumerate() {
                let content_hash = xxhash3_64(&result.content);
                let rrf_score = 1.0 / (self.k_param + rank as f32 + 1.0);
                
                rrf_scores
                    .entry(content_hash)
                    .and_modify(|(_, rrf, orig)| {
                        *rrf += rrf_score;
                        *orig = orig.max(result.score);
                    })
                    .or_insert((result.clone(), rrf_score, result.score));
            }
        }
        
        let mut final_results: Vec<_> = rrf_scores
            .into_iter()
            .map(|(_, (mut result, rrf_score, orig_score))| {
                // Hybrid score combining RRF and original
                result.score = rrf_score * 0.6 + orig_score * 0.4;
                result
            })
            .collect();
        
        final_results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(final_results)
    }
    
    fn name(&self) -> &str {
        "hybrid_rrf"
    }
}

struct WeightedFusion;

#[async_trait]
impl FusionStrategy for WeightedFusion {
    async fn fuse(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let mut combined_scores: DashMap<String, (ProcessedResult, f32, usize)> = DashMap::new();
        
        for (set_idx, results) in result_sets.into_iter().enumerate() {
            let weight = 1.0 / (set_idx + 1) as f32; // Decreasing weights
            
            for (rank, result) in results.into_iter().enumerate() {
                let key = result.content.clone();
                combined_scores
                    .entry(key)
                    .and_modify(|(_, score, count)| {
                        *score += result.score * weight;
                        *count += 1;
                    })
                    .or_insert((result.clone(), result.score * weight, 1));
            }
        }
        
        let mut final_results: Vec<_> = combined_scores
            .into_iter()
            .map(|(_, (mut result, total_score, count))| {
                result.score = total_score / count as f32;
                result
            })
            .collect();
        
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(final_results)
    }
    
    fn name(&self) -> &str {
        "weighted"
    }
}

struct ReciprocalRankFusion;

#[async_trait]
impl FusionStrategy for ReciprocalRankFusion {
    async fn fuse(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let k = 60; // RRF constant
        let mut rrf_scores: DashMap<String, (ProcessedResult, f32)> = DashMap::new();
        
        for results in result_sets {
            for (rank, result) in results.into_iter().enumerate() {
                let rrf_score = 1.0 / (k + rank + 1) as f32;
                let key = result.content.clone();
                
                rrf_scores
                    .entry(key)
                    .and_modify(|(_, score)| *score += rrf_score)
                    .or_insert((result.clone(), rrf_score));
            }
        }
        
        let mut final_results: Vec<_> = rrf_scores
            .into_iter()
            .map(|(_, (mut result, score))| {
                result.score = score;
                result
            })
            .collect();
        
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(final_results)
    }
    
    fn name(&self) -> &str {
        "reciprocal_rank"
    }
}

struct BordaCountFusion;

#[async_trait]
impl FusionStrategy for BordaCountFusion {
    async fn fuse(&self, result_sets: Vec<Vec<ProcessedResult>>) -> Result<Vec<ProcessedResult>> {
        let mut borda_scores: DashMap<String, (ProcessedResult, usize)> = DashMap::new();
        
        for results in result_sets {
            let n = results.len();
            for (rank, result) in results.into_iter().enumerate() {
                let borda_points = n - rank;
                let key = result.content.clone();
                
                borda_scores
                    .entry(key)
                    .and_modify(|(_, points)| *points += borda_points)
                    .or_insert((result.clone(), borda_points));
            }
        }
        
        let mut final_results: Vec<_> = borda_scores
            .into_iter()
            .map(|(_, (mut result, points))| {
                result.score = points as f32;
                result
            })
            .collect();
        
        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(final_results)
    }
    
    fn name(&self) -> &str {
        "borda_count"
    }
}

// ===== HELPER STRUCTURES =====

/// Wrapper for BinaryHeap ordering
struct OrderedResult(ProcessedResult);

impl Eq for OrderedResult {}

impl PartialEq for OrderedResult {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.total_cmp(&other.0.score) == Ordering::Equal
    }
}

impl Ord for OrderedResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap (highest scores first)
        other.0.score.total_cmp(&self.0.score)
    }
}

impl PartialOrd for OrderedResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Component scores for detailed analysis
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ComponentScores {
    pub semantic: f32,
    pub keyword: f32,
    pub temporal: f32,
    pub quality: f32,
    pub domain: f32,
    pub user_pref: f32,
    pub cross_validation: f32,
}

// ===== HELPER FUNCTIONS =====

/// Compute coherence score based on cross-references
fn compute_coherence_score(metadata: &std::collections::HashMap<String, serde_json::Value>) -> f32 {
    metadata.get("cross_references")
        .and_then(|v| v.as_array())
        .map(|refs| {
            let ref_count = refs.len() as f32;
            (ref_count / 10.0).min(1.0)
        })
        .unwrap_or(0.3)
}

/// Compute confidence score based on multiple factors
fn compute_confidence_score(base_score: f32, metadata: &std::collections::HashMap<String, serde_json::Value>) -> f32 {
    let score_factor = base_score.min(1.0);
    
    let validation_factor = metadata.get("cross_validated")
        .and_then(|v| v.as_i64())
        .map(|count| (count as f32 / 5.0).min(1.0))
        .unwrap_or(0.5);
    
    let source_factor = metadata.get("fusion_sources")
        .and_then(|v| v.as_i64())
        .map(|count| (count as f32 / 3.0).min(1.0))
        .unwrap_or(0.5);
    
    (score_factor * 0.5 + validation_factor * 0.3 + source_factor * 0.2).min(1.0)
}

fn compute_freshness_score(metadata: &std::collections::HashMap<String, serde_json::Value>) -> f32 {
    // Placeholder: would compute based on timestamp
    metadata.get("timestamp")
        .and_then(|v| v.as_i64())
        .map(|_| 0.8)
        .unwrap_or(0.5)
}

fn compute_diversity_score(content: &str) -> f32 {
    // Placeholder: would compute based on content uniqueness
    let unique_words: std::collections::HashSet<_> = content.split_whitespace().collect();
    let total_words = content.split_whitespace().count();
    
    if total_words > 0 {
        unique_words.len() as f32 / total_words as f32
    } else {
        0.0
    }
}

fn compute_authority_score(metadata: &std::collections::HashMap<String, serde_json::Value>) -> f32 {
    // Placeholder: would compute based on source authority
    metadata.get("source_authority")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.5)
}

fn apply_diversity_filter(results: Vec<ProcessedResult>) -> Vec<ProcessedResult> {
    if results.is_empty() {
        return results;
    }
    
    let mut diverse_results = Vec::new();
    let mut seen_content = std::collections::HashSet::new();
    
    for result in results {
        // Simple diversity: avoid exact duplicates
        let content_hash = dedup_hash(&result.content);
        if !seen_content.contains(&content_hash) {
            seen_content.insert(content_hash);
            diverse_results.push(result);
        }
    }
    
    diverse_results
}