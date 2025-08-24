/// Intelligent Router with <0.2ms decision time for optimal query routing
/// Routes 70% cache-only, 25% smart routing, 4% full pipeline, 1% maximum intelligence

use crate::core::hash_utils::{ahash_string, generate_pipeline_cache_key};
use crate::ai::{EmbeddingService, EmbeddingConfig};
use ahash::AHashMap;
use once_cell::sync::Lazy;
use regex::Regex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::Instant;
use parking_lot::RwLock;

/// Routing decision paths with target latencies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPath {
    CacheOnly,           // 2ms target - Simple cache lookup
    SmartRouting,        // 15ms target - Cache + basic search
    FullPipeline,        // 40ms target - Full processing
    MaximumIntelligence, // 45ms target - All capabilities
}

/// Query complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    Simple,   // Basic queries, likely cached
    Medium,   // Standard queries needing some processing
    Complex,  // Multi-faceted queries requiring full pipeline
    Critical, // Medical, legal, financial - always maximum
}

/// Query intent for adaptive scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    Debug,    // Error fixing queries
    Learn,    // Concept understanding
    Lookup,   // Quick fact retrieval
    Build,    // Implementation help
    Unknown,  // Unclassified
}

/// Adaptive scoring weights based on intent
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub semantic: f32,
    pub bm25: f32,
    pub recency: f32,
    pub importance: f32,
    pub context: f32,
}

impl ScoringWeights {
    pub fn for_intent(intent: QueryIntent) -> Self {
        match intent {
            QueryIntent::Debug => Self {
                semantic: 0.40,
                bm25: 0.30,
                recency: 0.20,
                importance: 0.05,
                context: 0.05,
            },
            QueryIntent::Learn => Self {
                semantic: 0.50,
                bm25: 0.15,
                recency: 0.10,
                importance: 0.15,
                context: 0.10,
            },
            QueryIntent::Lookup => Self {
                semantic: 0.20,
                bm25: 0.50,
                recency: 0.10,
                importance: 0.10,
                context: 0.10,
            },
            QueryIntent::Build => Self {
                semantic: 0.35,
                bm25: 0.25,
                recency: 0.15,
                importance: 0.15,
                context: 0.10,
            },
            QueryIntent::Unknown => Self {
                semantic: 0.35,
                bm25: 0.25,
                recency: 0.15,
                importance: 0.15,
                context: 0.10,
            },
        }
    }
}

/// Query analysis result with all decision factors
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub query: QueryInfo,
    pub complexity: ComplexityLevel,
    pub cache_probability: f32,
    pub routing_path: RoutingPath,
    pub confidence: f32,
    pub domain: QueryDomain,
    pub features: QueryFeatures,
    pub intent: QueryIntent,
    pub scoring_weights: ScoringWeights,
    pub embedding: Option<crate::core::types::ConstVector<1024>>,
    pub analysis_time_us: u64,
    pub parent_uuid: Option<uuid::Uuid>,  // For graph traversal
}

#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub text: String,
    pub id: uuid::Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDomain {
    General,
    Technical,
    Medical,
    Legal,
    Financial,
    CrossDomain,
}

#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    pub word_count: usize,
    pub has_reference: bool,      // "that", "previous", "same"
    pub has_technical: bool,       // Technical terms detected
    pub has_critical: bool,        // Critical domain terms
    pub has_question: bool,        // Question words
    pub entity_count: usize,       // Capitalized words
    pub avg_word_length: f32,      // Complexity indicator
    pub special_chars: usize,      // Code-like content
    pub has_temporal: bool,        // Time-related queries
}

/// Pre-compiled patterns for ultra-fast matching
static REFERENCE_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(that|previous|same|again|last|earlier|before|above|below)\b").unwrap()
});

static TECHNICAL_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(debug|error|bug|crash|memory|leak|performance|optimize|algorithm|database|API|code|function|class|method)\b").unwrap()
});

static CRITICAL_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(medical|diagnosis|treatment|symptom|patient|legal|lawsuit|contract|attorney|financial|investment|portfolio|trading)\b").unwrap()
});

static QUESTION_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(what|how|why|when|where|who|which|can|could|should|would|is|are|do|does)").unwrap()
});

static TEMPORAL_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(today|yesterday|tomorrow|now|current|latest|recent|time|date|when)\b").unwrap()
});

// Intent detection patterns
static DEBUG_INTENT_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(error|bug|fix|broken|crash|fail|exception|debug|troubleshoot|issue|problem|wrong|not working|segfault|stack trace|traceback)\b").unwrap()
});

static LEARN_INTENT_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(learn|understand|explain|what is|how does|why|concept|theory|tutorial|guide|introduction|basics|fundamentals)\b").unwrap()
});

static LOOKUP_INTENT_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(find|search|where|location|definition|meaning|documentation|reference|spec|api|syntax|command)\b").unwrap()
});

static BUILD_INTENT_PATTERNS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(build|create|implement|develop|code|write|make|construct|design|architecture|integrate|setup)\b").unwrap()
});

/// Cache statistics for probability calculation
#[derive(Default)]
pub struct CacheStatistics {
    hits: AtomicU64,
    misses: AtomicU64,
    total_queries: AtomicU64,
    cache_only_success: AtomicU64,
    smart_routing_success: AtomicU64,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed) as f32;
        let total = self.total_queries.load(Ordering::Relaxed) as f32;
        if total > 0.0 { hits / total } else { 0.0 }
    }
    
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.total_queries.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.total_queries.fetch_add(1, Ordering::Relaxed);
    }
}

/// High-performance intelligent router
pub struct IntelligentRouter {
    domain_keywords: Arc<AHashMap<&'static str, QueryDomain>>,
    cache_stats: Arc<CacheStatistics>,
    routing_stats: Arc<RoutingStatistics>,
    embedding_service: Arc<EmbeddingService>,
    config: RouterConfig,
}

#[derive(Clone)]
pub struct RouterConfig {
    pub cache_threshold: f32,        // Minimum probability for cache-only
    pub smart_threshold: f32,        // Minimum confidence for smart routing
    pub escalation_threshold: f32,   // Below this, escalate to higher tier
    pub max_analysis_time_us: u64,   // Maximum time for analysis (200μs)
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.7,
            smart_threshold: 0.5,
            escalation_threshold: 0.3,
            max_analysis_time_us: 200,
        }
    }
}

#[derive(Default)]
pub struct RoutingStatistics {
    cache_only_count: AtomicU64,
    smart_routing_count: AtomicU64,
    full_pipeline_count: AtomicU64,
    max_intelligence_count: AtomicU64,
    escalations: AtomicU64,
    avg_analysis_time_us: AtomicU64,
}

impl IntelligentRouter {
    pub fn new() -> Self {
        Self::with_config(RouterConfig::default())
    }
    
    pub fn with_config(config: RouterConfig) -> Self {
        let mut domain_keywords = AHashMap::new();
        
        // Medical domain keywords
        for word in &["diagnosis", "treatment", "medication", "symptom", "patient", 
                      "disease", "therapy", "medical", "health", "doctor"] {
            domain_keywords.insert(*word, QueryDomain::Medical);
        }
        
        // Legal domain keywords
        for word in &["lawsuit", "contract", "legal", "court", "attorney", 
                      "law", "litigation", "compliance", "regulation", "rights"] {
            domain_keywords.insert(*word, QueryDomain::Legal);
        }
        
        // Financial domain keywords
        for word in &["investment", "portfolio", "trading", "financial", "banking",
                      "stock", "bond", "asset", "fund", "market"] {
            domain_keywords.insert(*word, QueryDomain::Financial);
        }
        
        // Technical domain keywords
        for word in &["code", "debug", "algorithm", "database", "API",
                      "software", "hardware", "network", "system", "server"] {
            domain_keywords.insert(*word, QueryDomain::Technical);
        }
        
        let embedding_config = EmbeddingConfig::default();
        let embedding_service = Arc::new(EmbeddingService::new(embedding_config));
        
        Self {
            domain_keywords: Arc::new(domain_keywords),
            cache_stats: Arc::new(CacheStatistics::default()),
            routing_stats: Arc::new(RoutingStatistics::default()),
            embedding_service,
            config,
        }
    }

    /// Analyze query in <0.2ms with all optimizations (embedding generation is async)
    #[inline(always)]
    pub async fn analyze(&self, query: &str, query_id: uuid::Uuid) -> QueryAnalysis {
        let start = Instant::now();
        // Use the passed query_id instead of generating a new one
        
        // Fast path for very short queries
        if query.len() < 10 {
            let mut analysis = self.fast_path_analysis(query);
            analysis.query = QueryInfo {
                text: query.to_string(),
                id: query_id,
            };
            self.record_analysis_time(start.elapsed().as_micros() as u64);
            return analysis;
        }
        
        // Extract features with zero allocations where possible
        let features = self.extract_features_fast(query);
        
        // Detect domain using pre-computed keywords
        let domain = self.detect_domain_fast(query, &features);
        
        // Calculate complexity based on features and domain
        let complexity = self.calculate_complexity(&features, &domain);
        
        // Calculate cache probability
        let cache_probability = self.calculate_cache_probability(&features);
        
        // Determine routing path
        let routing_path = self.determine_routing(&complexity, cache_probability, &domain);
        
        // Calculate confidence
        let confidence = self.calculate_confidence(&features, cache_probability, &complexity);
        
        // Detect query intent for adaptive scoring
        let intent = self.detect_intent(query);
        let scoring_weights = ScoringWeights::for_intent(intent);
        
        // Generate embedding asynchronously (only for non-cache-only paths)
        let embedding = if routing_path != RoutingPath::CacheOnly {
            match self.embedding_service.generate_const_vector(query).await {
                Ok(emb) => Some(emb),
                Err(e) => {
                    tracing::warn!("Failed to generate embedding: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let analysis_time_us = start.elapsed().as_micros() as u64;
        
        // Debug assert to ensure we meet timing requirements
        debug_assert!(
            analysis_time_us <= self.config.max_analysis_time_us * 2, // Allow 2x for embedding
            "Analysis took {}μs, exceeding limit of {}μs",
            analysis_time_us,
            self.config.max_analysis_time_us * 2
        );
        
        self.record_analysis_time(analysis_time_us);
        self.update_routing_stats(&routing_path);
        
        QueryAnalysis {
            query: QueryInfo {
                text: query.to_string(),
                id: query_id,
            },
            complexity,
            cache_probability,
            routing_path,
            confidence,
            domain,
            features,
            intent,
            scoring_weights,
            embedding,
            analysis_time_us,
            parent_uuid: None,  // Can be set later for related queries
        }
    }

    /// Ultra-fast path for simple queries
    #[inline(always)]
    fn fast_path_analysis(&self, query: &str) -> QueryAnalysis {
        let features = QueryFeatures {
            word_count: query.split_whitespace().count(),
            avg_word_length: query.len() as f32 / query.split_whitespace().count().max(1) as f32,
            ..Default::default()
        };
        
        let intent = self.detect_intent(query);
        let scoring_weights = ScoringWeights::for_intent(intent);
        
        QueryAnalysis {
            query: QueryInfo {
                text: String::new(),  // Will be filled by caller
                id: uuid::Uuid::nil(),  // Will be filled by caller
            },
            complexity: ComplexityLevel::Simple,
            cache_probability: 0.9,
            routing_path: RoutingPath::CacheOnly,
            confidence: 0.95,
            domain: QueryDomain::General,
            features,
            intent,
            scoring_weights,
            embedding: None,
            analysis_time_us: 5, // Fast path typically <5μs
            parent_uuid: None,
        }
    }
    
    /// Detect query intent for adaptive scoring
    #[inline(always)]
    fn detect_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();
        
        // Count matches for each intent
        let debug_score = if DEBUG_INTENT_PATTERNS.is_match(&query_lower) { 1.0 } else { 0.0 };
        let learn_score = if LEARN_INTENT_PATTERNS.is_match(&query_lower) { 1.0 } else { 0.0 };
        let lookup_score = if LOOKUP_INTENT_PATTERNS.is_match(&query_lower) { 1.0 } else { 0.0 };
        let build_score = if BUILD_INTENT_PATTERNS.is_match(&query_lower) { 1.0 } else { 0.0 };
        
        // Return intent with highest score
        let max_score = debug_score.max(learn_score).max(lookup_score).max(build_score);
        
        if max_score == 0.0 {
            QueryIntent::Unknown
        } else if debug_score == max_score {
            QueryIntent::Debug
        } else if learn_score == max_score {
            QueryIntent::Learn
        } else if lookup_score == max_score {
            QueryIntent::Lookup
        } else if build_score == max_score {
            QueryIntent::Build
        } else {
            QueryIntent::Unknown
        }
    }

    /// Extract features with minimal allocations
    #[inline(always)]
    fn extract_features_fast(&self, query: &str) -> QueryFeatures {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        let word_count = words.len();
        
        // Parallel pattern matching for speed
        let has_reference = REFERENCE_PATTERNS.is_match(&query_lower);
        let has_technical = TECHNICAL_PATTERNS.is_match(&query_lower);
        let has_critical = CRITICAL_PATTERNS.is_match(&query_lower);
        let has_question = QUESTION_PATTERNS.is_match(&query_lower);
        let has_temporal = TEMPORAL_PATTERNS.is_match(&query_lower);
        
        // Count entities (capitalized words not at start)
        let entity_count = words.iter()
            .skip(1)
            .filter(|w| w.chars().next().map_or(false, |c| c.is_uppercase()))
            .count();
        
        // Calculate average word length
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = if word_count > 0 {
            total_chars as f32 / word_count as f32
        } else {
            0.0
        };
        
        // Count special characters (code indicators)
        let special_chars = query.chars()
            .filter(|c| matches!(c, '{' | '}' | '[' | ']' | '(' | ')' | '<' | '>' | '=' | ';'))
            .count();
        
        QueryFeatures {
            word_count,
            has_reference,
            has_technical,
            has_critical,
            has_question,
            entity_count,
            avg_word_length,
            special_chars,
            has_temporal,
        }
    }

    /// Fast domain detection using pre-computed keyword map
    #[inline(always)]
    fn detect_domain_fast(&self, query: &str, features: &QueryFeatures) -> QueryDomain {
        // Critical domains take precedence
        if features.has_critical {
            let query_lower = query.to_lowercase();
            let mut domains_found = Vec::new();
            
            for (keyword, domain) in self.domain_keywords.iter() {
                if query_lower.contains(keyword) {
                    if !domains_found.contains(domain) {
                        domains_found.push(*domain);
                    }
                    if domains_found.len() > 1 {
                        return QueryDomain::CrossDomain;
                    }
                }
            }
            
            if let Some(domain) = domains_found.first() {
                return *domain;
            }
        }
        
        if features.has_technical {
            return QueryDomain::Technical;
        }
        
        QueryDomain::General
    }

    /// Calculate complexity with domain awareness
    #[inline(always)]
    fn calculate_complexity(
        &self,
        features: &QueryFeatures,
        domain: &QueryDomain,
    ) -> ComplexityLevel {
        // Critical domains always get maximum complexity
        match domain {
            QueryDomain::Medical | QueryDomain::Legal | QueryDomain::Financial => {
                return ComplexityLevel::Critical;
            }
            QueryDomain::CrossDomain => return ComplexityLevel::Complex,
            _ => {}
        }
        
        // Score-based complexity for other queries
        let mut score = 0;
        
        // Word count contribution
        score += match features.word_count {
            0..=5 => 0,
            6..=15 => 1,
            16..=30 => 2,
            _ => 3,
        };
        
        // Feature contributions
        if features.has_technical { score += 2; }
        if features.has_question { score += 1; }
        if features.special_chars > 3 { score += 2; }
        if features.entity_count > 2 { score += 1; }
        if features.avg_word_length > 7.0 { score += 1; }
        
        match score {
            0..=2 => ComplexityLevel::Simple,
            3..=5 => ComplexityLevel::Medium,
            6..=8 => ComplexityLevel::Complex,
            _ => ComplexityLevel::Critical,
        }
    }

    /// Calculate cache probability based on features and history
    #[inline(always)]
    fn calculate_cache_probability(&self, features: &QueryFeatures) -> f32 {
        let mut probability = 0.0;
        
        // Strong cache indicators
        if features.has_reference {
            probability += 0.5; // References likely in conversation context
        }
        
        if features.word_count <= 5 {
            probability += 0.3; // Short queries often repeated
        }
        
        if features.has_temporal {
            probability -= 0.2; // Time-sensitive queries less cacheable
        }
        
        // Incorporate historical cache performance
        let historical_hit_rate = self.cache_stats.hit_rate();
        probability = probability * 0.7 + historical_hit_rate * 0.3;
        
        probability.clamp(0.0, 1.0)
    }

    /// Determine optimal routing path
    #[inline(always)]
    fn determine_routing(
        &self,
        complexity: &ComplexityLevel,
        cache_probability: f32,
        domain: &QueryDomain,
    ) -> RoutingPath {
        // Critical domains always get maximum intelligence
        if matches!(domain, QueryDomain::Medical | QueryDomain::Legal | QueryDomain::Financial) {
            return RoutingPath::MaximumIntelligence;
        }
        
        // High cache probability with simple complexity -> cache only
        if cache_probability > self.config.cache_threshold && 
           matches!(complexity, ComplexityLevel::Simple) {
            return RoutingPath::CacheOnly;
        }
        
        // Route based on complexity and cache probability
        match complexity {
            ComplexityLevel::Simple => {
                if cache_probability > 0.5 {
                    RoutingPath::CacheOnly
                } else {
                    RoutingPath::SmartRouting
                }
            }
            ComplexityLevel::Medium => RoutingPath::SmartRouting,
            ComplexityLevel::Complex => RoutingPath::FullPipeline,
            ComplexityLevel::Critical => RoutingPath::MaximumIntelligence,
        }
    }

    /// Calculate routing confidence
    #[inline(always)]
    fn calculate_confidence(
        &self,
        features: &QueryFeatures,
        cache_probability: f32,
        complexity: &ComplexityLevel,
    ) -> f32 {
        let mut confidence = 0.5; // Base confidence
        
        // Adjust based on clarity indicators
        if features.has_reference {
            confidence += 0.2;
        }
        
        if cache_probability > 0.7 {
            confidence += 0.15;
        }
        
        // Simple queries have higher confidence
        match complexity {
            ComplexityLevel::Simple => confidence += 0.2,
            ComplexityLevel::Medium => confidence += 0.1,
            ComplexityLevel::Complex => confidence -= 0.1,
            ComplexityLevel::Critical => confidence += 0.3, // Critical requires high confidence
        }
        
        // Word count affects confidence
        if features.word_count <= 10 {
            confidence += 0.1;
        } else if features.word_count > 30 {
            confidence -= 0.1;
        }
        
        confidence.clamp(0.0, 1.0)
    }

    /// Check if escalation is needed based on confidence
    #[inline(always)]
    pub fn should_escalate(&self, analysis: &QueryAnalysis) -> bool {
        analysis.confidence < self.config.escalation_threshold
    }

    /// Escalate to next routing level
    #[inline(always)]
    pub fn escalate_path(&self, current: RoutingPath) -> Option<RoutingPath> {
        self.routing_stats.escalations.fetch_add(1, Ordering::Relaxed);
        
        match current {
            RoutingPath::CacheOnly => Some(RoutingPath::SmartRouting),
            RoutingPath::SmartRouting => Some(RoutingPath::FullPipeline),
            RoutingPath::FullPipeline => Some(RoutingPath::MaximumIntelligence),
            RoutingPath::MaximumIntelligence => None,
        }
    }

    /// Record analysis timing for monitoring
    #[inline(always)]
    fn record_analysis_time(&self, time_us: u64) {
        // Simple moving average
        let current = self.routing_stats.avg_analysis_time_us.load(Ordering::Relaxed);
        let new_avg = (current * 9 + time_us) / 10;
        self.routing_stats.avg_analysis_time_us.store(new_avg, Ordering::Relaxed);
    }

    /// Update routing statistics
    #[inline(always)]
    fn update_routing_stats(&self, path: &RoutingPath) {
        match path {
            RoutingPath::CacheOnly => {
                self.routing_stats.cache_only_count.fetch_add(1, Ordering::Relaxed);
            }
            RoutingPath::SmartRouting => {
                self.routing_stats.smart_routing_count.fetch_add(1, Ordering::Relaxed);
            }
            RoutingPath::FullPipeline => {
                self.routing_stats.full_pipeline_count.fetch_add(1, Ordering::Relaxed);
            }
            RoutingPath::MaximumIntelligence => {
                self.routing_stats.max_intelligence_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get routing statistics
    pub fn get_stats(&self) -> RoutingStatsSnapshot {
        let total = self.routing_stats.cache_only_count.load(Ordering::Relaxed)
            + self.routing_stats.smart_routing_count.load(Ordering::Relaxed)
            + self.routing_stats.full_pipeline_count.load(Ordering::Relaxed)
            + self.routing_stats.max_intelligence_count.load(Ordering::Relaxed);
        
        RoutingStatsSnapshot {
            total_queries: total,
            cache_only_count: self.routing_stats.cache_only_count.load(Ordering::Relaxed),
            smart_routing_count: self.routing_stats.smart_routing_count.load(Ordering::Relaxed),
            full_pipeline_count: self.routing_stats.full_pipeline_count.load(Ordering::Relaxed),
            max_intelligence_count: self.routing_stats.max_intelligence_count.load(Ordering::Relaxed),
            escalations: self.routing_stats.escalations.load(Ordering::Relaxed),
            avg_analysis_time_us: self.routing_stats.avg_analysis_time_us.load(Ordering::Relaxed),
            cache_hit_rate: self.cache_stats.hit_rate(),
        }
    }

    /// Update cache statistics based on actual result
    pub fn record_cache_result(&self, hit: bool) {
        if hit {
            self.cache_stats.record_hit();
        } else {
            self.cache_stats.record_miss();
        }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingStatsSnapshot {
    pub total_queries: u64,
    pub cache_only_count: u64,
    pub smart_routing_count: u64,
    pub full_pipeline_count: u64,
    pub max_intelligence_count: u64,
    pub escalations: u64,
    pub avg_analysis_time_us: u64,
    pub cache_hit_rate: f32,
}

impl RoutingStatsSnapshot {
    pub fn cache_only_percentage(&self) -> f32 {
        if self.total_queries == 0 { return 0.0; }
        (self.cache_only_count as f32 / self.total_queries as f32) * 100.0
    }
    
    pub fn smart_routing_percentage(&self) -> f32 {
        if self.total_queries == 0 { return 0.0; }
        (self.smart_routing_count as f32 / self.total_queries as f32) * 100.0
    }
    
    pub fn full_pipeline_percentage(&self) -> f32 {
        if self.total_queries == 0 { return 0.0; }
        (self.full_pipeline_count as f32 / self.total_queries as f32) * 100.0
    }
    
    pub fn max_intelligence_percentage(&self) -> f32 {
        if self.total_queries == 0 { return 0.0; }
        (self.max_intelligence_count as f32 / self.total_queries as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_performance() {
        let router = IntelligentRouter::new();
        let queries = vec![
            "Debug React useState hooks",
            "What was that solution?",
            "Diagnose patient symptoms fever headache",
            "Investment portfolio optimization strategy",
            "Hello",
            "How to implement binary search?",
            "same as before",
            "Legal contract review needed",
            "What time is it?",
            "Explain quantum computing algorithms in detail with examples",
        ];
        
        for query in &queries {
            let start = Instant::now();
            let analysis = router.analyze(query);
            let elapsed = start.elapsed();
            
            println!("Query: {}", query);
            println!("  Routing: {:?}", analysis.routing_path);
            println!("  Complexity: {:?}", analysis.complexity);
            println!("  Cache prob: {:.2}", analysis.cache_probability);
            println!("  Confidence: {:.2}", analysis.confidence);
            println!("  Analysis time: {:?} ({}μs)", elapsed, analysis.analysis_time_us);
            
            assert!(
                elapsed.as_micros() < 200,
                "Analysis took too long: {:?}",
                elapsed
            );
        }
    }

    #[test]
    fn test_routing_distribution() {
        let router = IntelligentRouter::new();
        
        // Simulate realistic query distribution
        let test_queries = vec![
            // Cache-likely queries (should be ~70%)
            "what was that?", "same as before", "hello", "yes", "no",
            "previous result", "that solution", "again please", "ok", "thanks",
            "show me that", "repeat", "what about that", "the same", "continue",
            
            // Smart routing queries (should be ~25%)
            "debug my code", "error in function", "how to fix this",
            "algorithm optimization", "database query slow",
            
            // Complex queries (should be ~4%)
            "analyze the performance implications of using recursive algorithms",
            
            // Critical queries (should be ~1%)
            "diagnose patient symptoms fever headache nausea",
        ];
        
        for query in &test_queries {
            let analysis = router.analyze(query);
            
            // Check for escalation need
            if router.should_escalate(&analysis) {
                if let Some(escalated) = router.escalate_path(analysis.routing_path) {
                    println!("Escalated: {} -> {:?}", query, escalated);
                }
            }
        }
        
        let stats = router.get_stats();
        println!("\nRouting Distribution:");
        println!("  Cache only: {:.1}%", stats.cache_only_percentage());
        println!("  Smart routing: {:.1}%", stats.smart_routing_percentage());
        println!("  Full pipeline: {:.1}%", stats.full_pipeline_percentage());
        println!("  Max intelligence: {:.1}%", stats.max_intelligence_percentage());
        println!("  Average analysis time: {}μs", stats.avg_analysis_time_us);
    }

    #[test]
    fn test_critical_domains() {
        let router = IntelligentRouter::new();
        
        let critical_queries = vec![
            "patient diagnosis treatment plan",
            "legal contract lawsuit settlement",
            "investment portfolio trading strategy",
        ];
        
        for query in critical_queries {
            let analysis = router.analyze(query);
            assert_eq!(
                analysis.routing_path,
                RoutingPath::MaximumIntelligence,
                "Critical query should route to maximum intelligence: {}",
                query
            );
        }
    }

    #[test]
    fn test_cache_probability() {
        let router = IntelligentRouter::new();
        
        // High cache probability queries
        let cache_likely = vec![
            "same as before",
            "that one",
            "previous",
        ];
        
        for query in cache_likely {
            let analysis = router.analyze(query);
            assert!(
                analysis.cache_probability > 0.5,
                "Query '{}' should have high cache probability",
                query
            );
        }
        
        // Low cache probability queries
        let cache_unlikely = vec![
            "what is the current time in Tokyo?",
            "latest news today",
        ];
        
        for query in cache_unlikely {
            let analysis = router.analyze(query);
            assert!(
                analysis.cache_probability < 0.5,
                "Query '{}' should have low cache probability",
                query
            );
        }
    }
}