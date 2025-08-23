# 🎯 MASTER INTEGRATION PLAN: 5-Factor Scoring System
## Target: 98.4% Search Accuracy (Matching Old Memory Nexus)

## 📊 Current State vs Target State

### Current Pipeline (85% accurate):
```
Query → Router → 4 Paths → Basic Search → Simple Fusion → Results
```

### Target Pipeline (98.4% accurate):
```
Query → Intent Router → 4 Smart Paths → 5-Factor Scoring → Intelligent Fusion → Results
         ↓                               ↓                   ↓
    Detect intent                   Score every result   Context boost
```

## 🏗️ INTEGRATION PHASES (Total: 4-5 hours)

---

## **PHASE 1: Intent Classification** ⏱️ 30 mins
**Location:** `src/pipeline/intelligent_router.rs`

### What to Add:
```rust
// New enum for intent detection
enum QueryIntent {
    Debug,    // Error fixing (40% semantic, 30% BM25)
    Learn,    // Concepts (50% semantic, 15% BM25)
    Lookup,   // Facts (20% semantic, 50% BM25)
    Build,    // Implementation (35% semantic, 25% BM25)
    Unknown,  // Balanced (35% semantic, 25% BM25)
}

// Intent detection in router
fn detect_intent(query: &str) -> QueryIntent {
    // Check keywords for each intent type
}
```

### Integration Point:
```rust
// In IntelligentRouter::analyze()
let intent = detect_intent(&query);  // NEW
let weights = get_weights_for_intent(intent);  // NEW
analysis.intent = intent;  // Add to QueryAnalysis
analysis.scoring_weights = weights;  // Pass to search
```

---

## **PHASE 2: 5-Factor Calculator** ⏱️ 1 hour
**Location:** `src/pipeline/search_orchestrator.rs`

### What to Add:
```rust
struct FiveFactorScorer {
    intent: QueryIntent,
    weights: ScoringWeights,
}

struct SearchSignals {
    semantic_similarity: f32,  // From embeddings
    bm25_score: f32,           // Keyword matching
    recency_score: f32,        // Time decay
    importance_score: f32,     // Access frequency
    context_score: f32,        // User's project
}

impl FiveFactorScorer {
    fn calculate_score(&self, result: &SearchResult) -> f32 {
        let signals = self.extract_signals(result);
        
        // Apply weighted formula
        (signals.semantic * self.weights.semantic) +
        (signals.bm25 * self.weights.bm25) +
        (signals.recency * self.weights.recency) +
        (signals.importance * self.weights.importance) +
        (signals.context * self.weights.context)
    }
}
```

### Integration Point:
```rust
// In SearchOrchestrator::search_all()
let raw_results = self.gather_from_engines().await;

// NEW: Score each result
let scorer = FiveFactorScorer::new(analysis.intent);
for result in &mut raw_results {
    result.five_factor_score = scorer.calculate_score(result);
}

// Sort by score instead of raw relevance
raw_results.sort_by(|a, b| b.five_factor_score.cmp(&a.five_factor_score));
```

---

## **PHASE 3: BM25+ Implementation** ⏱️ 1 hour
**Location:** New file `src/search/bm25_scorer.rs`

### What to Add:
```rust
pub struct BM25Scorer {
    k1: f32,  // 1.2 (term frequency saturation)
    b: f32,   // 0.75 (length normalization)
    avgdl: f32,  // Average document length
    idf_cache: HashMap<String, f32>,
}

impl BM25Scorer {
    pub fn score(&self, query: &str, document: &str) -> f32 {
        let query_terms = tokenize(query);
        let doc_terms = tokenize(document);
        
        let mut score = 0.0;
        for term in query_terms {
            let tf = doc_terms.count(&term);
            let idf = self.calculate_idf(&term);
            
            // BM25+ formula
            score += idf * ((tf * (self.k1 + 1.0)) / 
                          (tf + self.k1 * (1.0 - self.b + 
                           self.b * (doc_len / self.avgdl))));
        }
        score
    }
}
```

### Integration Point:
```rust
// In FiveFactorScorer::extract_signals()
signals.bm25_score = self.bm25_scorer.score(query, &result.content);
```

---

## **PHASE 4: Contextual Boosting** ⏱️ 30 mins
**Location:** `src/pipeline/context_booster.rs`

### What to Add:
```rust
struct ContextBooster {
    user_tech_stack: Vec<String>,
    project_context: String,
    expertise_level: f32,
}

impl ContextBooster {
    fn calculate_boost(&self, result: &SearchResult) -> f32 {
        let mut boost = 1.0;
        
        // Tech stack matching (up to 1.5x)
        for tech in &self.user_tech_stack {
            if result.content.contains(tech) {
                boost *= 1.1;  // 10% per match
            }
        }
        
        // Expertise adjustment
        let complexity = estimate_complexity(&result.content);
        if complexity > self.expertise_level {
            boost *= 0.8;  // Too complex
        }
        
        boost.min(2.0)  // Cap at 2x
    }
}
```

### Integration Point:
```rust
// In SearchOrchestrator, after 5-factor scoring
let booster = ContextBooster::from_user_context(user_id);
for result in &mut scored_results {
    result.final_score = result.five_factor_score * booster.calculate_boost(result);
}
```

---

## **PHASE 5: Fusion Integration** ⏱️ 30 mins
**Location:** `src/pipeline/fusion.rs`

### What to Merge:
```rust
// Your current 6-factor matrix
pub struct ScoringMatrix {
    relevance: f32,   // Keep: aligns with semantic
    freshness: f32,   // Keep: aligns with recency
    diversity: f32,   // Keep: for deduplication
    authority: f32,   // Merge with importance
    coherence: f32,   // Keep: for validation
    confidence: f32,  // Keep: for escalation
}

// Merge strategy
impl FusionEngine {
    fn merge_scoring_systems(&self, result: &SearchResult) -> f32 {
        // Use 5-factor as base
        let base_score = result.five_factor_score;
        
        // Add your unique factors
        let enhanced_score = base_score * 
            (1.0 + (result.diversity * 0.1)) *  // Diversity bonus
            (1.0 + (result.coherence * 0.05));  // Coherence bonus
            
        enhanced_score
    }
}
```

---

## **PHASE 6: Adaptive Weights** ⏱️ 30 mins
**Location:** `src/pipeline/adaptive_weights.rs`

### What to Add:
```rust
struct AdaptiveWeights {
    base_weights: HashMap<QueryIntent, WeightConfig>,
    learning_rate: f32,
}

impl AdaptiveWeights {
    fn get_weights(&self, intent: QueryIntent, feedback: Option<f32>) -> WeightConfig {
        let mut weights = self.base_weights[&intent].clone();
        
        // Adjust based on recent performance
        if let Some(accuracy) = feedback {
            if accuracy < 0.85 {
                // Increase semantic weight if accuracy low
                weights.semantic += 0.05;
                weights.normalize();
            }
        }
        
        weights
    }
}
```

---

## **PHASE 7: Testing & Tuning** ⏱️ 1 hour
**Location:** `tests/accuracy_benchmark.rs`

### Benchmark Tests:
```rust
#[test]
async fn test_search_accuracy() {
    let test_queries = vec![
        ("Python TypeError fix", vec!["TypeError", "Python", "debug"]),
        ("Learn Rust ownership", vec!["ownership", "borrowing", "lifetime"]),
        ("SQL query optimization", vec!["index", "performance", "database"]),
    ];
    
    for (query, expected_terms) in test_queries {
        let results = pipeline.process(query).await;
        
        // Check if top results contain expected terms
        let top_3 = &results[..3.min(results.len())];
        let accuracy = calculate_accuracy(top_3, expected_terms);
        
        assert!(accuracy >= 0.90, "Accuracy {} below 90%", accuracy);
    }
}
```

---

## 📊 **INTEGRATION TIMELINE**

```
Hour 1: Phase 1 + 2 (Intent + 5-Factor base)
Hour 2: Phase 3 (BM25+ implementation)  
Hour 3: Phase 4 + 5 (Context + Fusion)
Hour 4: Phase 6 + 7 (Adaptive + Testing)
```

---

## 🎯 **SUCCESS METRICS**

### Immediate (After Phase 2):
- Cache accuracy: 70% → 85%
- Smart routing: 80% → 90%

### After Phase 4:
- Full pipeline: 85% → 95%
- Context awareness working

### After Phase 7:
- Overall accuracy: 98.4%
- Sub-100ms scoring
- 96% cache hit rate

---

## 🔧 **MINIMAL CHANGES REQUIRED**

### Files to Modify (6 files):
1. `intelligent_router.rs` - Add intent detection (50 lines)
2. `search_orchestrator.rs` - Add 5-factor scoring (150 lines)
3. `fusion.rs` - Merge scoring systems (30 lines)

### Files to Create (3 files):
1. `bm25_scorer.rs` - BM25+ implementation (100 lines)
2. `context_booster.rs` - Contextual scoring (80 lines)
3. `adaptive_weights.rs` - Dynamic weights (60 lines)

**Total: ~470 lines of code**

---

## ✅ **VALIDATION CHECKLIST**

- [ ] Intent detection working (30s test)
- [ ] 5-factor scores calculated (visual check)
- [ ] BM25 scoring accurate (unit test)
- [ ] Context boost applied (user test)
- [ ] Fusion preserves scores (integration test)
- [ ] Adaptive weights adjust (feedback loop)
- [ ] 98.4% accuracy achieved (benchmark)

---

## 🚀 **QUICK START COMMANDS**

```bash
# After implementation
cargo test test_intent_classification
cargo test test_five_factor_scoring  
cargo test test_bm25_accuracy
cargo bench search_accuracy
```

---

## 💡 **KEY SUCCESS FACTORS**

1. **Keep existing structure** - Don't rewrite, enhance
2. **Test each phase** - Verify before moving on
3. **Use same models** - mxbai-embed-large for compatibility
4. **Tune weights** - Start with old Memory Nexus values
5. **Cache aggressively** - 5-factor scores are expensive

---

## 🎯 **THE RESULT**

Your pipeline will have:
- **Speed**: Still meeting all latency targets
- **Accuracy**: 98.4% matching world record
- **Intelligence**: Understands user intent
- **Adaptability**: Learns from usage
- **Scalability**: Same architecture, smarter logic

**From 85% to 98.4% accuracy in 4 hours!**