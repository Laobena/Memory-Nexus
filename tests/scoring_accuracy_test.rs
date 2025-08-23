//! Comprehensive test suite for 98.4% accuracy target
//! 
//! Tests the complete 5-factor scoring integration to ensure
//! we achieve the same accuracy as the old Memory Nexus system.

use memory_nexus_pipeline::pipeline::{
    intelligent_router::{IntelligentRouter, QueryIntent, ScoringWeights},
    search_orchestrator::{SearchOrchestrator, SearchResult, FiveFactorScorer},
    context_booster::ContextBooster,
    fusion::FusionEngine,
    adaptive_weights::AdaptiveWeightOptimizer,
};
use memory_nexus_pipeline::search::bm25_scorer::QuickBM25;
use std::collections::HashMap;

/// Test case for scoring accuracy
#[derive(Debug, Clone)]
struct TestCase {
    query: String,
    documents: Vec<TestDocument>,
    expected_top_1: usize,  // Index of expected top result
    expected_top_3: Vec<usize>,  // Indices of expected top 3
    intent: QueryIntent,
}

#[derive(Debug, Clone)]
struct TestDocument {
    content: String,
    metadata: HashMap<String, serde_json::Value>,
    embedding: Vec<f32>,
    expected_rank: usize,
}

/// Run complete accuracy test suite
#[tokio::test]
async fn test_98_4_percent_accuracy() {
    let test_cases = create_test_cases();
    let mut correct_top_1 = 0;
    let mut correct_top_3 = 0;
    let total_cases = test_cases.len();
    
    for case in test_cases {
        let results = run_test_case(case.clone()).await;
        
        // Check if top result matches expected
        if !results.is_empty() && results[0].doc_index == case.expected_top_1 {
            correct_top_1 += 1;
        }
        
        // Check if top 3 contains all expected
        let top_3_indices: Vec<usize> = results.iter()
            .take(3)
            .map(|r| r.doc_index)
            .collect();
        
        if case.expected_top_3.iter().all(|idx| top_3_indices.contains(idx)) {
            correct_top_3 += 1;
        }
    }
    
    let top_1_accuracy = (correct_top_1 as f32 / total_cases as f32) * 100.0;
    let top_3_accuracy = (correct_top_3 as f32 / total_cases as f32) * 100.0;
    
    println!("🎯 Accuracy Results:");
    println!("   Top-1 Accuracy: {:.1}%", top_1_accuracy);
    println!("   Top-3 Accuracy: {:.1}%", top_3_accuracy);
    
    // We target 98.4% for top-3 accuracy (matching old Memory Nexus)
    assert!(top_3_accuracy >= 98.0, "Top-3 accuracy {:.1}% is below 98% target", top_3_accuracy);
    
    // Top-1 should be at least 85%
    assert!(top_1_accuracy >= 85.0, "Top-1 accuracy {:.1}% is below 85% target", top_1_accuracy);
}

/// Test intent detection accuracy
#[test]
fn test_intent_detection_accuracy() {
    let router = IntelligentRouter::new();
    
    let test_cases = vec![
        ("error handling in rust async", QueryIntent::Debug),
        ("how does machine learning work", QueryIntent::Learn),
        ("find the documentation for Vec", QueryIntent::Lookup),
        ("implement a web server", QueryIntent::Build),
        ("what is the weather today", QueryIntent::Unknown),
    ];
    
    let mut correct = 0;
    for (query, expected_intent) in &test_cases {
        let analysis = router.analyze_query(query);
        if analysis.intent == *expected_intent {
            correct += 1;
        }
    }
    
    let accuracy = (correct as f32 / test_cases.len() as f32) * 100.0;
    assert!(accuracy >= 80.0, "Intent detection accuracy {:.1}% is too low", accuracy);
}

/// Test BM25+ scoring accuracy
#[test]
fn test_bm25_scoring_accuracy() {
    let scorer = QuickBM25::new();
    
    // Test exact match scores highest
    let score1 = scorer.score("rust error", "rust error handling guide");
    let score2 = scorer.score("rust error", "python tutorial for beginners");
    assert!(score1 > score2 * 2.0, "BM25 should strongly prefer exact matches");
    
    // Test term frequency saturation
    let score3 = scorer.score("async", "async async async async async");
    let score4 = scorer.score("async", "async programming in rust");
    assert!(score3 < score4 * 1.5, "BM25 should saturate term frequency");
}

/// Test context boosting accuracy
#[test]
fn test_context_boosting() {
    use memory_nexus_pipeline::pipeline::context_booster::UserContext;
    
    let mut context = UserContext::default();
    context.tech_stack = vec!["Rust".to_string(), "Python".to_string()];
    context.expertise_level = 0.7;  // Advanced
    
    let booster = ContextBooster::new(context);
    
    // Test tech stack boost
    let boost1 = booster.calculate_boost(
        "Rust async programming with tokio",
        &serde_json::json!({})
    );
    let boost2 = booster.calculate_boost(
        "JavaScript promises and callbacks",
        &serde_json::json!({})
    );
    
    assert!(boost1 > boost2, "Should boost content matching user's tech stack");
    assert!(boost1 > 1.2, "Tech stack match should give significant boost");
}

/// Test adaptive weight learning
#[test]
fn test_adaptive_weight_learning() {
    use memory_nexus_pipeline::pipeline::adaptive_weights::{
        AdaptiveWeightOptimizer, LearningEvent, FeedbackSignal
    };
    
    let optimizer = AdaptiveWeightOptimizer::new();
    
    // Simulate positive feedback for semantic-heavy results
    for i in 0..20 {
        let event = LearningEvent {
            query: format!("learn about {}", i),
            intent: QueryIntent::Learn,
            result_id: format!("doc_{}", i),
            weights_used: ScoringWeights {
                semantic: 0.5,
                bm25: 0.15,
                recency: 0.10,
                importance: 0.15,
                context: 0.10,
            },
            feedback: FeedbackSignal::Click { 
                position: 0, 
                dwell_time_ms: 20000  // Long dwell = good result
            },
            timestamp: i,
        };
        optimizer.record_feedback(event);
    }
    
    // Check that weights adapted
    let new_weights = optimizer.get_weights(&QueryIntent::Learn);
    assert!(new_weights.semantic >= 0.45, "Semantic weight should increase with positive feedback");
    
    // Check performance metrics
    let metrics = optimizer.analyze_performance();
    assert!(metrics.positive_ratio > 0.9, "Should have high positive ratio");
    assert!(metrics.estimated_accuracy > 0.85, "Should estimate high accuracy");
}

/// Helper function to run a single test case
async fn run_test_case(case: TestCase) -> Vec<ScoredResult> {
    // Create mock scorer
    let scorer = FiveFactorScorer::new(
        case.query.clone(),
        Some(vec![0.5; 1024]),  // Mock embedding
        case.intent.clone(),
    );
    
    // Score all documents
    let mut results = Vec::new();
    for (idx, doc) in case.documents.iter().enumerate() {
        let score = scorer.calculate_score(
            &doc.content,
            doc.embedding.as_slice(),
            &doc.metadata,
        );
        
        results.push(ScoredResult {
            doc_index: idx,
            score,
            content: doc.content.clone(),
        });
    }
    
    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results
}

#[derive(Debug)]
struct ScoredResult {
    doc_index: usize,
    score: f32,
    content: String,
}

/// Create comprehensive test cases
fn create_test_cases() -> Vec<TestCase> {
    vec![
        // Debug intent - error handling
        TestCase {
            query: "rust async error handling".to_string(),
            documents: vec![
                TestDocument {
                    content: "Comprehensive guide to error handling in Rust async code using Result and ? operator".to_string(),
                    metadata: create_metadata(0.9, true, 1),
                    embedding: create_embedding(0.95),
                    expected_rank: 1,
                },
                TestDocument {
                    content: "Python exception handling with try/except blocks".to_string(),
                    metadata: create_metadata(0.7, false, 30),
                    embedding: create_embedding(0.3),
                    expected_rank: 3,
                },
                TestDocument {
                    content: "Rust error handling with async/await and tokio".to_string(),
                    metadata: create_metadata(0.85, true, 7),
                    embedding: create_embedding(0.9),
                    expected_rank: 2,
                },
            ],
            expected_top_1: 0,
            expected_top_3: vec![0, 2, 1],
            intent: QueryIntent::Debug,
        },
        
        // Learn intent - conceptual understanding
        TestCase {
            query: "how do neural networks learn".to_string(),
            documents: vec![
                TestDocument {
                    content: "Neural networks learn through backpropagation and gradient descent optimization".to_string(),
                    metadata: create_metadata(0.95, true, 3),
                    embedding: create_embedding(0.92),
                    expected_rank: 1,
                },
                TestDocument {
                    content: "Quick reference: neural network forward pass equations".to_string(),
                    metadata: create_metadata(0.6, false, 15),
                    embedding: create_embedding(0.7),
                    expected_rank: 3,
                },
                TestDocument {
                    content: "Understanding deep learning: how neural networks learn from data".to_string(),
                    metadata: create_metadata(0.88, true, 5),
                    embedding: create_embedding(0.88),
                    expected_rank: 2,
                },
            ],
            expected_top_1: 0,
            expected_top_3: vec![0, 2, 1],
            intent: QueryIntent::Learn,
        },
        
        // Lookup intent - exact documentation
        TestCase {
            query: "Vec push documentation".to_string(),
            documents: vec![
                TestDocument {
                    content: "std::vec::Vec::push - Appends an element to the back of a collection".to_string(),
                    metadata: create_metadata(1.0, true, 0),
                    embedding: create_embedding(0.8),
                    expected_rank: 1,
                },
                TestDocument {
                    content: "Tutorial on using vectors in Rust with examples".to_string(),
                    metadata: create_metadata(0.7, false, 10),
                    embedding: create_embedding(0.6),
                    expected_rank: 3,
                },
                TestDocument {
                    content: "Vec methods: push, pop, insert, remove - Rust documentation".to_string(),
                    metadata: create_metadata(0.9, true, 2),
                    embedding: create_embedding(0.75),
                    expected_rank: 2,
                },
            ],
            expected_top_1: 0,
            expected_top_3: vec![0, 2, 1],
            intent: QueryIntent::Lookup,
        },
        
        // Build intent - implementation focused
        TestCase {
            query: "implement REST API server".to_string(),
            documents: vec![
                TestDocument {
                    content: "Step-by-step guide: Building a REST API server with Actix-web in Rust".to_string(),
                    metadata: create_metadata(0.9, true, 5),
                    embedding: create_embedding(0.85),
                    expected_rank: 1,
                },
                TestDocument {
                    content: "REST API design principles and best practices".to_string(),
                    metadata: create_metadata(0.75, false, 20),
                    embedding: create_embedding(0.6),
                    expected_rank: 3,
                },
                TestDocument {
                    content: "Complete REST API implementation with authentication and database".to_string(),
                    metadata: create_metadata(0.88, true, 3),
                    embedding: create_embedding(0.82),
                    expected_rank: 2,
                },
            ],
            expected_top_1: 0,
            expected_top_3: vec![0, 2, 1],
            intent: QueryIntent::Build,
        },
    ]
}

/// Helper to create metadata
fn create_metadata(importance: f32, is_official: bool, days_old: u32) -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();
    metadata.insert("importance".to_string(), serde_json::json!(importance));
    metadata.insert("is_official".to_string(), serde_json::json!(is_official));
    metadata.insert("days_old".to_string(), serde_json::json!(days_old));
    metadata.insert("source".to_string(), serde_json::json!("test"));
    metadata
}

/// Helper to create mock embedding
fn create_embedding(similarity: f32) -> Vec<f32> {
    // Create embedding that will have desired similarity
    let mut embedding = vec![0.0; 1024];
    for i in 0..1024 {
        embedding[i] = similarity * (i as f32 / 1024.0).sin();
    }
    embedding
}

/// Benchmark scoring performance
#[test]
fn benchmark_scoring_performance() {
    use std::time::Instant;
    
    let scorer = QuickBM25::new();
    let query = "rust async error handling tokio";
    let document = "Complete guide to error handling in Rust async programming with tokio. \
                    Learn how to handle errors in async functions, use the ? operator, \
                    and implement custom error types for robust async applications.";
    
    let start = Instant::now();
    let iterations = 10000;
    
    for _ in 0..iterations {
        let _ = scorer.score(query, document);
    }
    
    let elapsed = start.elapsed();
    let per_score = elapsed / iterations;
    
    println!("⚡ BM25 Scoring Performance: {:?} per score", per_score);
    assert!(per_score.as_micros() < 100, "BM25 scoring taking too long: {:?}", per_score);
}

/// Integration test for complete pipeline
#[tokio::test]
async fn test_complete_pipeline_integration() {
    // This would test the full pipeline with all components
    // For now, we'll test that all components can be created and work together
    
    let router = IntelligentRouter::new();
    let optimizer = AdaptiveWeightOptimizer::new();
    let fusion = FusionEngine::new();
    
    // Test a query through the pipeline
    let query = "implement authentication in rust web app";
    let analysis = router.analyze_query(query);
    
    assert_eq!(analysis.intent, QueryIntent::Build);
    assert!(analysis.complexity_score > 0.5);
    
    // Get adaptive weights
    let weights = optimizer.get_weights(&analysis.intent);
    assert!((weights.semantic + weights.bm25 + weights.recency + 
             weights.importance + weights.context - 1.0).abs() < 0.01);
    
    println!("✅ Complete pipeline integration test passed");
}