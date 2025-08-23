//! Comprehensive tests for fusion module

#[cfg(test)]
mod fusion_tests {
    use crate::blocks::fusion::*;
    use crate::core::BlockResult;
    use uuid::Uuid;
    use std::time::{Duration, Instant};
    
    fn create_test_items(count: usize, engine: EngineType) -> Vec<FusionItem> {
        (0..count)
            .map(|i| FusionItem {
                id: Uuid::new_v4(),
                content: format!("Content {}", i).into_bytes(),
                relevance: 0.5 + (i as f32 / count as f32) * 0.5,
                freshness: 0.8 - (i as f32 / count as f32) * 0.3,
                diversity: 0.6 + (i % 3) as f32 * 0.1,
                authority: 0.7,
                coherence: 0.65,
                confidence: 0.8 + (i % 2) as f32 * 0.15,
                source_engine: engine,
                timestamp: chrono::Utc::now() - chrono::Duration::minutes(i as i64),
            })
            .collect()
    }
    
    #[tokio::test]
    async fn test_fusion_with_all_engines() {
        let config = FusionConfig {
            target_results: 8,
            min_quality: 0.6,
            enable_simd: true,
            dedup_threshold: 0.7,
            max_latency_ms: 5,
            auto_escalate: true,
        };
        
        let fusion = ResilientFusionBlock::new(config);
        
        // Create results from all engines
        let mut results = PartialResults::new();
        
        for engine in [
            EngineType::Accuracy,
            EngineType::Intelligence,
            EngineType::Learning,
            EngineType::Mining,
        ] {
            let items = create_test_items(50, engine);
            results.add_batch(items);
        }
        
        let start = Instant::now();
        let output = fusion.fuse_with_degradation(results).await.unwrap();
        let latency = start.elapsed();
        
        // Verify performance
        assert!(latency.as_millis() < 10, "Fusion took {:?}, expected <10ms", latency);
        assert_eq!(output.items.len(), 8);
        assert!(!output.degraded);
        assert_eq!(output.quality, 1.0);
        assert!(output.confidence > 0.7);
    }
    
    #[tokio::test]
    async fn test_fusion_with_partial_results() {
        let config = FusionConfig::default();
        let fusion = ResilientFusionBlock::new(config);
        
        // Only 2 engines available
        let mut results = PartialResults::new();
        results.add_batch(create_test_items(30, EngineType::Accuracy));
        results.add_batch(create_test_items(30, EngineType::Intelligence));
        results.mark_missing(EngineType::Learning);
        results.mark_missing(EngineType::Mining);
        
        let output = fusion.fuse_with_degradation(results).await.unwrap();
        
        assert!(output.degraded);
        assert!(output.quality < 1.0);
        assert!(output.quality > 0.4);  // Should still have decent quality
        assert!(!output.items.is_empty());
    }
    
    #[tokio::test]
    async fn test_minhash_deduplication() {
        let dedup = MinHashDeduplicator::new(0.7);
        
        let mut results = PartialResults::new();
        
        // Add duplicate content
        let duplicate_content = b"This is duplicate content for testing deduplication".to_vec();
        
        for i in 0..10 {
            results.add(FusionItem {
                id: Uuid::new_v4(),
                content: if i < 6 {
                    duplicate_content.clone()
                } else {
                    format!("Unique content {}", i).into_bytes()
                },
                relevance: 0.8,
                freshness: 0.7,
                diversity: 0.6,
                authority: 0.5,
                coherence: 0.4,
                confidence: 0.9,
                source_engine: EngineType::Accuracy,
                timestamp: chrono::Utc::now(),
            });
        }
        
        let deduplicated = dedup.deduplicate(&results).await.unwrap();
        
        // Should remove most duplicates
        assert!(deduplicated.len() < 10);
        assert!(deduplicated.len() >= 5);  // At least unique items + 1 duplicate
        
        let stats = dedup.stats();
        assert!(stats.deduplication_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_top_k_selection_strategies() {
        let items = (0..100)
            .map(|i| ScoredItem {
                item: create_test_items(1, EngineType::Accuracy)[0].clone(),
                score: (i as f32) / 100.0,
                components: ComponentScores::default(),
            })
            .collect();
        
        // Test heap-based selection
        let selector_heap = TopKSelector::new(10)
            .with_strategy(SelectionStrategy::HeapBased);
        let selected_heap = selector_heap.select(items.clone()).unwrap();
        assert_eq!(selected_heap.len(), 10);
        assert!(selected_heap[0].score >= selected_heap[9].score);
        
        // Test quick-select
        let selector_quick = TopKSelector::new(10)
            .with_strategy(SelectionStrategy::QuickSelect);
        let selected_quick = selector_quick.select(items.clone()).unwrap();
        assert_eq!(selected_quick.len(), 10);
        
        // Test partial sort
        let selector_partial = TopKSelector::new(10)
            .with_strategy(SelectionStrategy::PartialSort);
        let selected_partial = selector_partial.select(items).unwrap();
        assert_eq!(selected_partial.len(), 10);
    }
    
    #[tokio::test]
    async fn test_quality_tracking() {
        let tracker = QualityTracker::new();
        
        // Simulate multiple fusion operations
        for i in 0..20 {
            let quality = 0.7 + (i as f32 % 5.0) * 0.05;
            let latency = Duration::from_millis(3 + (i % 4));
            let was_partial = i % 3 == 0;
            let was_escalated = i % 7 == 0;
            
            tracker.record_fusion(
                quality,
                latency,
                was_partial,
                was_escalated,
                true,  // SIMD used
                100 + i * 10,
                80 + i * 8,
            );
        }
        
        let metrics = tracker.get_metrics();
        
        assert!(metrics.fusion_quality > 0.6);
        assert!(metrics.dedup_effectiveness > 0.1);
        assert!(metrics.simd_success_rate == 1.0);
        assert!(metrics.latency_p50.as_millis() <= 6);
        assert!(metrics.latency_p99.as_millis() <= 10);
        assert!(metrics.partial_result_rate < 0.5);
        assert!(metrics.escalation_rate < 0.3);
        
        // Check if quality is acceptable
        assert!(!tracker.is_degraded());
    }
    
    #[tokio::test]
    async fn test_fallback_scorer() {
        let items = create_test_items(50, EngineType::Accuracy);
        let weights = ScoringMatrix::default();
        
        let scorer = FallbackScorer::new().with_debug(true);
        let scored = scorer.score(&items, &weights);
        
        assert_eq!(scored.len(), items.len());
        
        // Verify scoring order
        for i in 1..scored.len() {
            assert!(
                scored[i - 1].score >= scored[i].score,
                "Items not properly ordered: {} < {}",
                scored[i - 1].score,
                scored[i].score
            );
        }
        
        // Verify component scores
        for item in &scored {
            assert!(item.components.semantic > 0.0);
            assert!(item.components.temporal > 0.0);
            assert!(item.components.quality > 0.0);
        }
    }
    
    #[tokio::test]
    async fn test_partial_handler_decisions() {
        let handler = PartialResultHandler::new();
        
        // Test with no results
        let empty = PartialResults::new();
        match handler.handle(&empty).unwrap() {
            PartialHandlingDecision::Reject { reason } => {
                assert!(reason.contains("No results"));
            }
            _ => panic!("Should reject empty results"),
        }
        
        // Test with minimum engines
        let mut partial = PartialResults::new();
        partial.add_batch(create_test_items(20, EngineType::Accuracy));
        partial.add_batch(create_test_items(20, EngineType::Intelligence));
        
        match handler.handle(&partial).unwrap() {
            PartialHandlingDecision::Accept { quality_factor } => {
                assert!(quality_factor > 0.5);
                assert!(quality_factor < 1.0);
            }
            _ => panic!("Should accept partial results with 2 engines"),
        }
        
        // Test with all engines
        let mut complete = partial.clone();
        complete.add_batch(create_test_items(20, EngineType::Learning));
        complete.add_batch(create_test_items(20, EngineType::Mining));
        
        match handler.handle(&complete).unwrap() {
            PartialHandlingDecision::Accept { quality_factor } => {
                assert_eq!(quality_factor, 1.0);
            }
            _ => panic!("Should accept complete results with quality 1.0"),
        }
    }
    
    #[cfg(feature = "fusion")]
    #[tokio::test]
    async fn test_simd_fusion_performance() {
        use std::time::Instant;
        
        let config = FusionConfig {
            enable_simd: true,
            ..Default::default()
        };
        
        let fusion = ResilientFusionBlock::new(config);
        
        // Create large dataset for performance testing
        let mut results = PartialResults::new();
        for engine in [EngineType::Accuracy, EngineType::Intelligence] {
            results.add_batch(create_test_items(100, engine));
        }
        
        // Measure SIMD performance
        let start = Instant::now();
        let output_simd = fusion.fuse_with_degradation(results.clone()).await.unwrap();
        let simd_time = start.elapsed();
        
        // Compare with scalar fallback
        let config_scalar = FusionConfig {
            enable_simd: false,
            ..Default::default()
        };
        let fusion_scalar = ResilientFusionBlock::new(config_scalar);
        
        let start = Instant::now();
        let output_scalar = fusion_scalar.fuse_with_degradation(results).await.unwrap();
        let scalar_time = start.elapsed();
        
        println!("SIMD time: {:?}, Scalar time: {:?}", simd_time, scalar_time);
        println!("Speedup: {:.2}x", scalar_time.as_micros() as f64 / simd_time.as_micros() as f64);
        
        // SIMD should be faster for large datasets
        if cfg!(all(feature = "simdeez", target_arch = "x86_64")) {
            assert!(simd_time < scalar_time);
        }
        
        // Results should be similar
        assert_eq!(output_simd.items.len(), output_scalar.items.len());
    }
    
    #[tokio::test]
    async fn test_scoring_matrix_adjustment() {
        let mut matrix = ScoringMatrix::default();
        
        // Test normalization
        matrix.relevance = 0.5;
        matrix.freshness = 0.3;
        matrix.diversity = 0.2;
        matrix.authority = 0.2;
        matrix.coherence = 0.1;
        matrix.confidence = 0.1;
        
        matrix.normalize();
        
        let sum = matrix.relevance + matrix.freshness + matrix.diversity +
                 matrix.authority + matrix.coherence + matrix.confidence;
        
        assert!((sum - 1.0).abs() < 0.001, "Weights should sum to 1.0");
        
        // Test partial adjustment
        matrix.adjust_for_partial(0.5);  // 50% data available
        
        // Relevance and confidence should be boosted
        assert!(matrix.relevance > 0.35);
        assert!(matrix.confidence > 0.1);
    }
}