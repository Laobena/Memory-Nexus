#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::pipeline::search_orchestrator::SearchResult;
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_binary_heap_top_k_selection() {
        let fusion = FusionEngine::new();
        
        // Create 100 test results with varying scores
        let mut results = Vec::new();
        for i in 0..100 {
            results.push(ProcessedResult {
                score: (100 - i) as f32 / 100.0,
                content: format!("Result {}", i),
                source: crate::core::types::DataSource::Cache,
                metadata: HashMap::new(),
            });
        }
        
        // Select top 8
        let top_k = fusion.select_top_k_efficient(results, 8);
        
        assert_eq!(top_k.len(), 8);
        
        // Verify they're sorted in descending order
        for i in 1..top_k.len() {
            assert!(top_k[i-1].score >= top_k[i].score);
        }
        
        // Verify we got the highest scores
        assert!(top_k[0].score >= 0.92);
    }
    
    #[tokio::test]
    async fn test_component_scores() {
        let scores = ComponentScores {
            semantic: 0.9,
            keyword: 0.8,
            temporal: 0.7,
            quality: 0.85,
            domain: 0.75,
            user_pref: 0.6,
            cross_validation: 0.3,
        };
        
        // Test serialization
        let json = serde_json::to_string(&scores).unwrap();
        let deserialized: ComponentScores = serde_json::from_str(&json).unwrap();
        
        assert_eq!(scores.semantic, deserialized.semantic);
        assert_eq!(scores.cross_validation, deserialized.cross_validation);
    }
    
    #[tokio::test]
    async fn test_search_result_fusion() {
        let fusion = FusionEngine::new();
        fusion.initialize(&Config::default()).await.unwrap();
        
        // Create test SearchResults
        let search_results = vec![
            SearchResult {
                id: "1".to_string(),
                content: "First result".to_string(),
                score: 0.95,
                source: crate::pipeline::search_orchestrator::SearchSource::Qdrant,
                metadata: HashMap::new(),
                confidence: 0.9,
            },
            SearchResult {
                id: "2".to_string(),
                content: "Second result".to_string(),
                score: 0.85,
                source: crate::pipeline::search_orchestrator::SearchSource::SurrealDB,
                metadata: HashMap::new(),
                confidence: 0.8,
            },
        ];
        
        // Test fusion without embeddings
        let fused = fusion.fuse_search_results(search_results.clone(), None).await.unwrap();
        assert!(!fused.is_empty());
        
        // Test fusion with embeddings
        let dummy_embedding = vec![0.1; 1024];
        let fused_with_emb = fusion.fuse_search_results(search_results, Some(&dummy_embedding)).await.unwrap();
        assert!(!fused_with_emb.is_empty());
        assert!(fused_with_emb.len() <= 8); // Should select top 8
    }
    
    #[test]
    fn test_ordered_result_heap_ordering() {
        use std::collections::BinaryHeap;
        
        let mut heap = BinaryHeap::new();
        
        // Add results with different scores
        heap.push(OrderedResult(ProcessedResult {
            score: 0.5,
            content: "Mid".to_string(),
            source: crate::core::types::DataSource::Cache,
            metadata: HashMap::new(),
        }));
        
        heap.push(OrderedResult(ProcessedResult {
            score: 0.9,
            content: "High".to_string(),
            source: crate::core::types::DataSource::Cache,
            metadata: HashMap::new(),
        }));
        
        heap.push(OrderedResult(ProcessedResult {
            score: 0.1,
            content: "Low".to_string(),
            source: crate::core::types::DataSource::Cache,
            metadata: HashMap::new(),
        }));
        
        // Pop should give highest score first
        let highest = heap.pop().unwrap();
        assert_eq!(highest.0.score, 0.9);
        
        let mid = heap.pop().unwrap();
        assert_eq!(mid.0.score, 0.5);
        
        let lowest = heap.pop().unwrap();
        assert_eq!(lowest.0.score, 0.1);
    }
}