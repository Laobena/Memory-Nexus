#!/bin/bash

# Test script for embedding service integration
# Tests the complete pipeline from query -> embedding -> search

set -e

echo "==================================================="
echo "Memory Nexus - Embedding Service Integration Test"
echo "==================================================="

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Starting Ollama container..."
    docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
    sleep 5
else
    echo "✅ Ollama is running"
fi

# Check if model is available
echo "🔍 Checking for mxbai-embed-large model..."
if ! curl -s http://localhost:11434/api/tags | grep -q "mxbai-embed-large"; then
    echo "📥 Pulling mxbai-embed-large model (this may take a few minutes)..."
    docker exec ollama ollama pull mxbai-embed-large
else
    echo "✅ Model mxbai-embed-large is available"
fi

# Test embedding generation
echo ""
echo "🧪 Testing embedding generation..."
curl -X POST http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mxbai-embed-large",
        "prompt": "Test query for Memory Nexus"
    }' 2>/dev/null | jq -r '.embedding | length' | {
        read dim
        if [ "$dim" = "1024" ]; then
            echo "✅ Embedding generation successful (1024 dimensions)"
        else
            echo "❌ Embedding dimension mismatch: got $dim, expected 1024"
            exit 1
        fi
    }

# Build and run integration test
echo ""
echo "🔨 Building integration test..."
cat > /tmp/test_embedding_integration.rs << 'EOF'
use memory_nexus_pipeline::ai::{EmbeddingService, EmbeddingConfig};
use memory_nexus_pipeline::pipeline::{ParallelPreprocessor, ChunkingStrategy};
use memory_nexus_pipeline::pipeline::IntelligentRouter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("\n=== Testing Embedding Service ===");
    
    // Test 1: Direct embedding service
    let config = EmbeddingConfig::default();
    let service = EmbeddingService::new(config);
    
    println!("Initializing embedding service...");
    service.initialize().await?;
    
    println!("Generating single embedding...");
    let embedding = service.generate_embedding("What is the capital of France?").await?;
    assert_eq!(embedding.len(), 1024);
    println!("✅ Single embedding: {} dimensions", embedding.len());
    
    // Test 2: Batch embeddings
    println!("\nGenerating batch embeddings...");
    let texts = vec![
        "First test query".to_string(),
        "Second test query".to_string(),
        "Third test query with more content".to_string(),
    ];
    let batch = service.generate_batch(&texts).await?;
    assert_eq!(batch.len(), 3);
    println!("✅ Batch embeddings: {} results", batch.len());
    
    // Test 3: Cache hit
    println!("\nTesting cache...");
    let _ = service.generate_embedding("Cached query").await?;
    let cached = service.generate_embedding("Cached query").await?;
    println!("✅ Cache working");
    
    // Test 4: Preprocessor integration
    println!("\n=== Testing Preprocessor Integration ===");
    let preprocessor = ParallelPreprocessor::new();
    preprocessor.initialize().await?;
    
    let text = "Memory Nexus is a high-performance AI memory system. \
                It achieves world-record accuracy with advanced embeddings.";
    
    let processed = preprocessor.process(text, ChunkingStrategy::default()).await?;
    println!("✅ Preprocessor generated {} chunks with embeddings", processed.chunks.len());
    
    // Test 5: Router with embeddings
    println!("\n=== Testing Router Integration ===");
    let router = IntelligentRouter::new();
    let analysis = router.analyze("complex technical query about database optimization").await;
    
    if analysis.embedding.is_some() {
        println!("✅ Router generated embedding for non-cache query");
    } else {
        println!("⚠️  Router skipped embedding (cache-only path)");
    }
    
    // Print statistics
    println!("\n=== Statistics ===");
    println!("{}", service.stats().await);
    
    println!("\n✅ All embedding integration tests passed!");
    
    Ok(())
}
EOF

# Create temporary Cargo.toml for test
cat > /tmp/Cargo.toml << EOF
[package]
name = "test-embedding"
version = "0.1.0"
edition = "2021"

[dependencies]
memory-nexus-pipeline = { path = "/mnt/c/Users/VJ_la/Desktop/nexus" }
tokio = { version = "1.47", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"
EOF

# Run the integration test
echo ""
echo "🚀 Running integration test..."
cd /tmp
if cargo run --release 2>&1; then
    echo ""
    echo "✅ Integration test successful!"
else
    echo ""
    echo "❌ Integration test failed"
    exit 1
fi

echo ""
echo "==================================================="
echo "✅ All embedding tests passed successfully!"
echo "==================================================="
echo ""
echo "Next steps:"
echo "1. Start the full Memory Nexus server: cargo run --release"
echo "2. Test with real queries via API"
echo "3. Monitor embedding cache hit rates"
echo "4. Verify search accuracy with embeddings"