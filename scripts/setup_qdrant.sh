#!/bin/bash
# Qdrant collection setup for Memory Nexus Pipeline
# Matches what the code expects in search_orchestrator.rs

QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}

echo "Setting up Qdrant collection for Memory Nexus..."

# Delete existing collection if it exists (for clean setup)
curl -X DELETE "${QDRANT_URL}/collections/memories" 2>/dev/null

# Create collection with proper configuration
curl -X PUT "${QDRANT_URL}/collections/memories" \
  -H "Content-Type: application/json" \
  -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "optimizers_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000,
    "default_segment_number": 4,
    "max_segment_size": 100000,
    "memmap_threshold": 20000,
    "indexing_threshold": 10000,
    "flush_interval_sec": 5,
    "max_optimization_threads": 1
  },
  "wal_config": {
    "wal_capacity_mb": 32,
    "wal_segments_ahead": 0
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99,
      "always_ram": true
    }
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 10000,
    "max_indexing_threads": 0,
    "on_disk": false,
    "payload_m": null
  }
}'

echo "Creating payload indexes for filtering..."

# Create payload index for content (text search)
curl -X PUT "${QDRANT_URL}/collections/memories/index" \
  -H "Content-Type: application/json" \
  -d '{
  "field_name": "content",
  "field_schema": "text"
}'

# Create payload index for metadata (object)
curl -X PUT "${QDRANT_URL}/collections/memories/index" \
  -H "Content-Type: application/json" \
  -d '{
  "field_name": "metadata",
  "field_schema": "keyword"
}'

# Create payload index for created_at (datetime filtering)
curl -X PUT "${QDRANT_URL}/collections/memories/index" \
  -H "Content-Type: application/json" \
  -d '{
  "field_name": "created_at",
  "field_schema": "datetime"
}'

# Create payload index for user_id (filtering by user)
curl -X PUT "${QDRANT_URL}/collections/memories/index" \
  -H "Content-Type: application/json" \
  -d '{
  "field_name": "user_id",
  "field_schema": "keyword"
}'

# Create payload index for memory_type
curl -X PUT "${QDRANT_URL}/collections/memories/index" \
  -H "Content-Type: application/json" \
  -d '{
  "field_name": "memory_type",
  "field_schema": "keyword"
}'

echo "Qdrant collection 'memories' configured successfully!"
echo ""
echo "Configuration:"
echo "  • Vector size: 1024 (for mxbai-embed-large)"
echo "  • Distance: Cosine similarity"
echo "  • Quantization: INT8 (32x compression)"
echo "  • HNSW: m=16, ef_construct=100"
echo "  • Indexed fields: content, metadata, created_at, user_id, memory_type"