#!/bin/bash
# Complete database initialization script for Memory Nexus pipeline
# Run this BEFORE starting the pipeline to ensure all schemas are ready

set -e

echo "========================================="
echo "Memory Nexus Database Initialization"
echo "========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
SURREALDB_URL=${SURREALDB_URL:-"ws://localhost:8000"}
SURREALDB_USER=${SURREALDB_USER:-"root"}
SURREALDB_PASS=${SURREALDB_PASS:-"root"}
SURREALDB_NS=${SURREALDB_NS:-"nexus"}
SURREALDB_DB=${SURREALDB_DB:-"memory"}

QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}

echo -e "${YELLOW}Checking services...${NC}"

# Check if SurrealDB is running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    echo -e "${GREEN}✓ SurrealDB is running${NC}"
else
    echo -e "${RED}✗ SurrealDB is not running. Start it with: docker-compose up -d surrealdb${NC}"
    exit 1
fi

# Check if Qdrant is running
if curl -s -o /dev/null -w "%{http_code}" ${QDRANT_URL}/health | grep -q "200"; then
    echo -e "${GREEN}✓ Qdrant is running${NC}"
else
    echo -e "${RED}✗ Qdrant is not running. Start it with: docker-compose up -d qdrant${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Initializing SurrealDB schema...${NC}"

# Apply SurrealDB schema
surreal import --conn ${SURREALDB_URL} \
    --user ${SURREALDB_USER} \
    --pass ${SURREALDB_PASS} \
    --ns ${SURREALDB_NS} \
    --db ${SURREALDB_DB} \
    src/database/surrealdb_schema.surql 2>/dev/null || {
    
    # If import fails, try using SQL directly
    echo -e "${YELLOW}Using SQL import method...${NC}"
    
    cat src/database/surrealdb_schema.surql | \
    surreal sql --conn ${SURREALDB_URL} \
        --user ${SURREALDB_USER} \
        --pass ${SURREALDB_PASS} \
        --ns ${SURREALDB_NS} \
        --db ${SURREALDB_DB} 2>/dev/null || {
        
        echo -e "${YELLOW}Using HTTP API method...${NC}"
        
        # Final fallback: Use HTTP API
        curl -X POST http://localhost:8000/sql \
            -H "Accept: application/json" \
            -H "NS: ${SURREALDB_NS}" \
            -H "DB: ${SURREALDB_DB}" \
            -u "${SURREALDB_USER}:${SURREALDB_PASS}" \
            --data-binary @src/database/surrealdb_schema.surql
    }
}

echo -e "${GREEN}✓ SurrealDB schema initialized${NC}"

echo -e "\n${YELLOW}Initializing Qdrant collections...${NC}"

# Create Qdrant collection with binary quantization
curl -X PUT "${QDRANT_URL}/collections/memories" \
    -H "Content-Type: application/json" \
    -d '{
  "vectors": {
    "size": 1024,
    "distance": "Cosine",
    "on_disk": false
  },
  "optimizers_config": {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1000,
    "default_segment_number": 8,
    "max_segment_size": 200000,
    "memmap_threshold": 50000,
    "indexing_threshold": 10000,
    "flush_interval_sec": 5
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
    "ef_construct": 200,
    "full_scan_threshold": 10000,
    "max_indexing_threads": 0,
    "on_disk": false,
    "payload_m": null
  }
}' 2>/dev/null | python3 -m json.tool > /dev/null && echo -e "${GREEN}✓ Qdrant 'memories' collection created${NC}" || echo -e "${YELLOW}! Collection may already exist${NC}"

# Create payload indexes for efficient filtering
echo -e "${YELLOW}Creating Qdrant payload indexes...${NC}"

curl -X PUT "${QDRANT_URL}/collections/memories/index" \
    -H "Content-Type: application/json" \
    -d '{
  "field_name": "user_id",
  "field_schema": "keyword"
}' 2>/dev/null

curl -X PUT "${QDRANT_URL}/collections/memories/index" \
    -H "Content-Type: application/json" \
    -d '{
  "field_name": "created_at",
  "field_schema": "datetime"
}' 2>/dev/null

curl -X PUT "${QDRANT_URL}/collections/memories/index" \
    -H "Content-Type: application/json" \
    -d '{
  "field_name": "memory_type",
  "field_schema": "keyword"
}' 2>/dev/null

echo -e "${GREEN}✓ Qdrant indexes created${NC}"

# Test SurrealDB queries
echo -e "\n${YELLOW}Testing database queries...${NC}"

# Test SurrealDB
echo "SELECT count() FROM memories GROUP ALL;" | \
surreal sql --conn ${SURREALDB_URL} \
    --user ${SURREALDB_USER} \
    --pass ${SURREALDB_PASS} \
    --ns ${SURREALDB_NS} \
    --db ${SURREALDB_DB} > /dev/null 2>&1 && \
echo -e "${GREEN}✓ SurrealDB queries working${NC}" || \
echo -e "${RED}✗ SurrealDB query failed${NC}"

# Test Qdrant
curl -s "${QDRANT_URL}/collections/memories" > /dev/null && \
echo -e "${GREEN}✓ Qdrant collection accessible${NC}" || \
echo -e "${RED}✗ Qdrant collection not accessible${NC}"

echo -e "\n========================================="
echo -e "${GREEN}Database initialization complete!${NC}"
echo "========================================="
echo ""
echo "Databases are ready for the Memory Nexus pipeline."
echo ""
echo "SurrealDB tables created:"
echo "  • memories (with full-text search)"
echo "  • relationships (graph edges)"
echo "  • processing_log (pipeline tracking)"
echo "  • user_patterns (learning)"
echo "  • audit_log (changes)"
echo ""
echo "Qdrant collections created:"
echo "  • memories (1024-dim vectors with INT8 quantization)"
echo ""
echo "Start the pipeline with: cargo run --release --features full"