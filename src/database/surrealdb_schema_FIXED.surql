-- FIXED SurrealDB Schema for Memory Nexus Pipeline
-- This schema matches what the pipeline code actually expects

-- ========================================
-- MEMORIES TABLE (Main storage)
-- ========================================
DEFINE TABLE memories SCHEMAFULL;

-- Fields the CODE actually uses (from Memory struct)
DEFINE FIELD uuid ON memories TYPE string ASSERT $value != NONE;
DEFINE FIELD original_uuid ON memories TYPE string ASSERT $value != NONE;  -- ADDED: Code needs this
DEFINE FIELD parent_uuid ON memories TYPE option<string>;
DEFINE FIELD content ON memories TYPE string ASSERT $value != NONE;
DEFINE FIELD memory_type ON memories TYPE string;  -- Removed strict enum for flexibility
DEFINE FIELD user_id ON memories TYPE string ASSERT $value != NONE;  -- ADDED: Code needs this
DEFINE FIELD session_id ON memories TYPE string;  -- ADDED: Code needs this
DEFINE FIELD created_at ON memories TYPE datetime DEFAULT time::now();
DEFINE FIELD last_accessed ON memories TYPE datetime DEFAULT time::now();  -- ADDED: Code needs this
DEFINE FIELD access_count ON memories TYPE int DEFAULT 0;  -- ADDED: Code needs this
DEFINE FIELD confidence_score ON memories TYPE float DEFAULT 1.0;  -- RENAMED from confidence
DEFINE FIELD processing_path ON memories TYPE string DEFAULT '';  -- ADDED: Code needs this
DEFINE FIELD processing_time_ms ON memories TYPE int DEFAULT 0;
DEFINE FIELD metadata ON memories TYPE object DEFAULT {};
DEFINE FIELD content_hash ON memories TYPE string;  -- For deduplication

-- Indexes for performance
DEFINE INDEX idx_memories_uuid ON memories FIELDS uuid UNIQUE;
DEFINE INDEX idx_memories_original ON memories FIELDS original_uuid;
DEFINE INDEX idx_memories_parent ON memories FIELDS parent_uuid;
DEFINE INDEX idx_memories_user ON memories FIELDS user_id;
DEFINE INDEX idx_memories_type ON memories FIELDS memory_type;
DEFINE INDEX idx_memories_created ON memories FIELDS created_at;
DEFINE INDEX idx_memories_hash ON memories FIELDS content_hash;

-- Full-text search index (CRITICAL for search_orchestrator.rs)
DEFINE INDEX idx_memories_content ON memories FIELDS content SEARCH ANALYZER ascii BM25;

-- ========================================
-- PROCESSING LOG (Pipeline tracking)
-- ========================================
DEFINE TABLE processing_log SCHEMAFULL;
DEFINE FIELD uuid ON processing_log TYPE string ASSERT $value != NONE;
DEFINE FIELD stage ON processing_log TYPE string ASSERT $value != NONE;
DEFINE FIELD status ON processing_log TYPE string ASSERT $value IN ['started', 'completed', 'failed', 'skipped'];
DEFINE FIELD timestamp ON processing_log TYPE datetime DEFAULT time::now();
DEFINE FIELD duration_ms ON processing_log TYPE int DEFAULT 0;
DEFINE FIELD metadata ON processing_log TYPE object DEFAULT {};

DEFINE INDEX idx_processing_uuid ON processing_log FIELDS uuid;
DEFINE INDEX idx_processing_stage ON processing_log FIELDS stage;
DEFINE INDEX idx_processing_timestamp ON processing_log FIELDS timestamp;

-- ========================================
-- RELATIONSHIPS (Graph connections)
-- ========================================
-- Using SurrealDB's native RELATE for graph relationships
-- Example: RELATE memory:uuid1->evolves_to->memory:uuid2

-- ========================================
-- EVENT TRIGGERS
-- ========================================

-- Auto-generate content hash for deduplication
DEFINE EVENT calculate_hash ON memories WHEN $event = "CREATE" THEN (
    UPDATE $this SET content_hash = crypto::sha256($this.content)
);

-- Update last_accessed on read
DEFINE EVENT update_access ON memories WHEN $event = "UPDATE" THEN (
    UPDATE $this SET 
        last_accessed = time::now(),
        access_count = $this.access_count + 1
);

-- Log processing stage duration
DEFINE EVENT log_duration ON processing_log 
WHEN $event = "CREATE" AND $after.status = "completed" THEN {
    UPDATE $this SET duration_ms = 0
};

-- ========================================
-- FUNCTIONS (For graph traversal)
-- ========================================

-- Functions will be added later when SurrealDB supports them fully

-- ========================================
-- PERMISSIONS (Production ready)
-- ========================================

-- Create scope for application access
DEFINE SCOPE nexus_app SESSION 24h
    SIGNIN (
        SELECT * FROM user WHERE email = $email AND crypto::argon2::compare(password, $password)
    )
    SIGNUP (
        CREATE user SET email = $email, password = crypto::argon2::generate($password)
    );

-- Define permissions
DEFINE TABLE memories PERMISSIONS
    FOR select WHERE user_id = $auth.id OR $auth.role = 'admin',
    FOR create WHERE $auth.id != NONE,
    FOR update NONE,  -- Immutable
    FOR delete WHERE $auth.role = 'admin';

DEFINE TABLE processing_log PERMISSIONS
    FOR select WHERE true,
    FOR create WHERE $auth.id != NONE,
    FOR update NONE,
    FOR delete WHERE $auth.role = 'admin';

-- ========================================
-- INITIAL DATA / TEST
-- ========================================

-- Create a test memory to verify schema
CREATE memories:test SET
    uuid = 'test-uuid-1',
    original_uuid = 'test-uuid-1',
    parent_uuid = NONE,
    content = 'Test memory for schema validation',
    memory_type = 'Query',
    user_id = 'system',
    session_id = 'test-session',
    created_at = time::now(),
    last_accessed = time::now(),
    access_count = 0,
    confidence_score = 1.0,
    processing_path = 'test.init',
    processing_time_ms = 0,
    metadata = {},
    content_hash = crypto::sha256('Test memory for schema validation');

-- Verify full-text search works
SELECT * FROM memories WHERE content @@ 'test' LIMIT 1;

INFO FOR DB;