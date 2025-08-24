-- Enhanced UUID System Schema for Memory Nexus
-- This schema implements the complete UUID tracking system with immutability and temporal intelligence

-- ========================================
-- 1. CORE TABLES
-- ========================================

-- Memories table (immutable)
DEFINE TABLE memories SCHEMAFULL;
DEFINE FIELD uuid ON memories TYPE string ASSERT $value != NONE;
DEFINE FIELD content ON memories TYPE string ASSERT $value != NONE;
DEFINE FIELD memory_type ON memories TYPE string ASSERT $value IN ['Query', 'Response', 'Context', 'Error', 'Embedding', 'SearchResult', 'FusionResult', 'Truth', 'Evolution'];
DEFINE FIELD created_at ON memories TYPE datetime DEFAULT time::now();
DEFINE FIELD confidence ON memories TYPE float DEFAULT 1.0;
DEFINE FIELD processing_time_ms ON memories TYPE int DEFAULT 0;
DEFINE FIELD metadata ON memories TYPE object DEFAULT {};
DEFINE FIELD parent_uuid ON memories TYPE option<string>;
DEFINE FIELD is_immutable ON memories TYPE bool DEFAULT true;
DEFINE FIELD hash ON memories TYPE string;

-- Create indexes
DEFINE INDEX idx_memories_uuid ON memories FIELDS uuid UNIQUE;
DEFINE INDEX idx_memories_type ON memories FIELDS memory_type;
DEFINE INDEX idx_memories_created ON memories FIELDS created_at;
DEFINE INDEX idx_memories_parent ON memories FIELDS parent_uuid;

-- Immutability enforcement
DEFINE EVENT ensure_immutability ON memories WHEN $event = "UPDATE" THEN (
    THROW "Memories are immutable. Create an evolution instead."
);

-- Relationships table (graph edges)
DEFINE TABLE relationships SCHEMAFULL;
DEFINE FIELD from_uuid ON relationships TYPE string ASSERT $value != NONE;
DEFINE FIELD to_uuid ON relationships TYPE string ASSERT $value != NONE;
DEFINE FIELD relationship_type ON relationships TYPE string ASSERT $value IN ['GENERATES', 'REFERENCES', 'EVOLVES_FROM', 'VALIDATES', 'ENRICHES', 'CONTRADICTS', 'SUPPORTS'];
DEFINE FIELD strength ON relationships TYPE float DEFAULT 1.0 ASSERT $value >= 0 AND $value <= 1;
DEFINE FIELD created_at ON relationships TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON relationships TYPE object DEFAULT {};

-- Create indexes for relationships
DEFINE INDEX idx_relationships_from ON relationships FIELDS from_uuid;
DEFINE INDEX idx_relationships_to ON relationships FIELDS to_uuid;
DEFINE INDEX idx_relationships_type ON relationships FIELDS relationship_type;

-- Processing log table
DEFINE TABLE processing_log SCHEMAFULL;
DEFINE FIELD uuid ON processing_log TYPE string ASSERT $value != NONE;
DEFINE FIELD stage ON processing_log TYPE string ASSERT $value != NONE;
DEFINE FIELD status ON processing_log TYPE string ASSERT $value IN ['started', 'completed', 'failed', 'skipped'];
DEFINE FIELD timestamp ON processing_log TYPE datetime DEFAULT time::now();
DEFINE FIELD duration_ms ON processing_log TYPE int DEFAULT 0;
DEFINE FIELD metadata ON processing_log TYPE object DEFAULT {};

-- Create indexes for processing log
DEFINE INDEX idx_processing_uuid ON processing_log FIELDS uuid;
DEFINE INDEX idx_processing_stage ON processing_log FIELDS stage;
DEFINE INDEX idx_processing_timestamp ON processing_log FIELDS timestamp;

-- User patterns table (for learning)
DEFINE TABLE user_patterns SCHEMAFULL;
DEFINE FIELD user_id ON user_patterns TYPE string ASSERT $value != NONE;
DEFINE FIELD pattern_type ON user_patterns TYPE string;
DEFINE FIELD frequency ON user_patterns TYPE int DEFAULT 1;
DEFINE FIELD last_seen ON user_patterns TYPE datetime DEFAULT time::now();
DEFINE FIELD context ON user_patterns TYPE object DEFAULT {};
DEFINE FIELD related_uuids ON user_patterns TYPE array DEFAULT [];

-- Create indexes for user patterns
DEFINE INDEX idx_patterns_user ON user_patterns FIELDS user_id;
DEFINE INDEX idx_patterns_type ON user_patterns FIELDS pattern_type;

-- Audit log table (tracks all changes)
DEFINE TABLE audit_log SCHEMAFULL;
DEFINE FIELD table_name ON audit_log TYPE string ASSERT $value != NONE;
DEFINE FIELD record_id ON audit_log TYPE string ASSERT $value != NONE;
DEFINE FIELD action ON audit_log TYPE string ASSERT $value IN ['CREATE', 'UPDATE', 'DELETE', 'ACCESS'];
DEFINE FIELD timestamp ON audit_log TYPE datetime DEFAULT time::now();
DEFINE FIELD user_id ON audit_log TYPE string;
DEFINE FIELD changes ON audit_log TYPE object DEFAULT {};
DEFINE FIELD before_value ON audit_log TYPE object;
DEFINE FIELD after_value ON audit_log TYPE object;

-- Create indexes for audit log
DEFINE INDEX idx_audit_table ON audit_log FIELDS table_name;
DEFINE INDEX idx_audit_record ON audit_log FIELDS record_id;
DEFINE INDEX idx_audit_timestamp ON audit_log FIELDS timestamp;

-- ========================================
-- 2. EVENT TRIGGERS
-- ========================================

-- Auto-generate UUID if not provided
DEFINE EVENT auto_uuid ON memories WHEN $event = "CREATE" AND $before.uuid = NONE THEN (
    UPDATE $this SET uuid = rand::uuid()
);

-- Calculate content hash for deduplication
DEFINE EVENT calculate_hash ON memories WHEN $event = "CREATE" THEN (
    UPDATE $this SET hash = crypto::md5($this.content)
);

-- Track query-response relationships automatically
DEFINE EVENT track_query_response ON memories WHEN $event = "CREATE" AND $after.memory_type = "Response" AND $after.parent_uuid != NONE THEN (
    CREATE relationships SET
        from_uuid = $after.parent_uuid,
        to_uuid = $after.uuid,
        relationship_type = "GENERATES",
        strength = $after.confidence,
        created_at = time::now()
);

-- Log all memory creations
DEFINE EVENT log_memory_creation ON memories WHEN $event = "CREATE" THEN (
    CREATE audit_log SET
        table_name = "memories",
        record_id = $after.uuid,
        action = "CREATE",
        timestamp = time::now(),
        after_value = $after
);

-- Track processing stages
DEFINE EVENT log_processing ON processing_log WHEN $event = "CREATE" THEN (
    LET $start_time = (SELECT timestamp FROM processing_log WHERE uuid = $after.uuid AND stage = $after.stage AND status = "started" LIMIT 1);
    IF $after.status = "completed" AND $start_time THEN (
        UPDATE $this SET duration_ms = time::unix($after.timestamp) - time::unix($start_time[0].timestamp)
    )
);

-- Update user patterns on query creation
DEFINE EVENT update_user_patterns ON memories WHEN $event = "CREATE" AND $after.memory_type = "Query" THEN (
    LET $user_id = $after.metadata.user_id;
    IF $user_id THEN (
        LET $existing = (SELECT * FROM user_patterns WHERE user_id = $user_id AND pattern_type = "query_pattern" LIMIT 1);
        IF $existing THEN (
            UPDATE user_patterns SET 
                frequency = frequency + 1,
                last_seen = time::now(),
                related_uuids = array::append(related_uuids, $after.uuid)
            WHERE user_id = $user_id AND pattern_type = "query_pattern"
        ) ELSE (
            CREATE user_patterns SET
                user_id = $user_id,
                pattern_type = "query_pattern",
                frequency = 1,
                last_seen = time::now(),
                related_uuids = [$after.uuid]
        )
    )
);

-- ========================================
-- 3. FUNCTIONS
-- ========================================

-- Function to get memory chain (full history)
DEFINE FUNCTION fn::get_memory_chain($uuid: string) {
    LET $memory = (SELECT * FROM memories WHERE uuid = $uuid);
    LET $ancestors = [];
    LET $current = $memory[0];
    
    WHILE $current.parent_uuid != NONE {
        LET $parent = (SELECT * FROM memories WHERE uuid = $current.parent_uuid);
        IF $parent {
            LET $ancestors = array::append($ancestors, $parent[0]);
            LET $current = $parent[0];
        } ELSE {
            BREAK;
        }
    };
    
    RETURN {
        memory: $memory[0],
        ancestors: $ancestors,
        depth: array::len($ancestors)
    };
};

-- Function to find related memories
DEFINE FUNCTION fn::find_related($uuid: string, $max_depth: int = 3) {
    LET $visited = [$uuid];
    LET $related = [];
    LET $depth = 0;
    
    WHILE $depth < $max_depth {
        LET $current_level = [];
        FOR $id IN $visited {
            LET $connections = (
                SELECT to_uuid as uuid, relationship_type, strength 
                FROM relationships 
                WHERE from_uuid = $id 
                AND to_uuid NOT IN $visited
            );
            LET $current_level = array::concat($current_level, $connections);
        };
        
        IF array::len($current_level) = 0 {
            BREAK;
        };
        
        LET $related = array::concat($related, $current_level);
        LET $visited = array::concat($visited, $current_level.*.uuid);
        LET $depth = $depth + 1;
    };
    
    RETURN $related;
};

-- Function to calculate temporal decay
DEFINE FUNCTION fn::temporal_decay($created_at: datetime, $half_life_hours: float = 24.0) {
    LET $age_hours = (time::unix(time::now()) - time::unix($created_at)) / 3600.0;
    LET $decay = math::pow(0.5, $age_hours / $half_life_hours);
    RETURN $decay;
};

-- ========================================
-- 4. PERMISSIONS (if needed)
-- ========================================

-- Define namespace and database
-- USE NS nexus DB memory;

-- Define scope for API access
-- DEFINE SCOPE api SESSION 24h
--     SIGNUP (CREATE user SET email = $email, pass = crypto::argon2::generate($pass))
--     SIGNIN (SELECT * FROM user WHERE email = $email AND crypto::argon2::compare(pass, $pass));

-- ========================================
-- 5. INITIAL DATA
-- ========================================

-- Create system memory for initialization
CREATE memories SET
    uuid = "system-init-" + rand::uuid(),
    content = "Memory Nexus UUID System Initialized",
    memory_type = "Context",
    confidence = 1.0,
    metadata = {
        system: true,
        version: "1.0.0",
        initialized_at: time::now()
    },
    is_immutable = true;

-- Log initialization
CREATE audit_log SET
    table_name = "system",
    record_id = "initialization",
    action = "CREATE",
    timestamp = time::now(),
    after_value = {
        message: "Database schema initialized successfully"
    };