//! Types and structures for the Enhanced UUID System
//! Defines all data structures used for UUID tracking, memory management, and temporal operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ============================================
// Memory Types
// ============================================

/// Types of memories that can be created
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// User query or question
    Query,
    /// System response to a query
    Response,
    /// Imported or processed document
    Document,
    /// User note or annotation
    Note,
    /// Generated summary of content
    Summary,
    /// Analysis result
    Analysis,
    /// Code snippet or implementation
    Code,
    /// Error or debug information
    Error,
    /// Learning data or pattern
    Learning,
    /// System metadata
    System,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Response => "response",
            Self::Document => "document",
            Self::Note => "note",
            Self::Summary => "summary",
            Self::Analysis => "analysis",
            Self::Code => "code",
            Self::Error => "error",
            Self::Learning => "learning",
            Self::System => "system",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "query" => Some(Self::Query),
            "response" => Some(Self::Response),
            "document" => Some(Self::Document),
            "note" => Some(Self::Note),
            "summary" => Some(Self::Summary),
            "analysis" => Some(Self::Analysis),
            "code" => Some(Self::Code),
            "error" => Some(Self::Error),
            "learning" => Some(Self::Learning),
            "system" => Some(Self::System),
            _ => None,
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================
// Evolution Types
// ============================================

/// Types of memory evolution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvolutionType {
    /// Improved version with more accuracy
    Refinement,
    /// Expanded with additional information
    Expansion,
    /// Corrected errors or inaccuracies
    Correction,
    /// Condensed to essential information
    Summarization,
    /// Translated or transformed format
    Translation,
    /// Merged from multiple sources
    Fusion,
    /// Split into multiple parts
    Fragmentation,
    /// Updated with new information
    Update,
}

impl EvolutionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Refinement => "refinement",
            Self::Expansion => "expansion",
            Self::Correction => "correction",
            Self::Summarization => "summarization",
            Self::Translation => "translation",
            Self::Fusion => "fusion",
            Self::Fragmentation => "fragmentation",
            Self::Update => "update",
        }
    }
}

impl fmt::Display for EvolutionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================
// Relationship Types
// ============================================

/// Types of relationships between memories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    /// Parent-child hierarchical relationship
    ParentChild,
    /// Links memory to its original truth
    TruthSource,
    /// Semantically similar content
    SemanticSimilar,
    /// Sequential in time
    TemporalSequence,
    /// User-specific preference connection
    UserPreference,
    /// Cross-reference between documents
    CrossReference,
    /// Derived from another memory
    DerivedFrom,
    /// Contradicts another memory
    Contradicts,
    /// Supports or validates another memory
    Supports,
    /// Part of a larger whole
    PartOf,
}

impl RelationshipType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ParentChild => "parent_child",
            Self::TruthSource => "truth_source",
            Self::SemanticSimilar => "semantic_similar",
            Self::TemporalSequence => "temporal_sequence",
            Self::UserPreference => "user_preference",
            Self::CrossReference => "cross_reference",
            Self::DerivedFrom => "derived_from",
            Self::Contradicts => "contradicts",
            Self::Supports => "supports",
            Self::PartOf => "part_of",
        }
    }
}

// ============================================
// Core Data Structures
// ============================================

/// Original truth record (immutable source data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalTruth {
    pub uuid: Uuid,
    pub raw_content: String,
    pub content_hash: String,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub source_metadata: HashMap<String, serde_json::Value>,
}

/// Memory record with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub uuid: Uuid,
    pub original_uuid: Uuid,
    pub content: String,
    pub memory_type: MemoryType,
    pub user_id: String,
    pub session_id: String,
    pub parent_uuid: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub confidence_score: f32,
    pub processing_path: String,
    pub processing_time_ms: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Relationship between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationship {
    pub from_uuid: Uuid,
    pub to_uuid: Uuid,
    pub relationship_type: RelationshipType,
    pub strength: f32,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Evolution record tracking memory changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvolution {
    pub from_uuid: Uuid,
    pub to_uuid: Uuid,
    pub evolution_type: EvolutionType,
    pub evolved_at: DateTime<Utc>,
    pub time_gap_hours: f32,
    pub change_summary: Option<String>,
}

/// Processing log entry for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingLog {
    pub uuid: Uuid,
    pub original_uuid: Uuid,
    pub memory_uuid: Option<Uuid>,
    pub stage: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

// ============================================
// Search and Query Results
// ============================================

/// Result from temporal search with time weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSearchResult {
    pub uuid: Uuid,
    pub memory_type: MemoryType,
    pub content: String,
    pub vector_score: f32,
    pub time_weight: f32,
    pub final_score: f32,
    pub age_hours: f32,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// User behavior patterns discovered through analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPatterns {
    pub user_id: String,
    pub peak_hours: Vec<u32>,           // Hours of day with most activity
    pub favorite_types: Vec<MemoryType>, // Most frequently used memory types
    pub common_domains: Vec<String>,     // Frequently accessed domains
    pub query_patterns: Vec<String>,     // Common query structures
    pub avg_session_length: f32,         // Average session duration in minutes
    pub discovered_at: DateTime<Utc>,
    pub confidence: f32,
}

/// Memory chain showing evolution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChain {
    pub root_uuid: Uuid,
    pub current_uuid: Uuid,
    pub evolution_count: usize,
    pub total_time_span_hours: f32,
    pub evolutions: Vec<MemoryEvolution>,
}

/// Statistics for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub uuid: Uuid,
    pub access_count: u32,
    pub relationship_count: usize,
    pub evolution_count: usize,
    pub avg_confidence: f32,
    pub total_processing_time_ms: u64,
    pub last_accessed: DateTime<Utc>,
}

// ============================================
// Configuration
// ============================================

/// Configuration for the UUID system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UUIDConfig {
    /// Enable automatic deduplication
    pub enable_deduplication: bool,
    
    /// Time decay factor for temporal search (hours)
    pub time_decay_hours: f32,
    
    /// Maximum evolution chain length
    pub max_evolution_depth: usize,
    
    /// Minimum confidence threshold for storage
    pub min_confidence_threshold: f32,
    
    /// Enable access tracking
    pub enable_access_tracking: bool,
    
    /// Enable processing audit logs
    pub enable_audit_logs: bool,
    
    /// Maximum relationships per memory
    pub max_relationships: usize,
    
    /// Pattern discovery minimum occurrences
    pub pattern_min_occurrences: usize,
}

impl Default for UUIDConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            time_decay_hours: 24.0,
            max_evolution_depth: 10,
            min_confidence_threshold: 0.3,
            enable_access_tracking: true,
            enable_audit_logs: true,
            max_relationships: 100,
            pattern_min_occurrences: 5,
        }
    }
}

// ============================================
// Errors
// ============================================

/// Errors that can occur in the UUID system
#[derive(Debug, thiserror::Error)]
pub enum UUIDError {
    #[error("Truth not found for UUID: {0}")]
    TruthNotFound(Uuid),
    
    #[error("Memory not found: {0}")]
    MemoryNotFound(Uuid),
    
    #[error("Duplicate content detected with hash: {0}")]
    DuplicateContent(String),
    
    #[error("Evolution chain too deep: {0} > {1}")]
    EvolutionDepthExceeded(usize, usize),
    
    #[error("Invalid relationship: {0}")]
    InvalidRelationship(String),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Vector store error: {0}")]
    VectorStoreError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

// ============================================
// Utility Functions
// ============================================

/// Calculate content hash for deduplication
pub fn calculate_content_hash(content: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Calculate time weight for temporal scoring
pub fn calculate_time_weight(created_at: DateTime<Utc>, decay_hours: f32) -> f32 {
    let now = Utc::now();
    let hours_old = (now - created_at).num_hours() as f32;
    (-hours_old / decay_hours).exp()
}

/// Get time context from hour of day
pub fn get_time_context(hour: u32) -> &'static str {
    match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=20 => "evening",
        _ => "night",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_type_conversion() {
        assert_eq!(MemoryType::from_str("query"), Some(MemoryType::Query));
        assert_eq!(MemoryType::Query.as_str(), "query");
        assert_eq!(MemoryType::from_str("invalid"), None);
    }
    
    #[test]
    fn test_content_hash() {
        let content = "Hello, world!";
        let hash1 = calculate_content_hash(content);
        let hash2 = calculate_content_hash(content);
        assert_eq!(hash1, hash2);
        
        let different = "Different content";
        let hash3 = calculate_content_hash(different);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_time_weight() {
        let now = Utc::now();
        let weight_now = calculate_time_weight(now, 24.0);
        assert!((weight_now - 1.0).abs() < 0.01);
        
        let day_old = now - chrono::Duration::days(1);
        let weight_day = calculate_time_weight(day_old, 24.0);
        assert!(weight_day > 0.3 && weight_day < 0.4);
    }
}