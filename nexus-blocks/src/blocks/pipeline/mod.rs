//! Pipeline Processing Blocks (Wrapped)
//! 
//! Core pipeline stages that wrap existing implementations:
//! - Router: Intelligent query routing (<0.2ms)
//! - Preprocessor: Parallel text processing (<10ms)
//! - Search: Orchestrated parallel search (<25ms)
//! - Fusion: Result merging with scoring (<5ms)

pub mod router_block;
pub mod preprocessor_block;
pub mod search_block;
pub mod fusion_block;

// Re-export wrapped pipeline blocks and their configs
pub use router_block::{
    IntelligentRouterBlock,
    RouterConfig,
};

pub use preprocessor_block::{
    PreprocessorBlock,
    PreprocessorConfig,
};

pub use search_block::{
    SearchOrchestratorBlock,
    SearchConfig,
};

pub use fusion_block::{
    ResilientFusionBlock,
    FusionConfig,
    ScoringMatrixConfig,
};

// Re-export types from the wrapped implementations
pub use memory_nexus::pipeline::intelligent_router::{
    RoutingPath, ComplexityLevel, QueryDomain,
};

pub use memory_nexus::pipeline::preprocessor_enhanced::{
    ChunkingStrategy, ProcessedChunk,
};

pub use memory_nexus::pipeline::search_orchestrator::{
    SearchRequest, SearchResponse,
};

pub use memory_nexus::pipeline::fusion::{
    FusedResult, FusionMetrics,
};