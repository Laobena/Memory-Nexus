//! Pipeline stage blocks - Router, Preprocessor, Search, Fusion

pub mod router;
pub mod preprocessor;
pub mod search;
pub mod fusion;

// Re-export main blocks
pub use router::intelligent_router::IntelligentRouterBlock;
pub use preprocessor::zero_copy_preprocessor::PreprocessorBlock;
pub use search::orchestrator::SearchOrchestratorBlock;
pub use fusion::simd_fusion::ResilientFusionBlock;