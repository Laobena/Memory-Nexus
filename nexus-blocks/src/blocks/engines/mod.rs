//! Four Specialized Processing Engines (Wrapped)
//! 
//! Each engine wraps an existing implementation from the main project:
//! - Accuracy: High-precision processing with 99% accuracy
//! - Intelligence: Context-aware cross-domain analysis
//! - Learning: Adaptive user preference modeling
//! - Mining: Pattern discovery and anomaly detection

pub mod accuracy_block;
pub mod intelligence_block;
pub mod learning_block;
pub mod mining_block;

// Re-export wrapped engine blocks and their configs
pub use accuracy_block::{
    AccuracyEngineBlock,
    AccuracyConfig,
};

pub use intelligence_block::{
    IntelligenceEngineBlock,
    IntelligenceConfig,
};

pub use learning_block::{
    LearningEngineBlock,
    LearningConfig,
};

pub use mining_block::{
    MiningEngineBlock,
    MiningConfig,
};