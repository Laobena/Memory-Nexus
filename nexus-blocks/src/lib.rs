//! Memory Nexus Blocks Library
//! 
//! Complete implementation of the Memory Nexus Pipeline Block System.
//! All blocks wrap existing implementations from the main project.

// Core infrastructure
pub mod core;

// Pipeline blocks (wrapped implementations)
pub mod blocks;

// Supporting modules (helpers, not actual blocks)
pub mod supporting_module;

// Pipeline Package System with Resilience (Phase 4)
pub mod packages;

// Re-export wrapped blocks
pub use blocks::{
    // Registration
    UUIDBlock, UUIDConfig,
    
    // Wrapped Engines
    AccuracyEngineBlock, AccuracyConfig,
    IntelligenceEngineBlock, IntelligenceConfig,
    LearningEngineBlock, LearningConfig,
    MiningEngineBlock, MiningConfig,
    
    // Wrapped Pipeline
    IntelligentRouterBlock, RouterConfig,
    PreprocessorBlock, PreprocessorConfig,
    SearchOrchestratorBlock, SearchConfig,
    ResilientFusionBlock, FusionConfig,
    
    // Wrapped Storage
    TieredCache, TieredCacheConfig,
    VectorStore, WriteAheadLog, TransactionManager,
    
    // System
    MemoryNexusSystem,
};

// Re-export core traits and types
pub use core::{
    traits::{
        PipelineBlock, PipelineContext,
        BlockInput, BlockOutput, BlockConfig, BlockMetadata,
        BlockCategory, DeploymentMode, HealthStatus,
    },
    errors::{BlockError, BlockResult},
};

// Re-export converter utilities
pub use blocks::converters::{
    extract_query,
    routing_to_output,
    search_results_to_output,
    fusion_results_to_output,
    chunks_to_output,
    update_context_from_analysis,
};

// Re-export Pipeline Package System
pub use packages::{
    // Main API
    Pipeline, PipelineOutput, PipelineError, HealthStatus,
    
    // Pre-built pipelines
    prebuilt::{
        CacheOnlyPipeline,
        SmartRoutingPipeline,
        FullPipeline,
        MaximumIntelligencePipeline,
        AdaptivePipeline,
    },
    
    // Factory and builder
    factory::{PipelineFactory, PipelineBuilder},
    
    // Dynamic composition
    composer::{DynamicComposer, CompositionStrategy},
    
    // Execution management
    executor::{ExecutionManager, ExecutionMode},
    
    // Orchestration
    orchestrator::{Orchestrator, Workflow},
    
    // Resilience
    resilience::{ResilientPipeline, ErrorHandler, RestartPolicy},
    
    // Health monitoring
    health_monitor::{HealthMonitor, StageHealth},
    
    // Degraded strategies
    degraded_strategies::{DegradedStrategy, DegradedModeStrategies},
    
    // Isolation
    isolation::{StageIsolation, IsolationBoundary},
};