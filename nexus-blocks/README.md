# Nexus Blocks

**A modular LEGO-like block system for building and testing AI memory pipelines**

## 🎯 Purpose

Nexus Blocks provides a standardized block interface for the Memory Nexus pipeline, enabling:
- **Pipeline Revamping** - Mix and match blocks to create custom pipelines
- **Block Testing** - Test individual components in isolation
- **Different Pipeline Packages** - Build specialized pipelines for different use cases
- **LEGO-like Composability** - Snap blocks together like building blocks

## 🔧 What This Is

This project **wraps existing implementations** from the main `src/` folder into standardized `PipelineBlock` interfaces. Think of it as:
- **Adapters** that make all components speak the same language
- **Building blocks** that can be combined in different ways
- **Test harness** for experimenting with pipeline configurations
- **Playground** for trying new pipeline architectures

## 📦 Wrapped Components (11 Blocks)

### Pipeline Blocks (4)
```rust
✅ IntelligentRouterBlock    - Wraps src/pipeline/intelligent_router.rs (<0.2ms)
✅ PreprocessorBlock         - Wraps src/pipeline/preprocessor_enhanced.rs (<10ms)
✅ SearchOrchestratorBlock   - Wraps src/pipeline/search_orchestrator.rs (<25ms)
✅ ResilientFusionBlock      - Wraps src/pipeline/fusion.rs (<5ms)
```

### Engine Blocks (4)
```rust
✅ AccuracyEngineBlock       - Wraps src/engines/accuracy.rs (8ms, 99% precision)
✅ IntelligenceEngineBlock   - Wraps src/engines/intelligence.rs (12ms, cross-domain)
✅ LearningEngineBlock       - Wraps src/engines/learning.rs (10ms, adaptive)
✅ MiningEngineBlock         - Wraps src/engines/mining.rs (15ms, patterns)
```

### Storage Blocks (3)
```rust
✅ TieredCache              - Wraps src/core/lock_free_cache.rs (1ms, 3-tier)
⚠️ VectorStore              - Placeholder for vector database
⚠️ WriteAheadLog            - Placeholder for WAL implementation
⚠️ TransactionManager       - Placeholder for transaction support
```

## 🏗️ Architecture

```
nexus-blocks/
├── src/
│   ├── blocks/
│   │   ├── converters.rs      # Type conversions (BlockInput ↔ existing types)
│   │   ├── engines/            # Wrapped engine implementations
│   │   ├── pipeline/           # Wrapped pipeline components
│   │   ├── storage/            # Wrapped storage systems
│   │   └── registration/       # UUID tracking system
│   ├── core/                   # PipelineBlock trait & infrastructure
│   └── supporting_module/      # Helper modules for blocks
```

## 🎮 How It Works

1. **Original Code Untouched** - All main implementations in `src/` remain unchanged
2. **Wrapper Pattern** - Each block wraps an existing implementation:
   ```rust
   pub struct RouterBlock {
       inner: Arc<IntelligentRouter>,  // Original implementation
       config: RouterConfig,            // Block configuration
   }
   ```
3. **Uniform Interface** - All blocks implement `PipelineBlock` trait
4. **Type Converters** - Handle translations between block types and original types

## 🚀 Usage Examples

### Build a Custom Pipeline
```rust
use nexus_blocks::*;

// Create blocks
let router = IntelligentRouterBlock::new(RouterConfig::default());
let preprocessor = PreprocessorBlock::new();
let search = SearchOrchestratorBlock::new();
let fusion = ResilientFusionBlock::new(FusionConfig::default());

// Snap them together
let pipeline = Pipeline::builder()
    .add_block(router)
    .add_block(preprocessor)
    .add_block(search)
    .add_block(fusion)
    .build();

// Process data
let result = pipeline.process(input).await?;
```

### Test Individual Blocks
```rust
// Test router in isolation
let router = IntelligentRouterBlock::new(config);
let mut context = PipelineContext::new();

router.initialize(BlockConfig::default()).await?;
let output = router.process(input, &mut context).await?;
assert!(output.latency_ms < 0.2);
```

### Create Different Pipeline Packages
```rust
// Fast path - cache only
let fast_pipeline = Pipeline::new()
    .with(RouterBlock::new())
    .with(CacheBlock::new());

// Smart path - with preprocessing
let smart_pipeline = Pipeline::new()
    .with(RouterBlock::new())
    .with(PreprocessorBlock::new())
    .with(CacheBlock::new());

// Full intelligence path
let full_pipeline = Pipeline::new()
    .with_all_blocks();
```

## 🔬 Testing & Experimentation

This block system enables:
- **A/B Testing** - Compare different pipeline configurations
- **Performance Testing** - Measure individual block latencies
- **Integration Testing** - Test block combinations
- **Experimental Pipelines** - Try new architectures without breaking production

## 📊 Performance Targets

All wrapped blocks maintain original performance:

| Block | Target | Status |
|-------|--------|--------|
| Router | <0.2ms | ✅ Preserved |
| Cache | <2ms | ✅ Preserved |
| Preprocessor | <10ms | ✅ Preserved |
| Search | <25ms | ✅ Preserved |
| Fusion | <5ms | ✅ Preserved |
| All Engines | <15ms each | ✅ Preserved |

## 🛠️ Development

```bash
# Build the blocks
cargo build --release

# Run tests
cargo test --all-features

# Check specific block
cargo test router_block

# Benchmark performance
cargo bench
```

## 🎯 Why This Matters

1. **Flexibility** - Reconfigure pipelines without touching core code
2. **Testing** - Test blocks individually or in combinations
3. **Evolution** - Easy to add new blocks or replace existing ones
4. **Experimentation** - Try different pipeline architectures
5. **Modularity** - True separation of concerns

## 🚧 Future Possibilities

- **Hot-swapping** - Replace blocks at runtime
- **Remote Blocks** - Blocks running as microservices
- **Custom Blocks** - User-defined blocks for special cases
- **Pipeline Marketplace** - Share and reuse pipeline configurations
- **Visual Pipeline Builder** - Drag-and-drop pipeline construction

---

**Note:** This is a wrapper/adapter layer. The actual implementations live in the main `src/` folder. This project makes them composable and testable as standardized blocks.