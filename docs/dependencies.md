PHASE 0: Foundation Dependencies & Build Configuration
Complete Cargo.toml with All Battle-Tested Dependencies
toml[package]
name = "memory-nexus-pipeline"
version = "2.0.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
# ========== CORE ASYNC RUNTIME ==========
tokio = { version = "1.47", features = ["full", "parking_lot", "test-util"] }
async-trait = "0.1"
futures = "0.3"

# ========== WEB FRAMEWORK ==========
axum = { version = "0.8", features = ["ws", "macros"] }
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["cors", "trace", "compression", "limit"] }
hyper = { version = "1.0", features = ["full"] }

# ========== DATABASES ==========
surrealdb = { version = "2.0", features = ["protocol-ws", "rustls"] }  # Note: v3.1 available but may need migration
qdrant-client = { version = "1.15", features = ["download-snapshots"] }
redis = { version = "0.24", features = ["tokio-comp", "connection-manager", "cluster"] }

# ========== SERIALIZATION ==========
serde = { version = "1.0", features = ["derive", "rc"] }
serde_json = "1.0"
bincode = "1.3"
rkyv = { version = "0.7", features = ["validation", "strict"] }

# ========== PARALLEL PROCESSING (Battle-Tested) ==========
rayon = "1.10"
crossbeam = { version = "0.8", features = ["crossbeam-channel"] }
parking_lot = { version = "0.12", features = ["arc_lock"] }
dashmap = { version = "6.0", features = ["rayon", "serde"] }

# ========== SIMD & LOW-LEVEL (Proven Performance) ==========
packed_simd_2 = "0.3"
wide = "0.7"
bytemuck = { version = "1.14", features = ["derive"] }
aligned = "0.4"

# ========== BINARY OPERATIONS (32x Compression) ==========
bitvec = "1.0"
bit-vec = "0.6"
roaring = "0.10"  # Roaring bitmaps for better compression

# ========== MEMORY OPTIMIZATION ==========
memmap2 = "0.9"
bytes = "1.5"
smallvec = { version = "1.13", features = ["union", "const_generics"] }
compact_str = "0.7"
string_cache = "0.8"

# ========== CACHING ==========
moka = { version = "0.12", features = ["future", "sync"] }
cached = { version = "0.49", features = ["async", "disk_store"] }

# ========== METRICS & MONITORING ==========
prometheus = { version = "0.13", features = ["process"] }
opentelemetry = { version = "0.21", features = ["rt-tokio"] }
opentelemetry-prometheus = "0.14"
metrics = "0.22"
metrics-exporter-prometheus = "0.13"

# ========== LOGGING & TRACING ==========
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-opentelemetry = "0.22"
tracing-appender = "0.2"

# ========== ERROR HANDLING ==========
anyhow = "1.0"
thiserror = "1.0"
color-eyre = "0.6"

# ========== UTILITIES ==========
uuid = { version = "1.6", features = ["v4", "fast-rng", "serde"] }
chrono = { version = "0.4", features = ["serde", "clock"] }
once_cell = "1.19"
lazy_static = "1.4"
arc-swap = "1.7"

# ========== HTTP CLIENT ==========
reqwest = { version = "0.12", features = ["json", "rustls-tls", "stream"] }
url = "2.5"

# ========== HASHING (Faster than default) ==========
ahash = "0.8"
rustc-hash = "2.0"
xxhash-rust = { version = "0.8", features = ["xxh3"] }

# ========== ALLOCATORS (13% speedup proven) ==========
mimalloc = { version = "0.1", default-features = false }
jemallocator = { version = "0.5", optional = true }

# ========== COMPRESSION ==========
zstd = "0.13"
lz4 = "1.24"
snap = "1.1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports", "async_tokio"] }
proptest = "1.4"
quickcheck = "1.0"
fake = "2.9"
approx = "0.5"
test-case = "3.1"
serial_test = "3.0"

[build-dependencies]
cc = "1.0"
which = "6.0"

[features]
default = ["simd", "parallel", "binary-opt", "mimalloc-allocator"]
simd = ["packed_simd_2", "wide"]
parallel = ["rayon", "crossbeam", "dashmap"]
binary-opt = ["bitvec", "bincode", "rkyv", "roaring"]
mimalloc-allocator = ["mimalloc"]
jemalloc-allocator = ["jemallocator"]
profile = ["prometheus", "opentelemetry"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
debug = false
overflow-checks = false
incremental = false

[profile.release-with-debug]
inherits = "release"
debug = true
strip = false

[profile.bench]
inherits = "release"
lto = false

# ========== NEW CAPABILITIES (2024-2025) ==========
# Machine Learning & AI Inference

## Rust ML Frameworks (Choose based on use case)
# Candle - HuggingFace's minimalist ML framework, best for LLM inference
candle = { git = "https://github.com/huggingface/candle" }
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"

# Burn - Flexible deep learning with JIT compilation, best for training
burn = "0.15"
burn-ndarray = "0.15"  # CPU backend
burn-cuda = { version = "0.15", optional = true }  # GPU backend

# ORT - ONNX Runtime wrapper, best for production inference
ort = { version = "2.0", features = ["cuda", "tensorrt"] }

# Tract - Pure Rust ONNX/TF inference, best for embedded/WASM
tract = "0.21"

## Vector Databases & Search
# LanceDB - Embedded vector DB with Arrow format
lancedb = "0.13"
arrow = "54.0"
arrow-array = "54.0"
arrow-schema = "54.0"

# USearch - 10-20x faster than FAISS for vector similarity
usearch = "2.16"

# Graph Databases (Alternatives to SurrealDB)
# IndraDB - Pure Rust graph database
indradb = "4.0"
indradb-lib = "4.0"

# Cozo - Graph-vector database with Datalog queries
cozo = "0.7"

## Advanced Vector Operations
# HNSW implementation for vector search
hnsw = "0.11"
instant-distance = "0.6"  # Fast HNSW implementation

## Embedding Models & Processing
# Text embeddings and tokenization
text-embeddings = "0.6"
tokenizers = { version = "0.21", features = ["http"] }
fastembed = "4.2"  # Fast embedding generation

## GPU Acceleration
# CUDA bindings for GPU compute
cust = { version = "0.3", optional = true }  # CUDA runtime
cudarc = { version = "0.10", optional = true }  # Safe CUDA wrapper

## Performance Monitoring & Profiling
# CPU profiling and flame graphs
pprof = { version = "0.14", features = ["flamegraph"] }
tracing-flame = "0.2"
tracy-client = { version = "0.17", optional = true }

## Advanced Async Runtime Features
# Async traits and utilities
async-stream = "0.3"
async-recursion = "1.1"
futures-concurrency = "7.8"  # Concurrent futures operations

## Data Processing Pipelines
# Stream processing and data pipelines
datafusion = "44.0"  # SQL query engine
polars = { version = "0.44", features = ["lazy"] }  # DataFrame library
arrow-flight = "54.0"  # Arrow Flight RPC

## Specialized Compression
# Columnar compression for analytics
parquet = { version = "54.0", features = ["async"] }
arrow-ipc = "54.0"  # Arrow IPC format

## WebAssembly Support
# WASM compilation and runtime
wasm-bindgen = { version = "0.2", optional = true }
wasmer = { version = "5.0", optional = true }
wasmtime = { version = "27.0", optional = true }

## Distributed Computing
# Cluster coordination and distributed systems
etcd-rs = "1.0"  # Distributed key-value store
raft = "0.7"  # Consensus algorithm
tonic = "0.12"  # gRPC framework

## Security & Cryptography
# Modern crypto libraries
ring = "0.17"  # Crypto primitives
rustls = "0.23"  # TLS implementation
argon2 = "0.5"  # Password hashing

## Testing & Benchmarking
# Advanced testing utilities
insta = { version = "1.41", features = ["json", "yaml"] }  # Snapshot testing
divan = "0.1"  # Microbenchmarking framework
loom = { version = "0.7", optional = true }  # Concurrency testing

[features]
# ML/AI features
ml-inference = ["candle-core", "ort", "tract"]
ml-training = ["burn", "burn-cuda"]
vector-search = ["usearch", "hnsw", "instant-distance"]

# Distributed features
distributed = ["etcd-rs", "raft", "tonic"]

# GPU acceleration
gpu = ["burn-cuda", "cust", "cudarc", "ort/cuda"]

# WebAssembly
wasm = ["wasm-bindgen", "tract"]  # Tract supports WASM

# Advanced profiling
profiling = ["pprof", "tracing-flame", "tracy-client"]
