#![allow(dead_code)] // Remove when implementing features
#![cfg_attr(has_simd, feature(portable_simd))]

// Global allocator configuration
#[cfg(all(not(target_env = "msvc"), feature = "mimalloc-allocator"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(not(target_env = "msvc"), feature = "jemalloc-allocator"))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ===== PUBLIC API =====
pub mod api;
pub mod core;
pub mod engines;
pub mod monitoring;
pub mod optimizations;
pub mod pipeline;

// ===== INTERNAL MODULES =====
pub(crate) mod ai;
pub(crate) mod cache;
pub(crate) mod config;
pub(crate) mod database;
pub(crate) mod errors;
pub(crate) mod health;
pub(crate) mod math;
pub(crate) mod server;
pub(crate) mod types;
pub(crate) mod vectors;

// Re-exports for convenience
pub use crate::core::{Config, NexusError, Result};
pub use crate::pipeline::Pipeline;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

// Feature detection at compile time
#[cfg(has_avx2)]
pub const HAS_AVX2: bool = true;
#[cfg(not(has_avx2))]
pub const HAS_AVX2: bool = false;

#[cfg(has_simd)]
pub const HAS_SIMD: bool = true;
#[cfg(not(has_simd))]
pub const HAS_SIMD: bool = false;