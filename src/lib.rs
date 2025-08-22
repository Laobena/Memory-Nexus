#![allow(dead_code)] // Remove when implementing features
#![cfg_attr(has_simd, feature(portable_simd))]

// =========================================================================================
// Global Allocator Configuration - Critical for Performance (2-4x speedup)
// =========================================================================================
// Research shows dramatic performance differences between allocators:
// - jemalloc: 4ns per small allocation (consistently fastest)
// - mimalloc: Critical for musl targets (7x speedup observed)
// - System allocator: 8-9ns (2x slower for small allocations)
//
// Based on production patterns from Discord (millions of connections) and Cloudflare

// Use mimalloc for musl targets (containerized/Alpine Linux environments)
#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Use jemalloc for general deployment (fastest for small allocations)
#[cfg(all(not(target_env = "musl"), not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Note: Windows MSVC uses the system allocator as jemalloc/mimalloc have limited Windows support

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