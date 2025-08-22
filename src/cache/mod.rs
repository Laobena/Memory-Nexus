//! Cache Module for Memory Nexus
//!
//! Streamlined caching system using consolidated lock-free cache

pub mod factory;

// Use consolidated cache from core module
pub use crate::core::lock_free_cache::{
    LockFreeCache as SafeWTinyLFUCache,
    LockFreeCache as CacheInterface,
    CacheConfig,
    CacheStats as CacheMetrics
};
pub use factory::CacheFactory;
