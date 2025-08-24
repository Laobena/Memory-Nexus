/// Consolidated hash utilities for the Memory Nexus pipeline
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use ahash::{AHasher, RandomState};

#[cfg(feature = "xxhash")]
use xxhash_rust::xxh3::{xxh3_64 as xxh3_64_impl, xxh3_128 as xxh3_128_impl};

/// Fast string hashing using AHash (best for hash maps)
#[inline(always)]
pub fn ahash_string(s: &str) -> u64 {
    let mut hasher = AHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Standard hash using DefaultHasher (stable across runs)
#[inline(always)]
pub fn default_hash(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// XXHash3 for high-quality, fast hashing
#[inline(always)]
pub fn xxhash3_64(s: &str) -> u64 {
    #[cfg(feature = "xxhash")]
    {
        xxh3_64_impl(s.as_bytes())
    }
    #[cfg(not(feature = "xxhash"))]
    {
        // Fallback to ahash when xxhash is not available
        ahash_string(s)
    }
}

/// XXHash3 128-bit for when you need more bits
#[inline(always)]
pub fn xxhash3_128(s: &str) -> u128 {
    #[cfg(feature = "xxhash")]
    {
        xxh3_128_impl(s.as_bytes())
    }
    #[cfg(not(feature = "xxhash"))]
    {
        // Fallback to two ahash calls for 128-bit
        let h1 = ahash_string(s);
        let h2 = ahash_string(&format!("{}_salt", s));
        ((h1 as u128) << 64) | (h2 as u128)
    }
}

/// Generate cache key with prefix
#[inline(always)]
pub fn generate_cache_key(prefix: &str, content: &str) -> String {
    format!("{}:{:x}", prefix, ahash_string(content))
}

/// Generate cache key for pipeline requests
#[inline(always)]
pub fn generate_pipeline_cache_key(user_id: Option<&str>, content: &str) -> String {
    let user = user_id.unwrap_or("anon");
    format!("pipeline:{}:{:x}", user, ahash_string(content))
}

/// Generate cache key for embeddings
#[inline(always)]
pub fn generate_embedding_cache_key(content: &str) -> String {
    format!("embedding:{:x}", xxhash3_64(content))
}

/// Check if content should be stored based on heuristics
#[inline(always)]
pub fn should_store_content(content: &str, metadata_has_store: bool) -> bool {
    metadata_has_store || content.len() > 1000
}

/// Fast hash for deduplication
#[inline(always)]
pub fn dedup_hash(s: &str) -> u64 {
    ahash_string(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let test_str = "test content";
        
        // AHash should be consistent within same run
        let hash1 = ahash_string(test_str);
        let hash2 = ahash_string(test_str);
        assert_eq!(hash1, hash2);
        
        // Default hash should be consistent
        let dhash1 = default_hash(test_str);
        let dhash2 = default_hash(test_str);
        assert_eq!(dhash1, dhash2);
    }
    
    #[test]
    fn test_cache_key_generation() {
        let key = generate_pipeline_cache_key(Some("user123"), "query content");
        assert!(key.starts_with("pipeline:user123:"));
        
        let key2 = generate_embedding_cache_key("test embedding");
        assert!(key2.starts_with("embedding:"));
    }
    
    #[test]
    fn test_should_store() {
        assert!(!should_store_content("short", false));
        assert!(should_store_content("short", true));
        assert!(should_store_content(&"x".repeat(1001), false));
    }
}