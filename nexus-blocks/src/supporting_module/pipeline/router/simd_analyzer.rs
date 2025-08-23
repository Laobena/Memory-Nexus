//! SIMD-accelerated pattern matching for routing decisions

use std::arch::x86_64::*;
use memchr::{memchr, memchr2, memchr3};

/// SIMD pattern analyzer for fast text analysis
pub struct SimdAnalyzer {
    /// Patterns to match
    patterns: Vec<Pattern>,
}

/// Pattern definition for matching
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: u32,
    pub keywords: Vec<String>,
    pub priority: u8,
}

/// Pattern matcher using SIMD
pub struct PatternMatcher {
    /// Compiled patterns for SIMD matching
    compiled: Vec<CompiledPattern>,
}

#[derive(Debug)]
struct CompiledPattern {
    id: u32,
    bytes: Vec<u8>,
    mask: Vec<u8>,
}

impl SimdAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: Self::default_patterns(),
        }
    }
    
    /// Analyze text and return pattern ID
    pub fn analyze_pattern(&self, text: &str) -> Option<u32> {
        let bytes = text.as_bytes();
        
        // Fast path: check for common patterns using memchr
        if self.has_cache_pattern(bytes) {
            return Some(0); // Cache route
        }
        
        if self.has_search_pattern(bytes) {
            return Some(1); // Search route
        }
        
        // Use SIMD for complex pattern matching
        if is_x86_feature_detected!("avx2") {
            unsafe { self.analyze_simd_avx2(bytes) }
        } else if is_x86_feature_detected!("sse4.2") {
            unsafe { self.analyze_simd_sse42(bytes) }
        } else {
            self.analyze_scalar(bytes)
        }
    }
    
    /// Check for cache-related patterns
    fn has_cache_pattern(&self, bytes: &[u8]) -> bool {
        // Look for "get", "fetch", "retrieve"
        memchr(b'g', bytes).is_some() && 
            (bytes.windows(3).any(|w| w == b"get") ||
             bytes.windows(5).any(|w| w == b"fetch"))
    }
    
    /// Check for search-related patterns
    fn has_search_pattern(&self, bytes: &[u8]) -> bool {
        // Look for "search", "find", "query"
        memchr(b's', bytes).is_some() && 
            bytes.windows(6).any(|w| w == b"search") ||
        memchr(b'q', bytes).is_some() && 
            bytes.windows(5).any(|w| w == b"query")
    }
    
    /// SIMD analysis using AVX2
    #[target_feature(enable = "avx2")]
    unsafe fn analyze_simd_avx2(&self, bytes: &[u8]) -> Option<u32> {
        if bytes.len() < 32 {
            return self.analyze_scalar(bytes);
        }
        
        // Create patterns for common routing keywords
        let cache_pattern = _mm256_set1_epi8(b'c' as i8);
        let search_pattern = _mm256_set1_epi8(b's' as i8);
        let storage_pattern = _mm256_set1_epi8(b't' as i8);
        
        let mut cache_score = 0u32;
        let mut search_score = 0u32;
        let mut storage_score = 0u32;
        
        // Process 32 bytes at a time
        for chunk in bytes.chunks_exact(32) {
            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Compare with patterns
            let cache_matches = _mm256_cmpeq_epi8(data, cache_pattern);
            let search_matches = _mm256_cmpeq_epi8(data, search_pattern);
            let storage_matches = _mm256_cmpeq_epi8(data, storage_pattern);
            
            // Count matches
            cache_score += _mm256_movemask_epi8(cache_matches).count_ones();
            search_score += _mm256_movemask_epi8(search_matches).count_ones();
            storage_score += _mm256_movemask_epi8(storage_matches).count_ones();
        }
        
        // Determine route based on scores
        if cache_score > search_score && cache_score > storage_score {
            Some(0) // Cache route
        } else if search_score > storage_score {
            Some(1) // Search route
        } else if storage_score > 0 {
            Some(2) // Storage route
        } else {
            Some(3) // Fusion route (default)
        }
    }
    
    /// SIMD analysis using SSE4.2
    #[target_feature(enable = "sse4.2")]
    unsafe fn analyze_simd_sse42(&self, bytes: &[u8]) -> Option<u32> {
        if bytes.len() < 16 {
            return self.analyze_scalar(bytes);
        }
        
        // Use SSE4.2 string instructions for pattern matching
        let patterns = [
            (*b"cache\0\0\0\0\0\0\0\0\0\0\0", 0),
            (*b"search\0\0\0\0\0\0\0\0\0\0", 1),
            (*b"store\0\0\0\0\0\0\0\0\0\0\0", 2),
            (*b"query\0\0\0\0\0\0\0\0\0\0\0", 1),
        ];
        
        for (pattern_bytes, route_id) in patterns.iter() {
            let pattern = _mm_loadu_si128(pattern_bytes.as_ptr() as *const __m128i);
            
            for chunk in bytes.chunks_exact(16) {
                let data = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
                
                // Use PCMPISTRI for substring search
                let idx = _mm_cmpistri(
                    pattern,
                    data,
                    _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT,
                );
                
                if idx < 16 {
                    return Some(*route_id);
                }
            }
        }
        
        Some(3) // Default to fusion
    }
    
    /// Scalar fallback analysis
    fn analyze_scalar(&self, bytes: &[u8]) -> Option<u32> {
        let text = std::str::from_utf8(bytes).ok()?;
        let lower = text.to_lowercase();
        
        if lower.contains("cache") || lower.contains("get") {
            Some(0)
        } else if lower.contains("search") || lower.contains("query") {
            Some(1)
        } else if lower.contains("store") || lower.contains("save") {
            Some(2)
        } else {
            Some(3)
        }
    }
    
    /// Default routing patterns
    fn default_patterns() -> Vec<Pattern> {
        vec![
            Pattern {
                id: 0,
                keywords: vec!["cache".to_string(), "get".to_string(), "fetch".to_string()],
                priority: 1,
            },
            Pattern {
                id: 1,
                keywords: vec!["search".to_string(), "find".to_string(), "query".to_string()],
                priority: 2,
            },
            Pattern {
                id: 2,
                keywords: vec!["store".to_string(), "save".to_string(), "write".to_string()],
                priority: 3,
            },
            Pattern {
                id: 3,
                keywords: vec!["fusion".to_string(), "merge".to_string(), "combine".to_string()],
                priority: 4,
            },
        ]
    }
}

impl PatternMatcher {
    pub fn new(patterns: Vec<Pattern>) -> Self {
        let compiled = patterns
            .into_iter()
            .map(|p| {
                let bytes = p.keywords[0].as_bytes().to_vec();
                let mask = vec![0xFF; bytes.len()];
                CompiledPattern {
                    id: p.id,
                    bytes,
                    mask,
                }
            })
            .collect();
            
        Self { compiled }
    }
    
    /// Match patterns in text using SIMD
    pub fn match_patterns(&self, text: &str) -> Vec<u32> {
        let mut matches = Vec::new();
        let bytes = text.as_bytes();
        
        for pattern in &self.compiled {
            if self.match_single_pattern(bytes, pattern) {
                matches.push(pattern.id);
            }
        }
        
        matches
    }
    
    /// Match a single pattern
    fn match_single_pattern(&self, haystack: &[u8], pattern: &CompiledPattern) -> bool {
        if pattern.bytes.len() > haystack.len() {
            return false;
        }
        
        // Use memchr for the first byte
        let first_byte = pattern.bytes[0];
        let mut pos = 0;
        
        while let Some(idx) = memchr(first_byte, &haystack[pos..]) {
            let abs_idx = pos + idx;
            
            if abs_idx + pattern.bytes.len() <= haystack.len() {
                let slice = &haystack[abs_idx..abs_idx + pattern.bytes.len()];
                if slice == pattern.bytes.as_slice() {
                    return true;
                }
            }
            
            pos = abs_idx + 1;
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_analyzer_basic() {
        let analyzer = SimdAnalyzer::new();
        
        assert_eq!(analyzer.analyze_pattern("get user data"), Some(0));
        assert_eq!(analyzer.analyze_pattern("search for documents"), Some(1));
        assert_eq!(analyzer.analyze_pattern("store new record"), Some(2));
        assert_eq!(analyzer.analyze_pattern("random text"), Some(3));
    }
    
    #[test]
    fn test_pattern_matcher() {
        let patterns = vec![
            Pattern {
                id: 1,
                keywords: vec!["hello".to_string()],
                priority: 1,
            },
            Pattern {
                id: 2,
                keywords: vec!["world".to_string()],
                priority: 1,
            },
        ];
        
        let matcher = PatternMatcher::new(patterns);
        let matches = matcher.match_patterns("hello world");
        
        assert!(matches.contains(&1));
        assert!(matches.contains(&2));
    }
}