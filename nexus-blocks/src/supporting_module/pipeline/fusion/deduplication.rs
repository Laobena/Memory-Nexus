//! MinHash deduplication with SIMD acceleration

use crate::core::{BlockError, BlockResult};
use super::{FusionItem, PartialResults};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::debug;

/// Deduplication configuration
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    /// Number of hash functions
    pub num_hashes: usize,
    /// Similarity threshold (0.0-1.0)
    pub threshold: f32,
    /// Use SIMD for hash comparison
    pub use_simd: bool,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            threshold: 0.7,
            use_simd: true,
        }
    }
}

/// MinHash signature
#[derive(Debug, Clone)]
struct MinHashSignature {
    hashes: Vec<u64>,
}

/// MinHash deduplicator
pub struct MinHashDeduplicator {
    threshold: f32,
    num_hashes: usize,
    stats: Arc<RwLock<DeduplicationStats>>,
}

#[derive(Debug, Default)]
struct DeduplicationStats {
    total_items: u64,
    duplicates_found: u64,
    signatures_computed: u64,
}

impl MinHashDeduplicator {
    /// Create new deduplicator
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            num_hashes: 128,
            stats: Arc::new(RwLock::new(DeduplicationStats::default())),
        }
    }
    
    /// Deduplicate items
    pub async fn deduplicate(&self, results: &PartialResults) -> BlockResult<Vec<FusionItem>> {
        let items = results.get_items();
        
        if items.len() <= 1 {
            return Ok(items);
        }
        
        // Compute signatures
        let signatures = self.compute_signatures(&items)?;
        
        // Find duplicates using SIMD if available
        let duplicates = if cfg!(feature = "fusion") {
            self.find_duplicates_simd(&signatures)?
        } else {
            self.find_duplicates_scalar(&signatures)?
        };
        
        // Filter out duplicates
        let mut unique = Vec::new();
        let mut seen = HashSet::new();
        
        for (i, item) in items.into_iter().enumerate() {
            if !duplicates.contains(&i) || seen.insert(i) {
                unique.push(item);
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_items += items.len() as u64;
            stats.duplicates_found += duplicates.len() as u64;
            stats.signatures_computed += signatures.len() as u64;
        }
        
        debug!(
            "Deduplicated {} items to {}, removed {} duplicates",
            items.len(),
            unique.len(),
            items.len() - unique.len()
        );
        
        Ok(unique)
    }
    
    /// Compute MinHash signatures
    fn compute_signatures(&self, items: &[FusionItem]) -> BlockResult<Vec<MinHashSignature>> {
        let mut signatures = Vec::with_capacity(items.len());
        
        for item in items {
            let sig = self.compute_signature(&item.content)?;
            signatures.push(sig);
        }
        
        Ok(signatures)
    }
    
    /// Compute signature for single item
    fn compute_signature(&self, content: &[u8]) -> BlockResult<MinHashSignature> {
        let mut hashes = Vec::with_capacity(self.num_hashes);
        
        // Generate shingles (3-grams)
        let shingles = self.generate_shingles(content, 3);
        
        // Compute MinHash
        for i in 0..self.num_hashes {
            let mut min_hash = u64::MAX;
            
            for shingle in &shingles {
                let hash = self.hash_shingle(shingle, i);
                min_hash = min_hash.min(hash);
            }
            
            hashes.push(min_hash);
        }
        
        Ok(MinHashSignature { hashes })
    }
    
    /// Generate shingles from content
    fn generate_shingles(&self, content: &[u8], k: usize) -> Vec<Vec<u8>> {
        if content.len() < k {
            return vec![content.to_vec()];
        }
        
        let mut shingles = Vec::new();
        for i in 0..=content.len() - k {
            shingles.push(content[i..i + k].to_vec());
        }
        
        shingles
    }
    
    /// Hash a shingle with seed
    fn hash_shingle(&self, shingle: &[u8], seed: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        use ahash::AHasher;
        
        let mut hasher = AHasher::new_with_keys(seed as u128, (seed as u128) << 64);
        shingle.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Find duplicates using SIMD
    #[cfg(feature = "fusion")]
    fn find_duplicates_simd(&self, signatures: &[MinHashSignature]) -> BlockResult<HashSet<usize>> {
        let mut duplicates = HashSet::new();
        
        #[cfg(all(feature = "simdeez", target_arch = "x86_64"))]
        {
            use simdeez::prelude::*;
            
            for i in 0..signatures.len() {
                for j in i + 1..signatures.len() {
                    let similarity = self.compute_similarity_simd(&signatures[i], &signatures[j])?;
                    
                    if similarity >= self.threshold {
                        duplicates.insert(j); // Mark later item as duplicate
                    }
                }
            }
        }
        
        #[cfg(not(all(feature = "simdeez", target_arch = "x86_64")))]
        {
            return self.find_duplicates_scalar(signatures);
        }
        
        Ok(duplicates)
    }
    
    /// Compute similarity using SIMD
    #[cfg(all(feature = "simdeez", feature = "fusion", target_arch = "x86_64"))]
    fn compute_similarity_simd(&self, sig1: &MinHashSignature, sig2: &MinHashSignature) -> BlockResult<f32> {
        use simdeez::prelude::*;
        use simdeez::simd_runtime_generate;
        
        simd_runtime_generate!(
            fn count_matches(a: &[u64], b: &[u64]) -> u32 {
                let mut matches = 0u32;
                let mut idx = 0;
                
                while idx + S::Vi64::WIDTH <= a.len() {
                    let va = S::Vi64::load_from_slice(&a[idx..]);
                    let vb = S::Vi64::load_from_slice(&b[idx..]);
                    
                    let eq = va.cmp_eq(vb);
                    matches += eq.move_mask().count_ones();
                    
                    idx += S::Vi64::WIDTH;
                }
                
                // Handle remainder
                while idx < a.len() {
                    if a[idx] == b[idx] {
                        matches += 1;
                    }
                    idx += 1;
                }
                
                matches
            }
        );
        
        let matches = count_matches_runtime_select(&sig1.hashes, &sig2.hashes);
        let similarity = matches as f32 / self.num_hashes as f32;
        
        Ok(similarity)
    }
    
    /// Find duplicates using scalar operations
    fn find_duplicates_scalar(&self, signatures: &[MinHashSignature]) -> BlockResult<HashSet<usize>> {
        let mut duplicates = HashSet::new();
        
        for i in 0..signatures.len() {
            for j in i + 1..signatures.len() {
                let similarity = self.compute_similarity_scalar(&signatures[i], &signatures[j]);
                
                if similarity >= self.threshold {
                    duplicates.insert(j);
                }
            }
        }
        
        Ok(duplicates)
    }
    
    /// Compute similarity (scalar)
    fn compute_similarity_scalar(&self, sig1: &MinHashSignature, sig2: &MinHashSignature) -> f32 {
        let matches = sig1.hashes.iter()
            .zip(sig2.hashes.iter())
            .filter(|(a, b)| a == b)
            .count();
        
        matches as f32 / self.num_hashes as f32
    }
    
    /// Get deduplication statistics
    pub fn stats(&self) -> DeduplicationStatsSummary {
        let stats = self.stats.read();
        
        DeduplicationStatsSummary {
            total_items: stats.total_items,
            duplicates_found: stats.duplicates_found,
            deduplication_rate: if stats.total_items > 0 {
                stats.duplicates_found as f32 / stats.total_items as f32
            } else {
                0.0
            },
        }
    }
}

/// Deduplication statistics summary
#[derive(Debug)]
pub struct DeduplicationStatsSummary {
    pub total_items: u64,
    pub duplicates_found: u64,
    pub deduplication_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_minhash_deduplication() {
        let dedup = MinHashDeduplicator::new(0.7);
        
        let mut results = PartialResults::new();
        
        // Add duplicate items
        let content = b"This is test content for deduplication".to_vec();
        for i in 0..5 {
            results.add(FusionItem {
                id: uuid::Uuid::new_v4(),
                content: if i < 3 { content.clone() } else { vec![i] },
                relevance: 0.8,
                freshness: 0.7,
                diversity: 0.6,
                authority: 0.5,
                coherence: 0.4,
                confidence: 0.9,
                source_engine: super::EngineType::Accuracy,
                timestamp: chrono::Utc::now(),
            });
        }
        
        let deduplicated = dedup.deduplicate(&results).await.unwrap();
        
        // Should have removed duplicates
        assert!(deduplicated.len() < 5);
        assert!(deduplicated.len() >= 3); // At least 3 unique items
    }
}