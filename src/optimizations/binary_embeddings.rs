use crate::core::Result;
use bitvec::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

static BINARY_OPS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Binary embedding for 32x compression and 24x search speedup
pub struct BinaryEmbedding {
    bits: BitVec<u8, Lsb0>,
    dimension: usize,
}

impl BinaryEmbedding {
    /// Create from dense float vector
    pub fn from_dense(dense: &[f32]) -> Self {
        BINARY_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        let mut bits = BitVec::with_capacity(dense.len());
        let mean = calculate_mean(dense);
        
        // Binary quantization: 1 if above mean, 0 if below
        for &value in dense {
            bits.push(value > mean);
        }
        
        Self {
            bits,
            dimension: dense.len(),
        }
    }
    
    /// Create from binary data
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Self {
        let bits = BitVec::from_slice(bytes);
        Self { bits, dimension }
    }
    
    /// Convert to bytes for storage (32x compression)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bits.as_raw_slice().to_vec()
    }
    
    /// Hamming distance (for fast similarity search)
    pub fn hamming_distance(&self, other: &BinaryEmbedding) -> u32 {
        if self.dimension != other.dimension {
            return u32::MAX;
        }
        
        BINARY_OPS_COUNT.fetch_add(1, Ordering::Relaxed);
        
        // XOR and count set bits
        let mut distance = 0u32;
        for (a, b) in self.bits.iter().zip(other.bits.iter()) {
            if a != b {
                distance += 1;
            }
        }
        
        distance
    }
    
    /// Jaccard similarity (alternative metric)
    pub fn jaccard_similarity(&self, other: &BinaryEmbedding) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }
        
        let mut intersection = 0u32;
        let mut union = 0u32;
        
        for (a, b) in self.bits.iter().zip(other.bits.iter()) {
            if *a && *b {
                intersection += 1;
            }
            if *a || *b {
                union += 1;
            }
        }
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
    
    /// Fast batch search using binary embeddings
    pub fn batch_search(&self, candidates: &[BinaryEmbedding], top_k: usize) -> Vec<(usize, u32)> {
        use rayon::prelude::*;
        
        BINARY_OPS_COUNT.fetch_add(candidates.len() as u64, Ordering::Relaxed);
        
        // Parallel distance computation
        let mut distances: Vec<_> = candidates
            .par_iter()
            .enumerate()
            .map(|(idx, candidate)| (idx, self.hamming_distance(candidate)))
            .collect();
        
        // Sort by distance (lower is better)
        distances.sort_by_key(|&(_, dist)| dist);
        distances.truncate(top_k);
        
        distances
    }
    
    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // Original: 4 bytes per float * dimension
        // Compressed: 1 bit per dimension / 8 bits per byte
        let original_bytes = self.dimension * 4;
        let compressed_bytes = (self.dimension + 7) / 8;
        
        original_bytes as f32 / compressed_bytes as f32
    }
}

/// Binary embedding index for fast search
pub struct BinaryIndex {
    embeddings: Vec<BinaryEmbedding>,
    metadata: Vec<String>,
}

impl BinaryIndex {
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            metadata: Vec::new(),
        }
    }
    
    /// Add embedding to index
    pub fn add(&mut self, embedding: BinaryEmbedding, metadata: String) {
        self.embeddings.push(embedding);
        self.metadata.push(metadata);
    }
    
    /// Search index
    pub fn search(&self, query: &BinaryEmbedding, top_k: usize) -> Vec<(String, u32)> {
        let results = query.batch_search(&self.embeddings, top_k);
        
        results
            .into_iter()
            .map(|(idx, dist)| (self.metadata[idx].clone(), dist))
            .collect()
    }
    
    /// Get index size
    pub fn size(&self) -> usize {
        self.embeddings.len()
    }
    
    /// Calculate memory usage
    pub fn memory_usage_bytes(&self) -> usize {
        self.embeddings
            .iter()
            .map(|e| e.to_bytes().len())
            .sum::<usize>()
            + self.metadata
                .iter()
                .map(|m| m.len())
                .sum::<usize>()
    }
}

/// Multi-bit quantization for higher precision
pub struct MultiBitEmbedding {
    embeddings: Vec<BinaryEmbedding>,
    bits_per_dimension: usize,
}

impl MultiBitEmbedding {
    /// Create with specified bits per dimension (2, 4, or 8)
    pub fn from_dense(dense: &[f32], bits_per_dimension: usize) -> Self {
        let num_embeddings = match bits_per_dimension {
            2 => 2,
            4 => 4,
            8 => 8,
            _ => 1,
        };
        
        let mut embeddings = Vec::with_capacity(num_embeddings);
        
        for i in 0..num_embeddings {
            // Use different thresholds for each bit
            let threshold = -1.0 + (2.0 * (i + 1) as f32 / (num_embeddings + 1) as f32);
            
            let mut bits = BitVec::with_capacity(dense.len());
            for &value in dense {
                bits.push(value > threshold);
            }
            
            embeddings.push(BinaryEmbedding {
                bits,
                dimension: dense.len(),
            });
        }
        
        Self {
            embeddings,
            bits_per_dimension,
        }
    }
    
    /// Compute distance using multiple bits
    pub fn distance(&self, other: &MultiBitEmbedding) -> u32 {
        self.embeddings
            .iter()
            .zip(other.embeddings.iter())
            .map(|(a, b)| a.hamming_distance(b))
            .sum()
    }
}

fn calculate_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Get binary operation count
pub fn get_operation_count() -> u64 {
    BINARY_OPS_COUNT.load(Ordering::Relaxed)
}