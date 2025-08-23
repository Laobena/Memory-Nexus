//! Chunking strategies for text processing

use crate::core::BlockResult;

/// Chunking strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks
    Fixed,
    /// Semantic boundary-aware chunks
    Semantic,
    /// Sliding window chunks
    Sliding,
}

/// Chunk boundary types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkBoundary {
    /// No specific boundary
    None,
    /// Word boundaries
    Word,
    /// Sentence boundaries
    Sentence,
    /// Paragraph boundaries
    Paragraph,
}

/// Chunk configuration
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in tokens
    pub size: usize,
    /// Overlap between chunks in tokens
    pub overlap: usize,
    /// Boundary type for semantic chunking
    pub boundary_type: ChunkBoundary,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            size: 512,
            overlap: 50,
            boundary_type: ChunkBoundary::Sentence,
        }
    }
}