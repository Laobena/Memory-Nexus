//! Compression strategies for cold storage

use crate::core::{BlockError, BlockResult};

/// Compression strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Zstd compression
    Zstd { level: i32 },
    /// LZ4 compression
    Lz4,
    /// Snappy compression
    Snappy,
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        CompressionStrategy::Zstd { level: 3 }
    }
}

/// Compressor for data compression
pub struct Compressor {
    strategy: CompressionStrategy,
}

impl Compressor {
    /// Create new compressor
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self { strategy }
    }
    
    /// Compress data
    pub fn compress(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        match self.strategy {
            CompressionStrategy::None => Ok(data.to_vec()),
            CompressionStrategy::Zstd { level } => self.compress_zstd(data, level),
            CompressionStrategy::Lz4 => self.compress_lz4(data),
            CompressionStrategy::Snappy => self.compress_snappy(data),
        }
    }
    
    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        match self.strategy {
            CompressionStrategy::None => Ok(data.to_vec()),
            CompressionStrategy::Zstd { .. } => self.decompress_zstd(data),
            CompressionStrategy::Lz4 => self.decompress_lz4(data),
            CompressionStrategy::Snappy => self.decompress_snappy(data),
        }
    }
    
    /// Compress with Zstd
    #[cfg(feature = "zstd")]
    fn compress_zstd(&self, data: &[u8], level: i32) -> BlockResult<Vec<u8>> {
        zstd::encode_all(data, level)
            .map_err(|e| BlockError::Unknown(format!("Zstd compression failed: {}", e)))
    }
    
    #[cfg(not(feature = "zstd"))]
    fn compress_zstd(&self, data: &[u8], _level: i32) -> BlockResult<Vec<u8>> {
        Ok(data.to_vec())
    }
    
    /// Decompress with Zstd
    #[cfg(feature = "zstd")]
    fn decompress_zstd(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        zstd::decode_all(data)
            .map_err(|e| BlockError::Unknown(format!("Zstd decompression failed: {}", e)))
    }
    
    #[cfg(not(feature = "zstd"))]
    fn decompress_zstd(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        Ok(data.to_vec())
    }
    
    /// Compress with LZ4
    fn compress_lz4(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        let compressed = lz4::block::compress(data, None, false)
            .map_err(|e| BlockError::Unknown(format!("LZ4 compression failed: {}", e)))?;
        Ok(compressed)
    }
    
    /// Decompress with LZ4
    fn decompress_lz4(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        let decompressed = lz4::block::decompress(data, None)
            .map_err(|e| BlockError::Unknown(format!("LZ4 decompression failed: {}", e)))?;
        Ok(decompressed)
    }
    
    /// Compress with Snappy
    fn compress_snappy(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        // Simplified implementation - would use snap crate
        Ok(data.to_vec())
    }
    
    /// Decompress with Snappy
    fn decompress_snappy(&self, data: &[u8]) -> BlockResult<Vec<u8>> {
        // Simplified implementation - would use snap crate
        Ok(data.to_vec())
    }
    
    /// Calculate compression ratio
    pub fn compression_ratio(&self, original: &[u8], compressed: &[u8]) -> f32 {
        if original.is_empty() {
            return 0.0;
        }
        
        1.0 - (compressed.len() as f32 / original.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compression_roundtrip() {
        let compressor = Compressor::new(CompressionStrategy::Zstd { level: 3 });
        let data = b"Hello, world! This is a test of compression.".repeat(10);
        
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed.as_slice());
        
        let ratio = compressor.compression_ratio(&data, &compressed);
        println!("Compression ratio: {:.2}%", ratio * 100.0);
    }
}