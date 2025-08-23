//! Data validation and integrity checks

use crate::core::{BlockError, BlockResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Checksum types for validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChecksumType {
    /// CRC32 checksum
    Crc32,
    /// XXHash checksum
    XxHash,
    /// Blake3 hash
    Blake3,
    /// Simple hash
    Simple,
}

/// Validation result
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// Data is valid
    Valid,
    /// Data is corrupted
    Corrupted {
        expected: Vec<u8>,
        actual: Vec<u8>,
        error: String,
    },
    /// Checksum mismatch
    ChecksumMismatch {
        expected: u64,
        actual: u64,
    },
}

/// Data validator for integrity checks
pub struct DataValidator {
    checksum_type: ChecksumType,
    validate_on_read: bool,
    validate_on_write: bool,
}

impl DataValidator {
    /// Create new validator
    pub fn new(checksum_type: ChecksumType) -> Self {
        Self {
            checksum_type,
            validate_on_read: true,
            validate_on_write: true,
        }
    }
    
    /// Calculate checksum for data
    pub fn calculate_checksum(&self, data: &[u8]) -> u64 {
        match self.checksum_type {
            ChecksumType::Crc32 => self.crc32_checksum(data),
            ChecksumType::XxHash => self.xxhash_checksum(data),
            ChecksumType::Blake3 => self.blake3_checksum(data),
            ChecksumType::Simple => self.simple_checksum(data),
        }
    }
    
    /// Validate data with checksum
    pub fn validate(&self, data: &[u8], expected_checksum: u64) -> ValidationResult {
        let actual_checksum = self.calculate_checksum(data);
        
        if actual_checksum == expected_checksum {
            ValidationResult::Valid
        } else {
            ValidationResult::ChecksumMismatch {
                expected: expected_checksum,
                actual: actual_checksum,
            }
        }
    }
    
    /// Validate key-value pair
    pub fn validate_kv(&self, key: &Uuid, value: &[u8], checksum: u64) -> ValidationResult {
        // Combine key and value for validation
        let mut data = key.as_bytes().to_vec();
        data.extend_from_slice(value);
        
        self.validate(&data, checksum)
    }
    
    /// CRC32 checksum
    fn crc32_checksum(&self, data: &[u8]) -> u64 {
        // Simple CRC32 implementation
        let mut crc = 0xFFFFFFFFu32;
        
        for byte in data {
            crc ^= *byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        
        (!crc) as u64
    }
    
    /// XXHash checksum (simplified)
    fn xxhash_checksum(&self, data: &[u8]) -> u64 {
        // Simplified XXHash implementation
        let mut h64 = 0x9e3779b185ebca87u64;
        
        for chunk in data.chunks(8) {
            let mut v = 0u64;
            for (i, &byte) in chunk.iter().enumerate() {
                v |= (byte as u64) << (i * 8);
            }
            
            h64 ^= v;
            h64 = h64.rotate_left(31);
            h64 = h64.wrapping_mul(0x165667919E3779F9);
        }
        
        h64
    }
    
    /// Blake3 checksum (placeholder)
    fn blake3_checksum(&self, data: &[u8]) -> u64 {
        // Would use blake3 crate in production
        self.simple_checksum(data)
    }
    
    /// Simple checksum using default hasher
    fn simple_checksum(&self, data: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Create protected write with checksum
    pub fn protect_write(&self, key: Uuid, value: Vec<u8>) -> ProtectedData {
        let checksum = if self.validate_on_write {
            let mut data = key.as_bytes().to_vec();
            data.extend_from_slice(&value);
            self.calculate_checksum(&data)
        } else {
            0
        };
        
        ProtectedData {
            key,
            value,
            checksum,
            checksum_type: self.checksum_type,
        }
    }
    
    /// Validate protected data
    pub fn validate_protected(&self, data: &ProtectedData) -> ValidationResult {
        if !self.validate_on_read {
            return ValidationResult::Valid;
        }
        
        self.validate_kv(&data.key, &data.value, data.checksum)
    }
}

/// Protected data with checksum
#[derive(Debug, Clone)]
pub struct ProtectedData {
    pub key: Uuid,
    pub value: Vec<u8>,
    pub checksum: u64,
    pub checksum_type: ChecksumType,
}

impl ProtectedData {
    /// Create new protected data
    pub fn new(key: Uuid, value: Vec<u8>, checksum_type: ChecksumType) -> Self {
        let validator = DataValidator::new(checksum_type);
        validator.protect_write(key, value)
    }
    
    /// Validate integrity
    pub fn validate(&self) -> ValidationResult {
        let validator = DataValidator::new(self.checksum_type);
        validator.validate_protected(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_checksum_validation() {
        let validator = DataValidator::new(ChecksumType::Simple);
        let data = b"test data";
        
        let checksum = validator.calculate_checksum(data);
        let result = validator.validate(data, checksum);
        
        assert!(matches!(result, ValidationResult::Valid));
        
        // Test with wrong checksum
        let result = validator.validate(data, checksum + 1);
        assert!(matches!(result, ValidationResult::ChecksumMismatch { .. }));
    }
    
    #[test]
    fn test_protected_data() {
        let key = Uuid::new_v4();
        let value = vec![1, 2, 3, 4, 5];
        
        let protected = ProtectedData::new(key, value.clone(), ChecksumType::Crc32);
        
        // Should be valid
        assert!(matches!(protected.validate(), ValidationResult::Valid));
        
        // Corrupt data
        let mut corrupted = protected.clone();
        corrupted.value[0] = 255;
        
        // Should detect corruption
        assert!(matches!(corrupted.validate(), ValidationResult::ChecksumMismatch { .. }));
    }
}