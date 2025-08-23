//! UTF-8 validation and repair utilities

use std::borrow::Cow;

/// UTF-8 validation result
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// Text is valid UTF-8
    Valid,
    /// Text has errors at specified positions
    Invalid(Vec<Utf8Error>),
}

/// UTF-8 error information
#[derive(Debug, Clone)]
pub struct Utf8Error {
    /// Byte position of error
    pub position: usize,
    /// Length of invalid sequence
    pub length: usize,
}

/// UTF-8 validator with repair capabilities
pub struct Utf8Validator {
    /// Replacement character for invalid sequences
    replacement: char,
}

impl Utf8Validator {
    pub fn new() -> Self {
        Self {
            replacement: '\u{FFFD}', // Unicode replacement character
        }
    }
    
    /// Validate UTF-8 bytes
    pub fn validate(&self, bytes: &[u8]) -> ValidationResult {
        match std::str::from_utf8(bytes) {
            Ok(_) => ValidationResult::Valid,
            Err(e) => {
                let mut errors = Vec::new();
                let mut pos = e.valid_up_to();
                
                // Collect all errors
                let mut remaining = &bytes[pos..];
                while !remaining.is_empty() {
                    match std::str::from_utf8(remaining) {
                        Ok(_) => break,
                        Err(e) => {
                            let error_len = e.error_len().unwrap_or(1);
                            errors.push(Utf8Error {
                                position: pos,
                                length: error_len,
                            });
                            
                            pos += e.valid_up_to() + error_len;
                            if pos >= bytes.len() {
                                break;
                            }
                            remaining = &bytes[pos..];
                        }
                    }
                }
                
                ValidationResult::Invalid(errors)
            }
        }
    }
    
    /// Repair UTF-8 with lossy conversion
    pub fn repair_lossy(&self, bytes: &[u8]) -> Vec<u8> {
        String::from_utf8_lossy(bytes).into_owned().into_bytes()
    }
    
    /// Repair UTF-8 with custom replacement
    pub fn repair_with_replacement(&self, bytes: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(bytes.len());
        let mut pos = 0;
        
        while pos < bytes.len() {
            match std::str::from_utf8(&bytes[pos..]) {
                Ok(valid) => {
                    result.extend_from_slice(valid.as_bytes());
                    break;
                }
                Err(e) => {
                    // Add valid portion
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to > 0 {
                        result.extend_from_slice(&bytes[pos..pos + valid_up_to]);
                    }
                    
                    // Add replacement character
                    let replacement_bytes = self.replacement.to_string().into_bytes();
                    result.extend_from_slice(&replacement_bytes);
                    
                    // Skip invalid bytes
                    pos += valid_up_to + e.error_len().unwrap_or(1);
                }
            }
        }
        
        result
    }
    
    /// Try to repair specific error patterns
    pub fn smart_repair(&self, bytes: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(bytes.len());
        let mut i = 0;
        
        while i < bytes.len() {
            let byte = bytes[i];
            
            // Check for common encoding issues
            if byte < 0x80 {
                // ASCII - always valid
                result.push(byte);
                i += 1;
            } else if byte >= 0xC0 && byte < 0xE0 {
                // 2-byte sequence
                if i + 1 < bytes.len() && bytes[i + 1] >= 0x80 && bytes[i + 1] < 0xC0 {
                    result.push(byte);
                    result.push(bytes[i + 1]);
                    i += 2;
                } else {
                    // Invalid sequence, replace
                    result.extend_from_slice(&self.replacement.to_string().into_bytes());
                    i += 1;
                }
            } else if byte >= 0xE0 && byte < 0xF0 {
                // 3-byte sequence
                if i + 2 < bytes.len() 
                    && bytes[i + 1] >= 0x80 && bytes[i + 1] < 0xC0
                    && bytes[i + 2] >= 0x80 && bytes[i + 2] < 0xC0 {
                    result.push(byte);
                    result.push(bytes[i + 1]);
                    result.push(bytes[i + 2]);
                    i += 3;
                } else {
                    // Invalid sequence, replace
                    result.extend_from_slice(&self.replacement.to_string().into_bytes());
                    i += 1;
                }
            } else if byte >= 0xF0 && byte < 0xF8 {
                // 4-byte sequence
                if i + 3 < bytes.len()
                    && bytes[i + 1] >= 0x80 && bytes[i + 1] < 0xC0
                    && bytes[i + 2] >= 0x80 && bytes[i + 2] < 0xC0
                    && bytes[i + 3] >= 0x80 && bytes[i + 3] < 0xC0 {
                    result.push(byte);
                    result.push(bytes[i + 1]);
                    result.push(bytes[i + 2]);
                    result.push(bytes[i + 3]);
                    i += 4;
                } else {
                    // Invalid sequence, replace
                    result.extend_from_slice(&self.replacement.to_string().into_bytes());
                    i += 1;
                }
            } else {
                // Invalid start byte, replace
                result.extend_from_slice(&self.replacement.to_string().into_bytes());
                i += 1;
            }
        }
        
        result
    }
}

impl Default for Utf8Validator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_utf8() {
        let validator = Utf8Validator::new();
        let valid = "Hello, 世界! 🦀".as_bytes();
        
        match validator.validate(valid) {
            ValidationResult::Valid => {}
            _ => panic!("Expected valid UTF-8"),
        }
    }
    
    #[test]
    fn test_invalid_utf8() {
        let validator = Utf8Validator::new();
        let invalid = vec![0xFF, 0xFE, b'h', b'e', b'l', b'l', b'o'];
        
        match validator.validate(&invalid) {
            ValidationResult::Invalid(errors) => {
                assert!(!errors.is_empty());
            }
            _ => panic!("Expected invalid UTF-8"),
        }
    }
    
    #[test]
    fn test_repair_lossy() {
        let validator = Utf8Validator::new();
        let invalid = vec![b'h', b'e', 0xFF, 0xFE, b'l', b'l', b'o'];
        
        let repaired = validator.repair_lossy(&invalid);
        let repaired_str = std::str::from_utf8(&repaired).unwrap();
        
        assert!(repaired_str.contains("he"));
        assert!(repaired_str.contains("llo"));
    }
}