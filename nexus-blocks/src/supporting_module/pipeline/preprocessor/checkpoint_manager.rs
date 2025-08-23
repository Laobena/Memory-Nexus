//! Checkpoint management for document processing recovery

use crate::core::{BlockError, BlockResult};
use super::ProcessedChunk;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Checkpoint for document processing
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Document ID
    pub document_id: String,
    /// Last processed chunk index
    pub last_chunk_index: usize,
    /// Last processed offset in document
    pub last_offset: usize,
    /// Chunks processed so far
    pub processed_chunks: Vec<ProcessedChunk>,
    /// Timestamp of checkpoint
    pub timestamp: Instant,
}

/// Recovery state from checkpoint
#[derive(Debug, Clone)]
pub struct RecoveryState {
    /// Last processed chunk
    pub last_chunk: usize,
    /// Last processed offset
    pub last_offset: usize,
    /// Already processed chunks
    pub processed_chunks: Vec<ProcessedChunk>,
}

/// Checkpoint manager for recovery
pub struct CheckpointManager {
    /// Active checkpoints
    checkpoints: Arc<DashMap<String, Checkpoint>>,
    /// Checkpoint retention duration
    retention: Duration,
    /// Maximum checkpoints to keep
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            checkpoints: Arc::new(DashMap::new()),
            retention: Duration::from_secs(3600), // 1 hour
            max_checkpoints: 100,
        }
    }
    
    /// Save checkpoint for document
    pub async fn save_checkpoint(
        &self,
        document_id: &str,
        chunks: &[ProcessedChunk],
    ) -> BlockResult<()> {
        let last_chunk = chunks.last()
            .ok_or_else(|| BlockError::Processing("No chunks to checkpoint".into()))?;
        
        let checkpoint = Checkpoint {
            document_id: document_id.to_string(),
            last_chunk_index: chunks.len() - 1,
            last_offset: last_chunk.end_offset,
            processed_chunks: chunks.to_vec(),
            timestamp: Instant::now(),
        };
        
        self.checkpoints.insert(document_id.to_string(), checkpoint);
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints();
        
        debug!("Saved checkpoint for document {}", document_id);
        Ok(())
    }
    
    /// Save intermediate checkpoint
    pub async fn save_intermediate(
        &self,
        document_id: &str,
        chunk_index: usize,
    ) -> BlockResult<()> {
        if let Some(mut checkpoint) = self.checkpoints.get_mut(document_id) {
            checkpoint.last_chunk_index = chunk_index;
            checkpoint.timestamp = Instant::now();
            debug!("Updated intermediate checkpoint for {} at chunk {}", document_id, chunk_index);
        }
        Ok(())
    }
    
    /// Recover from checkpoint
    pub async fn recover(&self, document_id: &str) -> BlockResult<Option<RecoveryState>> {
        if let Some(checkpoint) = self.checkpoints.get(document_id) {
            info!("Recovering from checkpoint for document {}", document_id);
            
            Ok(Some(RecoveryState {
                last_chunk: checkpoint.last_chunk_index,
                last_offset: checkpoint.last_offset,
                processed_chunks: checkpoint.processed_chunks.clone(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Remove checkpoint
    pub fn remove_checkpoint(&self, document_id: &str) {
        self.checkpoints.remove(document_id);
        debug!("Removed checkpoint for document {}", document_id);
    }
    
    /// Clean up old checkpoints
    fn cleanup_old_checkpoints(&self) {
        let now = Instant::now();
        let mut to_remove = Vec::new();
        
        // Find expired checkpoints
        for entry in self.checkpoints.iter() {
            if now.duration_since(entry.timestamp) > self.retention {
                to_remove.push(entry.key().clone());
            }
        }
        
        // Remove expired
        for key in to_remove {
            self.checkpoints.remove(&key);
        }
        
        // Enforce max checkpoints
        if self.checkpoints.len() > self.max_checkpoints {
            let mut entries: Vec<_> = self.checkpoints.iter()
                .map(|e| (e.key().clone(), e.timestamp))
                .collect();
            
            entries.sort_by_key(|e| e.1);
            
            let to_remove = entries.len() - self.max_checkpoints;
            for (key, _) in entries.into_iter().take(to_remove) {
                self.checkpoints.remove(&key);
            }
        }
    }
    
    /// Get checkpoint statistics
    pub fn stats(&self) -> CheckpointStats {
        CheckpointStats {
            total_checkpoints: self.checkpoints.len(),
            oldest_checkpoint: self.checkpoints.iter()
                .map(|e| e.timestamp)
                .min(),
        }
    }
}

/// Checkpoint statistics
#[derive(Debug)]
pub struct CheckpointStats {
    pub total_checkpoints: usize,
    pub oldest_checkpoint: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_checkpoint_save_recover() {
        let manager = CheckpointManager::new();
        
        let chunks = vec![
            ProcessedChunk {
                id: 0,
                start_offset: 0,
                end_offset: 100,
                content: vec![1, 2, 3],
                metadata: Default::default(),
            },
        ];
        
        // Save checkpoint
        manager.save_checkpoint("doc1", &chunks).await.unwrap();
        
        // Recover
        let recovery = manager.recover("doc1").await.unwrap();
        assert!(recovery.is_some());
        
        let state = recovery.unwrap();
        assert_eq!(state.last_chunk, 0);
        assert_eq!(state.last_offset, 100);
    }
}