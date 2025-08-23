//! Write-ahead logging for crash recovery

use crate::core::{BlockError, BlockResult};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// WAL configuration
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum WAL file size before rotation
    pub max_file_size: usize,
    /// Maximum number of WAL files to keep
    pub max_files: usize,
    /// Sync to disk after every write
    pub sync_on_write: bool,
    /// Checkpoint interval
    pub checkpoint_interval: std::time::Duration,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_file_size: 10_485_760, // 10MB
            max_files: 5,
            sync_on_write: true,
            checkpoint_interval: std::time::Duration::from_secs(60),
        }
    }
}

/// WAL entry for atomic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Transaction ID
    pub tx_id: Uuid,
    /// Operation to perform
    pub operation: WalOperation,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Checksum for validation
    pub checksum: u32,
}

/// WAL operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    Write { key: Uuid, value: Vec<u8> },
    Delete { key: Uuid },
    Update { key: Uuid, value: Vec<u8> },
    Batch { operations: Vec<WalOperation> },
    Commit { tx_id: Uuid },
    Rollback { tx_id: Uuid },
}

/// Write-ahead log implementation
pub struct WriteAheadLog {
    /// Configuration
    config: Arc<WalConfig>,
    
    /// Current WAL entries
    entries: Arc<RwLock<VecDeque<WalEntry>>>,
    
    /// Uncommitted transactions
    uncommitted: Arc<RwLock<Vec<WalEntry>>>,
    
    /// WAL storage backend
    #[cfg(feature = "sled")]
    storage: Arc<sled::Db>,
    
    /// Last checkpoint
    last_checkpoint: Arc<RwLock<DateTime<Utc>>>,
}

impl WriteAheadLog {
    /// Create new WAL
    pub fn new(config: WalConfig) -> BlockResult<Self> {
        #[cfg(feature = "sled")]
        let storage = sled::open("wal.db")?;
        
        Ok(Self {
            config: Arc::new(config),
            entries: Arc::new(RwLock::new(VecDeque::new())),
            uncommitted: Arc::new(RwLock::new(Vec::new())),
            #[cfg(feature = "sled")]
            storage: Arc::new(storage),
            last_checkpoint: Arc::new(RwLock::new(Utc::now())),
        })
    }
    
    /// Append entry to WAL
    pub async fn append(&self, mut entry: WalEntry) -> BlockResult<()> {
        // Calculate checksum
        entry.checksum = self.calculate_checksum(&entry);
        
        // Add to entries
        {
            let mut entries = self.entries.write();
            entries.push_back(entry.clone());
            
            // Track uncommitted
            if !matches!(entry.operation, WalOperation::Commit { .. } | WalOperation::Rollback { .. }) {
                self.uncommitted.write().push(entry.clone());
            }
        }
        
        // Persist to storage
        #[cfg(feature = "sled")]
        {
            let key = entry.tx_id.as_bytes();
            let value = bincode::serialize(&entry)?;
            self.storage.insert(key, value)?;
            
            if self.config.sync_on_write {
                self.storage.flush()?;
            }
        }
        
        debug!("Appended WAL entry for tx: {}", entry.tx_id);
        Ok(())
    }
    
    /// Commit transaction
    pub async fn commit(&self, tx_id: Uuid) -> BlockResult<()> {
        let entry = WalEntry {
            tx_id,
            operation: WalOperation::Commit { tx_id },
            timestamp: Utc::now(),
            checksum: 0,
        };
        
        self.append(entry).await?;
        
        // Remove from uncommitted
        let mut uncommitted = self.uncommitted.write();
        uncommitted.retain(|e| e.tx_id != tx_id);
        
        info!("Committed transaction: {}", tx_id);
        Ok(())
    }
    
    /// Rollback transaction
    pub async fn rollback(&self, tx_id: Uuid) -> BlockResult<()> {
        let entry = WalEntry {
            tx_id,
            operation: WalOperation::Rollback { tx_id },
            timestamp: Utc::now(),
            checksum: 0,
        };
        
        self.append(entry).await?;
        
        // Remove from uncommitted
        let mut uncommitted = self.uncommitted.write();
        uncommitted.retain(|e| e.tx_id != tx_id);
        
        warn!("Rolled back transaction: {}", tx_id);
        Ok(())
    }
    
    /// Get uncommitted entries for recovery
    pub async fn get_uncommitted(&self) -> BlockResult<Vec<WalEntry>> {
        Ok(self.uncommitted.read().clone())
    }
    
    /// Checkpoint WAL
    pub async fn checkpoint(&self) -> BlockResult<()> {
        let now = Utc::now();
        
        // Get committed entries
        let entries = self.entries.read();
        let committed: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, WalOperation::Commit { .. }))
            .cloned()
            .collect();
        
        // Clear old entries
        drop(entries);
        let mut entries = self.entries.write();
        entries.retain(|e| e.timestamp > *self.last_checkpoint.read());
        
        *self.last_checkpoint.write() = now;
        
        info!("Checkpointed WAL, retained {} entries", entries.len());
        Ok(())
    }
    
    /// Replay WAL entries for recovery
    pub async fn replay<F>(&self, mut apply_fn: F) -> BlockResult<usize>
    where
        F: FnMut(WalEntry) -> BlockResult<()>,
    {
        let mut replayed = 0;
        
        #[cfg(feature = "sled")]
        {
            // Load all entries from storage
            for result in self.storage.iter() {
                let (key, value) = result?;
                let entry: WalEntry = bincode::deserialize(&value)?;
                
                // Validate checksum
                if !self.validate_checksum(&entry) {
                    error!("Invalid checksum for entry: {:?}", entry.tx_id);
                    continue;
                }
                
                // Apply entry
                match apply_fn(entry.clone()) {
                    Ok(_) => {
                        replayed += 1;
                        debug!("Replayed entry: {:?}", entry.tx_id);
                    }
                    Err(e) => {
                        warn!("Failed to replay entry {:?}: {}", entry.tx_id, e);
                    }
                }
            }
        }
        
        info!("Replayed {} WAL entries", replayed);
        Ok(replayed)
    }
    
    /// Calculate checksum for entry
    fn calculate_checksum(&self, entry: &WalEntry) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        entry.tx_id.hash(&mut hasher);
        entry.timestamp.timestamp().hash(&mut hasher);
        
        match &entry.operation {
            WalOperation::Write { key, value } => {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
            WalOperation::Delete { key } => {
                key.hash(&mut hasher);
            }
            WalOperation::Update { key, value } => {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
            _ => {}
        }
        
        hasher.finish() as u32
    }
    
    /// Validate checksum
    fn validate_checksum(&self, entry: &WalEntry) -> bool {
        let calculated = self.calculate_checksum(entry);
        calculated == entry.checksum
    }
    
    /// Truncate WAL (dangerous - only for maintenance)
    pub async fn truncate(&self) -> BlockResult<()> {
        self.entries.write().clear();
        self.uncommitted.write().clear();
        
        #[cfg(feature = "sled")]
        {
            self.storage.clear()?;
            self.storage.flush()?;
        }
        
        warn!("WAL truncated");
        Ok(())
    }
    
    /// Get WAL statistics
    pub fn stats(&self) -> WalStats {
        let entries = self.entries.read();
        let uncommitted = self.uncommitted.read();
        
        WalStats {
            total_entries: entries.len(),
            uncommitted_entries: uncommitted.len(),
            last_checkpoint: *self.last_checkpoint.read(),
            #[cfg(feature = "sled")]
            storage_size: self.storage.size_on_disk().unwrap_or(0),
            #[cfg(not(feature = "sled"))]
            storage_size: 0,
        }
    }
}

/// WAL statistics
#[derive(Debug)]
pub struct WalStats {
    pub total_entries: usize,
    pub uncommitted_entries: usize,
    pub last_checkpoint: DateTime<Utc>,
    pub storage_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_wal_append_commit() {
        let wal = WriteAheadLog::new(WalConfig::default()).unwrap();
        let tx_id = Uuid::new_v4();
        
        // Append write operation
        let entry = WalEntry {
            tx_id,
            operation: WalOperation::Write {
                key: Uuid::new_v4(),
                value: vec![1, 2, 3],
            },
            timestamp: Utc::now(),
            checksum: 0,
        };
        
        wal.append(entry).await.unwrap();
        
        // Should be uncommitted
        let uncommitted = wal.get_uncommitted().await.unwrap();
        assert_eq!(uncommitted.len(), 1);
        
        // Commit
        wal.commit(tx_id).await.unwrap();
        
        // Should be committed now
        let uncommitted = wal.get_uncommitted().await.unwrap();
        assert_eq!(uncommitted.len(), 0);
    }
}