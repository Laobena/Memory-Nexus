//! Crash recovery mechanism

use crate::core::{BlockError, BlockResult};
use super::wal::{WalEntry, WalOperation, WriteAheadLog};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Recovery state during crash recovery
#[derive(Debug, Clone)]
pub struct RecoveryState {
    /// Entries to replay
    pub pending_entries: Vec<WalEntry>,
    /// Successfully replayed
    pub replayed_count: usize,
    /// Failed entries
    pub failed_entries: Vec<WalEntry>,
    /// Recovery progress (0.0 to 1.0)
    pub progress: f32,
}

/// Recovery statistics
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub total_entries: usize,
    pub replayed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration_ms: u64,
}

/// Recovery manager for crash recovery
pub struct RecoveryManager {
    /// WAL for recovery
    wal: Arc<WriteAheadLog>,
    
    /// Recovery state
    state: Arc<RwLock<Option<RecoveryState>>>,
    
    /// Recovery callbacks
    callbacks: Arc<RwLock<RecoveryCallbacks>>,
}

/// Recovery callbacks for different operations
struct RecoveryCallbacks {
    write_fn: Option<Box<dyn Fn(Uuid, Vec<u8>) -> BlockResult<()> + Send + Sync>>,
    delete_fn: Option<Box<dyn Fn(Uuid) -> BlockResult<()> + Send + Sync>>,
    update_fn: Option<Box<dyn Fn(Uuid, Vec<u8>) -> BlockResult<()> + Send + Sync>>,
}

impl RecoveryManager {
    /// Create new recovery manager
    pub fn new(wal: Arc<WriteAheadLog>) -> Self {
        Self {
            wal,
            state: Arc::new(RwLock::new(None)),
            callbacks: Arc::new(RwLock::new(RecoveryCallbacks {
                write_fn: None,
                delete_fn: None,
                update_fn: None,
            })),
        }
    }
    
    /// Set write callback
    pub fn set_write_callback<F>(&self, callback: F)
    where
        F: Fn(Uuid, Vec<u8>) -> BlockResult<()> + Send + Sync + 'static,
    {
        self.callbacks.write().write_fn = Some(Box::new(callback));
    }
    
    /// Set delete callback
    pub fn set_delete_callback<F>(&self, callback: F)
    where
        F: Fn(Uuid) -> BlockResult<()> + Send + Sync + 'static,
    {
        self.callbacks.write().delete_fn = Some(Box::new(callback));
    }
    
    /// Set update callback
    pub fn set_update_callback<F>(&self, callback: F)
    where
        F: Fn(Uuid, Vec<u8>) -> BlockResult<()> + Send + Sync + 'static,
    {
        self.callbacks.write().update_fn = Some(Box::new(callback));
    }
    
    /// Perform crash recovery
    pub async fn recover(&self) -> BlockResult<RecoveryStats> {
        info!("Starting crash recovery");
        let start = std::time::Instant::now();
        
        // Get uncommitted entries from WAL
        let pending = self.wal.get_uncommitted().await?;
        
        if pending.is_empty() {
            info!("No uncommitted entries found, recovery complete");
            return Ok(RecoveryStats::default());
        }
        
        info!("Found {} uncommitted entries to recover", pending.len());
        
        // Initialize recovery state
        *self.state.write() = Some(RecoveryState {
            pending_entries: pending.clone(),
            replayed_count: 0,
            failed_entries: Vec::new(),
            progress: 0.0,
        });
        
        // Group entries by transaction
        let transactions = self.group_by_transaction(&pending);
        
        let mut stats = RecoveryStats {
            total_entries: pending.len(),
            ..Default::default()
        };
        
        // Replay each transaction
        for (tx_id, entries) in transactions {
            match self.replay_transaction(tx_id, entries).await {
                Ok(count) => {
                    stats.replayed += count;
                    debug!("Replayed {} entries for transaction {}", count, tx_id);
                }
                Err(e) => {
                    warn!("Failed to replay transaction {}: {}", tx_id, e);
                    stats.failed += entries.len();
                    
                    // Update failed entries in state
                    if let Some(mut state) = self.state.write().as_mut() {
                        state.failed_entries.extend(entries);
                    }
                }
            }
            
            // Update progress
            if let Some(mut state) = self.state.write().as_mut() {
                state.progress = stats.replayed as f32 / stats.total_entries as f32;
            }
        }
        
        stats.duration_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Recovery complete: {} replayed, {} failed, {} skipped in {}ms",
            stats.replayed, stats.failed, stats.skipped, stats.duration_ms
        );
        
        // Clear recovery state
        *self.state.write() = None;
        
        Ok(stats)
    }
    
    /// Group entries by transaction ID
    fn group_by_transaction(&self, entries: &[WalEntry]) -> HashMap<Uuid, Vec<WalEntry>> {
        let mut transactions = HashMap::new();
        
        for entry in entries {
            transactions.entry(entry.tx_id)
                .or_insert_with(Vec::new)
                .push(entry.clone());
        }
        
        transactions
    }
    
    /// Replay a single transaction
    async fn replay_transaction(
        &self,
        tx_id: Uuid,
        entries: Vec<WalEntry>,
    ) -> BlockResult<usize> {
        debug!("Replaying transaction {} with {} entries", tx_id, entries.len());
        
        let mut replayed = 0;
        let callbacks = self.callbacks.read();
        
        for entry in entries {
            match self.replay_entry(&entry, &callbacks).await {
                Ok(_) => {
                    replayed += 1;
                    
                    // Update state
                    if let Some(mut state) = self.state.write().as_mut() {
                        state.replayed_count += 1;
                    }
                }
                Err(e) => {
                    error!("Failed to replay entry {:?}: {}", entry.tx_id, e);
                    
                    // Decide whether to continue or abort transaction
                    if self.should_abort_transaction(&e) {
                        return Err(e);
                    }
                }
            }
        }
        
        // Mark transaction as recovered
        self.wal.commit(tx_id).await?;
        
        Ok(replayed)
    }
    
    /// Replay a single WAL entry
    async fn replay_entry(
        &self,
        entry: &WalEntry,
        callbacks: &RecoveryCallbacks,
    ) -> BlockResult<()> {
        match &entry.operation {
            WalOperation::Write { key, value } => {
                if let Some(ref write_fn) = callbacks.write_fn {
                    write_fn(*key, value.clone())?;
                } else {
                    warn!("No write callback registered");
                }
            }
            WalOperation::Delete { key } => {
                if let Some(ref delete_fn) = callbacks.delete_fn {
                    delete_fn(*key)?;
                } else {
                    warn!("No delete callback registered");
                }
            }
            WalOperation::Update { key, value } => {
                if let Some(ref update_fn) = callbacks.update_fn {
                    update_fn(*key, value.clone())?;
                } else {
                    warn!("No update callback registered");
                }
            }
            WalOperation::Batch { operations } => {
                // Recursively replay batch operations
                for op in operations {
                    let batch_entry = WalEntry {
                        tx_id: entry.tx_id,
                        operation: op.clone(),
                        timestamp: entry.timestamp,
                        checksum: 0,
                    };
                    self.replay_entry(&batch_entry, callbacks).await?;
                }
            }
            WalOperation::Commit { .. } | WalOperation::Rollback { .. } => {
                // These are markers, no action needed
            }
        }
        
        Ok(())
    }
    
    /// Determine if transaction should be aborted on error
    fn should_abort_transaction(&self, error: &BlockError) -> bool {
        match error {
            BlockError::Timeout(_) => false, // Continue on timeout
            _ => true, // Abort on other errors
        }
    }
    
    /// Get current recovery state
    pub fn get_state(&self) -> Option<RecoveryState> {
        self.state.read().clone()
    }
    
    /// Validate recovered data
    pub async fn validate_recovery(&self) -> BlockResult<bool> {
        info!("Validating recovered data");
        
        // Check WAL for any remaining uncommitted entries
        let uncommitted = self.wal.get_uncommitted().await?;
        
        if !uncommitted.is_empty() {
            warn!("Found {} uncommitted entries after recovery", uncommitted.len());
            return Ok(false);
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supporting_module::storage::wal::WalConfig;
    
    #[tokio::test]
    async fn test_recovery_manager() {
        let wal = Arc::new(WriteAheadLog::new(WalConfig::default()).unwrap());
        let recovery = RecoveryManager::new(wal.clone());
        
        // Set callbacks
        let storage = Arc::new(RwLock::new(HashMap::<Uuid, Vec<u8>>::new()));
        let storage_clone = storage.clone();
        
        recovery.set_write_callback(move |key, value| {
            storage_clone.write().insert(key, value);
            Ok(())
        });
        
        // Add some entries to WAL
        let tx_id = Uuid::new_v4();
        let key = Uuid::new_v4();
        
        wal.append(WalEntry {
            tx_id,
            operation: WalOperation::Write {
                key,
                value: vec![1, 2, 3],
            },
            timestamp: chrono::Utc::now(),
            checksum: 0,
        }).await.unwrap();
        
        // Perform recovery
        let stats = recovery.recover().await.unwrap();
        
        assert_eq!(stats.replayed, 1);
        assert_eq!(stats.failed, 0);
        
        // Check data was recovered
        assert!(storage.read().contains_key(&key));
    }
}