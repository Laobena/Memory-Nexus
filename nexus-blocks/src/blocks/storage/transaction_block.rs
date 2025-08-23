//! Transaction management with ACID guarantees

use crate::core::{BlockError, BlockResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};
use uuid::Uuid;

/// Transaction ID type
pub type TransactionId = Uuid;

/// Isolation levels for transactions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        IsolationLevel::ReadCommitted
    }
}

/// Transaction state
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionState {
    Active,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
}

/// Transaction metadata
#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: TransactionId,
    pub state: TransactionState,
    pub isolation_level: IsolationLevel,
    pub started_at: Instant,
    pub operations: Vec<TransactionOp>,
    pub locks: Vec<LockInfo>,
}

/// Transaction operation
#[derive(Debug, Clone)]
pub enum TransactionOp {
    Read { key: Uuid },
    Write { key: Uuid, value: Vec<u8> },
    Delete { key: Uuid },
}

/// Lock information
#[derive(Debug, Clone)]
pub struct LockInfo {
    pub key: Uuid,
    pub lock_type: LockType,
    pub acquired_at: Instant,
}

/// Lock types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LockType {
    Shared,
    Exclusive,
}

/// Transaction manager for coordinating transactions
pub struct TransactionManager {
    /// Active transactions
    transactions: Arc<RwLock<HashMap<TransactionId, Transaction>>>,
    
    /// Lock table for concurrency control
    lock_table: Arc<RwLock<HashMap<Uuid, Vec<(TransactionId, LockType)>>>>,
    
    /// Transaction log
    tx_log: Arc<RwLock<Vec<TransactionLogEntry>>>,
    
    /// Deadlock detector
    deadlock_detector: Arc<DeadlockDetector>,
    
    /// Configuration
    config: Arc<TransactionConfig>,
}

/// Transaction configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    pub default_isolation: IsolationLevel,
    pub max_transaction_duration: Duration,
    pub deadlock_detection_interval: Duration,
    pub enable_mvcc: bool, // Multi-version concurrency control
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            default_isolation: IsolationLevel::ReadCommitted,
            max_transaction_duration: Duration::from_secs(60),
            deadlock_detection_interval: Duration::from_secs(1),
            enable_mvcc: true,
        }
    }
}

/// Transaction log entry
#[derive(Debug, Clone)]
struct TransactionLogEntry {
    tx_id: TransactionId,
    timestamp: Instant,
    event: TransactionEvent,
}

/// Transaction events for logging
#[derive(Debug, Clone)]
enum TransactionEvent {
    Started,
    Committed,
    Aborted,
    DeadlockDetected,
}

/// Deadlock detector
struct DeadlockDetector {
    /// Wait-for graph: tx1 -> tx2 means tx1 is waiting for tx2
    wait_graph: Arc<RwLock<HashMap<TransactionId, Vec<TransactionId>>>>,
}

impl TransactionManager {
    /// Create new transaction manager
    pub fn new(config: TransactionConfig) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            lock_table: Arc::new(RwLock::new(HashMap::new())),
            tx_log: Arc::new(RwLock::new(Vec::new())),
            deadlock_detector: Arc::new(DeadlockDetector {
                wait_graph: Arc::new(RwLock::new(HashMap::new())),
            }),
            config: Arc::new(config),
        }
    }
    
    /// Begin new transaction
    pub async fn begin(&self) -> BlockResult<TransactionId> {
        self.begin_with_isolation(self.config.default_isolation).await
    }
    
    /// Begin transaction with specific isolation level
    pub async fn begin_with_isolation(
        &self,
        isolation: IsolationLevel,
    ) -> BlockResult<TransactionId> {
        let tx_id = Uuid::new_v4();
        
        let transaction = Transaction {
            id: tx_id,
            state: TransactionState::Active,
            isolation_level: isolation,
            started_at: Instant::now(),
            operations: Vec::new(),
            locks: Vec::new(),
        };
        
        self.transactions.write().insert(tx_id, transaction);
        
        self.log_event(tx_id, TransactionEvent::Started);
        
        debug!("Started transaction: {} with isolation {:?}", tx_id, isolation);
        Ok(tx_id)
    }
    
    /// Acquire lock for transaction
    pub async fn acquire_lock(
        &self,
        tx_id: TransactionId,
        key: Uuid,
        lock_type: LockType,
    ) -> BlockResult<()> {
        // Check if transaction exists
        let mut transactions = self.transactions.write();
        let tx = transactions.get_mut(&tx_id)
            .ok_or_else(|| BlockError::Unknown(format!("Transaction {} not found", tx_id)))?;
        
        // Check for conflicts
        let mut lock_table = self.lock_table.write();
        let existing_locks = lock_table.entry(key).or_insert_with(Vec::new);
        
        // Check compatibility
        for (other_tx_id, other_lock) in existing_locks.iter() {
            if *other_tx_id != tx_id {
                if !self.locks_compatible(*other_lock, lock_type) {
                    // Potential deadlock - add to wait graph
                    self.deadlock_detector.add_wait(tx_id, *other_tx_id);
                    
                    // Check for deadlock
                    if self.deadlock_detector.has_cycle() {
                        warn!("Deadlock detected for transaction: {}", tx_id);
                        self.log_event(tx_id, TransactionEvent::DeadlockDetected);
                        return Err(BlockError::Unknown("Deadlock detected".into()));
                    }
                    
                    // Wait or fail based on isolation level
                    match tx.isolation_level {
                        IsolationLevel::ReadUncommitted => {
                            // Allow dirty reads
                            if lock_type == LockType::Shared {
                                // Proceed anyway
                            } else {
                                return Err(BlockError::Unknown("Lock conflict".into()));
                            }
                        }
                        _ => {
                            return Err(BlockError::Unknown("Lock conflict".into()));
                        }
                    }
                }
            }
        }
        
        // Acquire lock
        existing_locks.push((tx_id, lock_type));
        
        tx.locks.push(LockInfo {
            key,
            lock_type,
            acquired_at: Instant::now(),
        });
        
        debug!("Transaction {} acquired {:?} lock on key {}", tx_id, lock_type, key);
        Ok(())
    }
    
    /// Release locks for transaction
    pub async fn release_locks(&self, tx_id: TransactionId) -> BlockResult<()> {
        let mut lock_table = self.lock_table.write();
        
        // Find and remove all locks held by this transaction
        let mut released = 0;
        lock_table.retain(|_key, locks| {
            locks.retain(|(id, _)| {
                if *id == tx_id {
                    released += 1;
                    false
                } else {
                    true
                }
            });
            !locks.is_empty()
        });
        
        // Clear from wait graph
        self.deadlock_detector.remove_transaction(tx_id);
        
        debug!("Released {} locks for transaction {}", released, tx_id);
        Ok(())
    }
    
    /// Prepare transaction for commit (2PC)
    pub async fn prepare(&self, tx_id: TransactionId) -> BlockResult<()> {
        let mut transactions = self.transactions.write();
        let tx = transactions.get_mut(&tx_id)
            .ok_or_else(|| BlockError::Unknown(format!("Transaction {} not found", tx_id)))?;
        
        if tx.state != TransactionState::Active {
            return Err(BlockError::Unknown(format!(
                "Transaction {} not in active state",
                tx_id
            )));
        }
        
        tx.state = TransactionState::Preparing;
        
        // Validate transaction
        if tx.started_at.elapsed() > self.config.max_transaction_duration {
            tx.state = TransactionState::Aborting;
            return Err(BlockError::Timeout(self.config.max_transaction_duration));
        }
        
        tx.state = TransactionState::Prepared;
        debug!("Transaction {} prepared for commit", tx_id);
        Ok(())
    }
    
    /// Commit transaction
    pub async fn commit(&self, tx_id: TransactionId) -> BlockResult<()> {
        // Prepare first
        self.prepare(tx_id).await?;
        
        let mut transactions = self.transactions.write();
        let tx = transactions.get_mut(&tx_id)
            .ok_or_else(|| BlockError::Unknown(format!("Transaction {} not found", tx_id)))?;
        
        tx.state = TransactionState::Committing;
        
        // Release locks
        drop(transactions);
        self.release_locks(tx_id).await?;
        
        // Mark as committed
        let mut transactions = self.transactions.write();
        if let Some(tx) = transactions.get_mut(&tx_id) {
            tx.state = TransactionState::Committed;
        }
        
        self.log_event(tx_id, TransactionEvent::Committed);
        
        debug!("Transaction {} committed", tx_id);
        Ok(())
    }
    
    /// Rollback transaction
    pub async fn rollback(&self, tx_id: TransactionId) -> BlockResult<()> {
        let mut transactions = self.transactions.write();
        let tx = transactions.get_mut(&tx_id)
            .ok_or_else(|| BlockError::Unknown(format!("Transaction {} not found", tx_id)))?;
        
        tx.state = TransactionState::Aborting;
        
        // Release locks
        drop(transactions);
        self.release_locks(tx_id).await?;
        
        // Mark as aborted
        let mut transactions = self.transactions.write();
        if let Some(tx) = transactions.get_mut(&tx_id) {
            tx.state = TransactionState::Aborted;
        }
        
        self.log_event(tx_id, TransactionEvent::Aborted);
        
        warn!("Transaction {} rolled back", tx_id);
        Ok(())
    }
    
    /// Check if two locks are compatible
    fn locks_compatible(&self, lock1: LockType, lock2: LockType) -> bool {
        match (lock1, lock2) {
            (LockType::Shared, LockType::Shared) => true,
            _ => false,
        }
    }
    
    /// Log transaction event
    fn log_event(&self, tx_id: TransactionId, event: TransactionEvent) {
        let entry = TransactionLogEntry {
            tx_id,
            timestamp: Instant::now(),
            event,
        };
        
        self.tx_log.write().push(entry);
    }
    
    /// Get transaction state
    pub fn get_state(&self, tx_id: TransactionId) -> Option<TransactionState> {
        self.transactions.read().get(&tx_id).map(|tx| tx.state.clone())
    }
    
    /// Clean up old transactions
    pub async fn cleanup(&self) {
        let now = Instant::now();
        let max_duration = self.config.max_transaction_duration;
        
        let mut to_rollback = Vec::new();
        
        {
            let transactions = self.transactions.read();
            for (tx_id, tx) in transactions.iter() {
                if tx.started_at.elapsed() > max_duration * 2 {
                    if tx.state == TransactionState::Committed || tx.state == TransactionState::Aborted {
                        // Safe to remove
                    } else {
                        // Need to rollback
                        to_rollback.push(*tx_id);
                    }
                }
            }
        }
        
        // Rollback stale transactions
        for tx_id in to_rollback {
            let _ = self.rollback(tx_id).await;
        }
        
        // Remove completed transactions
        self.transactions.write().retain(|_, tx| {
            tx.state != TransactionState::Committed && tx.state != TransactionState::Aborted
        });
    }
}

impl DeadlockDetector {
    /// Add wait dependency
    fn add_wait(&self, waiting: TransactionId, holding: TransactionId) {
        let mut graph = self.wait_graph.write();
        graph.entry(waiting).or_insert_with(Vec::new).push(holding);
    }
    
    /// Remove transaction from wait graph
    fn remove_transaction(&self, tx_id: TransactionId) {
        let mut graph = self.wait_graph.write();
        graph.remove(&tx_id);
        
        // Remove from other transactions' wait lists
        for (_, waits) in graph.iter_mut() {
            waits.retain(|&id| id != tx_id);
        }
    }
    
    /// Check for cycle in wait graph (deadlock)
    fn has_cycle(&self) -> bool {
        let graph = self.wait_graph.read();
        
        // Simple cycle detection using DFS
        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();
        
        for tx_id in graph.keys() {
            if self.has_cycle_dfs(&graph, *tx_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        
        false
    }
    
    fn has_cycle_dfs(
        &self,
        graph: &HashMap<TransactionId, Vec<TransactionId>>,
        node: TransactionId,
        visited: &mut HashMap<TransactionId, bool>,
        rec_stack: &mut HashMap<TransactionId, bool>,
    ) -> bool {
        if *rec_stack.get(&node).unwrap_or(&false) {
            return true;
        }
        
        if *visited.get(&node).unwrap_or(&false) {
            return false;
        }
        
        visited.insert(node, true);
        rec_stack.insert(node, true);
        
        if let Some(neighbors) = graph.get(&node) {
            for neighbor in neighbors {
                if self.has_cycle_dfs(graph, *neighbor, visited, rec_stack) {
                    return true;
                }
            }
        }
        
        rec_stack.insert(node, false);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_transaction_lifecycle() {
        let manager = TransactionManager::new(TransactionConfig::default());
        
        // Begin transaction
        let tx_id = manager.begin().await.unwrap();
        assert_eq!(manager.get_state(tx_id), Some(TransactionState::Active));
        
        // Acquire lock
        let key = Uuid::new_v4();
        manager.acquire_lock(tx_id, key, LockType::Exclusive).await.unwrap();
        
        // Commit
        manager.commit(tx_id).await.unwrap();
        assert_eq!(manager.get_state(tx_id), Some(TransactionState::Committed));
    }
}