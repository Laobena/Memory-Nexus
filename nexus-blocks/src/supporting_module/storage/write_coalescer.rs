//! Write coalescing for I/O optimization

use crate::core::{BlockError, BlockResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, info};
use uuid::Uuid;

/// Write strategy for coalescing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WriteStrategy {
    /// Write immediately
    WriteThrough,
    /// Buffer writes and batch them
    WriteBack,
    /// Adaptive based on load
    Adaptive,
}

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum time to wait before flushing
    pub max_wait_time: Duration,
    /// Enable compression for batches
    pub compress_batch: bool,
    /// Minimum batch size for compression
    pub compression_threshold: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            max_wait_time: Duration::from_millis(100),
            compress_batch: true,
            compression_threshold: 10,
        }
    }
}

/// Write request
#[derive(Debug, Clone)]
struct WriteRequest {
    key: Uuid,
    value: Vec<u8>,
    timestamp: Instant,
}

/// Write coalescer for batching writes
pub struct WriteCoalescer {
    /// Configuration
    config: Arc<BatchConfig>,
    
    /// Write strategy
    strategy: Arc<RwLock<WriteStrategy>>,
    
    /// Pending writes buffer
    pending: Arc<RwLock<HashMap<Uuid, WriteRequest>>>,
    
    /// Write channel
    write_tx: mpsc::UnboundedSender<WriteRequest>,
    
    /// Statistics
    stats: Arc<RwLock<CoalescerStats>>,
}

/// Coalescer statistics
#[derive(Debug, Default)]
struct CoalescerStats {
    total_writes: u64,
    batches_flushed: u64,
    bytes_written: u64,
    bytes_saved: u64,
    average_batch_size: f64,
}

impl WriteCoalescer {
    /// Create new write coalescer
    pub fn new(config: BatchConfig, strategy: WriteStrategy) -> (Self, mpsc::UnboundedReceiver<Vec<WriteRequest>>) {
        let (write_tx, write_rx) = mpsc::unbounded_channel();
        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        
        let coalescer = Self {
            config: Arc::new(config.clone()),
            strategy: Arc::new(RwLock::new(strategy)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            write_tx,
            stats: Arc::new(RwLock::new(CoalescerStats::default())),
        };
        
        // Start background flusher
        let coalescer_clone = coalescer.clone();
        tokio::spawn(async move {
            coalescer_clone.run_flusher(write_rx, batch_tx).await;
        });
        
        (coalescer, batch_rx)
    }
    
    /// Submit write request
    pub async fn write(&self, key: Uuid, value: Vec<u8>) -> BlockResult<()> {
        let request = WriteRequest {
            key,
            value: value.clone(),
            timestamp: Instant::now(),
        };
        
        match *self.strategy.read() {
            WriteStrategy::WriteThrough => {
                // Send immediately
                self.write_tx.send(request)
                    .map_err(|_| BlockError::Unknown("Write channel closed".into()))?;
            }
            WriteStrategy::WriteBack => {
                // Add to pending buffer
                self.pending.write().insert(key, request);
                
                // Check if we should flush
                if self.should_flush() {
                    self.flush().await?;
                }
            }
            WriteStrategy::Adaptive => {
                // Decide based on current load
                if self.pending.read().len() < self.config.max_batch_size / 2 {
                    // Low load, buffer it
                    self.pending.write().insert(key, request);
                } else {
                    // High load, write through
                    self.write_tx.send(request)
                        .map_err(|_| BlockError::Unknown("Write channel closed".into()))?;
                }
            }
        }
        
        self.stats.write().total_writes += 1;
        Ok(())
    }
    
    /// Check if we should flush
    fn should_flush(&self) -> bool {
        let pending = self.pending.read();
        
        // Flush if batch is full
        if pending.len() >= self.config.max_batch_size {
            return true;
        }
        
        // Flush if oldest entry is too old
        if let Some(oldest) = pending.values().min_by_key(|r| r.timestamp) {
            if oldest.timestamp.elapsed() > self.config.max_wait_time {
                return true;
            }
        }
        
        false
    }
    
    /// Flush pending writes
    pub async fn flush(&self) -> BlockResult<()> {
        let mut pending = self.pending.write();
        
        if pending.is_empty() {
            return Ok(());
        }
        
        let batch: Vec<WriteRequest> = pending.drain().map(|(_, v)| v).collect();
        let batch_size = batch.len();
        
        // Send batch for processing
        for request in batch {
            self.write_tx.send(request)
                .map_err(|_| BlockError::Unknown("Write channel closed".into()))?;
        }
        
        // Update stats
        let mut stats = self.stats.write();
        stats.batches_flushed += 1;
        stats.average_batch_size = 
            (stats.average_batch_size * (stats.batches_flushed - 1) as f64 + batch_size as f64) 
            / stats.batches_flushed as f64;
        
        debug!("Flushed batch of {} writes", batch_size);
        Ok(())
    }
    
    /// Run background flusher
    async fn run_flusher(
        &self,
        mut write_rx: mpsc::UnboundedReceiver<WriteRequest>,
        batch_tx: mpsc::UnboundedSender<Vec<WriteRequest>>,
    ) {
        let mut interval = interval(self.config.max_wait_time);
        let mut current_batch = Vec::new();
        
        loop {
            tokio::select! {
                Some(request) = write_rx.recv() => {
                    current_batch.push(request);
                    
                    if current_batch.len() >= self.config.max_batch_size {
                        self.send_batch(&batch_tx, &mut current_batch).await;
                    }
                }
                _ = interval.tick() => {
                    if !current_batch.is_empty() {
                        self.send_batch(&batch_tx, &mut current_batch).await;
                    }
                }
            }
        }
    }
    
    /// Send batch for processing
    async fn send_batch(
        &self,
        batch_tx: &mpsc::UnboundedSender<Vec<WriteRequest>>,
        batch: &mut Vec<WriteRequest>,
    ) {
        if batch.is_empty() {
            return;
        }
        
        let batch_to_send = std::mem::take(batch);
        
        // Optionally compress batch
        let final_batch = if self.config.compress_batch && 
                            batch_to_send.len() >= self.config.compression_threshold {
            self.compress_batch(batch_to_send).await
        } else {
            batch_to_send
        };
        
        if let Err(e) = batch_tx.send(final_batch) {
            tracing::error!("Failed to send batch: {}", e);
        }
    }
    
    /// Compress batch of writes
    async fn compress_batch(&self, batch: Vec<WriteRequest>) -> Vec<WriteRequest> {
        // Group by key to coalesce updates
        let mut coalesced: HashMap<Uuid, WriteRequest> = HashMap::new();
        
        for request in batch {
            coalesced.insert(request.key, request);
        }
        
        let original_size = batch.len();
        let coalesced_size = coalesced.len();
        
        if coalesced_size < original_size {
            let saved = original_size - coalesced_size;
            self.stats.write().bytes_saved += saved as u64;
            
            info!("Coalesced {} writes into {} (saved {})", 
                  original_size, coalesced_size, saved);
        }
        
        coalesced.into_values().collect()
    }
    
    /// Update write strategy
    pub fn set_strategy(&self, strategy: WriteStrategy) {
        *self.strategy.write() = strategy;
        debug!("Updated write strategy to {:?}", strategy);
    }
    
    /// Get statistics
    pub fn stats(&self) -> WriteCoalescerStats {
        let stats = self.stats.read();
        
        WriteCoalescerStats {
            total_writes: stats.total_writes,
            batches_flushed: stats.batches_flushed,
            bytes_written: stats.bytes_written,
            bytes_saved: stats.bytes_saved,
            average_batch_size: stats.average_batch_size,
            io_reduction_percent: if stats.bytes_written > 0 {
                (stats.bytes_saved as f64 / (stats.bytes_written + stats.bytes_saved) as f64) * 100.0
            } else {
                0.0
            },
        }
    }
}

impl Clone for WriteCoalescer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            strategy: self.strategy.clone(),
            pending: self.pending.clone(),
            write_tx: self.write_tx.clone(),
            stats: self.stats.clone(),
        }
    }
}

/// Write coalescer statistics
#[derive(Debug)]
pub struct WriteCoalescerStats {
    pub total_writes: u64,
    pub batches_flushed: u64,
    pub bytes_written: u64,
    pub bytes_saved: u64,
    pub average_batch_size: f64,
    pub io_reduction_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_write_coalescing() {
        let (coalescer, mut batch_rx) = WriteCoalescer::new(
            BatchConfig {
                max_batch_size: 3,
                ..Default::default()
            },
            WriteStrategy::WriteBack,
        );
        
        // Write multiple values for same key
        let key = Uuid::new_v4();
        coalescer.write(key, vec![1]).await.unwrap();
        coalescer.write(key, vec![2]).await.unwrap();
        coalescer.write(key, vec![3]).await.unwrap();
        
        // Should trigger flush
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should receive coalesced batch
        if let Ok(batch) = batch_rx.try_recv() {
            // Should have coalesced to single write
            assert_eq!(batch.len(), 1);
            assert_eq!(batch[0].value, vec![3]); // Last value
        }
    }
}