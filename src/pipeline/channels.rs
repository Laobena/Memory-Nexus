//! Route-Specific Channel Strategies for Optimal Latency
//! 
//! Based on production patterns from Discord and Cloudflare, different execution
//! routes require different channel strategies for achieving target latencies.
//! 
//! - CacheOnly (2ms): Lock-free ArrayQueue for minimal overhead
//! - SmartRouting (15ms): Tokio mpsc with backpressure
//! - FullPipeline (40-45ms): Adaptive batching for throughput

use crate::core::{Result, NexusError, ZeroCopyMessage};
use crossbeam::queue::ArrayQueue;
use tokio::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

// ===== CACHE-ONLY CHANNEL (2ms target) =====

/// Lock-free channel for CacheOnly route (2ms target)
/// Uses bounded ArrayQueue for zero allocation and minimal overhead
pub struct CacheOnlyChannel {
    queue: Arc<ArrayQueue<ZeroCopyMessage>>,
    capacity: usize,
}

impl CacheOnlyChannel {
    /// Create a new CacheOnly channel with fixed capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
            capacity,
        }
    }
    
    /// Try to send a message (non-blocking)
    #[inline(always)]
    pub fn try_send(&self, msg: ZeroCopyMessage) -> Result<()> {
        self.queue.push(msg)
            .map_err(|_| NexusError::QueueFull("CacheOnly channel full".to_string()))
    }
    
    /// Try to receive a message (non-blocking)
    #[inline(always)]
    pub fn try_recv(&self) -> Option<ZeroCopyMessage> {
        self.queue.pop()
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    /// Get current length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

// ===== SMART ROUTING CHANNEL (15ms target) =====

/// Tokio mpsc channel for SmartRouting (15ms target)
/// Provides backpressure and async operations
pub struct SmartRoutingChannel {
    sender: mpsc::Sender<ZeroCopyMessage>,
    receiver: Arc<Mutex<mpsc::Receiver<ZeroCopyMessage>>>,
    buffer_size: usize,
}

impl SmartRoutingChannel {
    /// Create a new SmartRouting channel with backpressure
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer_size);
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
            buffer_size,
        }
    }
    
    /// Send a message with backpressure
    pub async fn send(&self, msg: ZeroCopyMessage) -> Result<()> {
        self.sender.send(msg).await
            .map_err(|_| NexusError::Pipeline("SmartRouting channel closed".to_string()))
    }
    
    /// Try to send without waiting
    pub fn try_send(&self, msg: ZeroCopyMessage) -> Result<()> {
        self.sender.try_send(msg)
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => 
                    NexusError::QueueFull("SmartRouting channel full".to_string()),
                mpsc::error::TrySendError::Closed(_) => 
                    NexusError::Pipeline("SmartRouting channel closed".to_string()),
            })
    }
    
    /// Receive a message
    pub async fn recv(&self) -> Option<ZeroCopyMessage> {
        let mut receiver = self.receiver.lock();
        receiver.recv().await
    }
    
    /// Clone the sender for multiple producers
    pub fn clone_sender(&self) -> mpsc::Sender<ZeroCopyMessage> {
        self.sender.clone()
    }
}

// ===== ADAPTIVE BATCHER (40-45ms target) =====

/// Adaptive batching for FullPipeline/MaximumIntelligence routes
/// Optimizes throughput by batching messages based on time and size
pub struct AdaptiveBatcher {
    buffer: Arc<Mutex<Vec<ZeroCopyMessage>>>,
    min_batch_size: usize,
    max_batch_size: usize,
    max_wait_time: Duration,
    last_flush: Arc<Mutex<Instant>>,
    throughput_monitor: Arc<ThroughputMonitor>,
}

impl AdaptiveBatcher {
    /// Create a new adaptive batcher
    pub fn new(min_batch: usize, max_batch: usize, max_wait: Duration) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::with_capacity(max_batch))),
            min_batch_size: min_batch,
            max_batch_size: max_batch,
            max_wait_time: max_wait,
            last_flush: Arc::new(Mutex::new(Instant::now())),
            throughput_monitor: Arc::new(ThroughputMonitor::new()),
        }
    }
    
    /// Add a message to the batch
    pub fn add(&self, msg: ZeroCopyMessage) -> Option<Vec<ZeroCopyMessage>> {
        let mut buffer = self.buffer.lock();
        buffer.push(msg);
        
        // Check if we should flush
        if self.should_flush(&buffer) {
            let batch = std::mem::replace(&mut *buffer, Vec::with_capacity(self.max_batch_size));
            *self.last_flush.lock() = Instant::now();
            self.throughput_monitor.record_batch(batch.len());
            Some(batch)
        } else {
            None
        }
    }
    
    /// Force flush the current batch
    pub fn flush(&self) -> Vec<ZeroCopyMessage> {
        let mut buffer = self.buffer.lock();
        let batch = std::mem::replace(&mut *buffer, Vec::with_capacity(self.max_batch_size));
        *self.last_flush.lock() = Instant::now();
        if !batch.is_empty() {
            self.throughput_monitor.record_batch(batch.len());
        }
        batch
    }
    
    /// Check if we should flush based on size and time
    fn should_flush(&self, buffer: &[ZeroCopyMessage]) -> bool {
        // Flush if buffer is full
        if buffer.len() >= self.max_batch_size {
            return true;
        }
        
        // Flush if we have enough messages and time has passed
        if buffer.len() >= self.min_batch_size {
            let elapsed = self.last_flush.lock().elapsed();
            if elapsed >= self.max_wait_time / 2 {
                return true;
            }
        }
        
        // Force flush if max wait time exceeded
        self.last_flush.lock().elapsed() >= self.max_wait_time && !buffer.is_empty()
    }
    
    /// Adjust batch sizes based on throughput
    pub fn adapt_batch_size(&mut self) {
        let throughput = self.throughput_monitor.get_throughput();
        
        // Increase batch size if throughput is high
        if throughput > 1000.0 && self.max_batch_size < 256 {
            self.max_batch_size = (self.max_batch_size * 3) / 2;
            self.min_batch_size = self.max_batch_size / 4;
        }
        // Decrease batch size if throughput is low
        else if throughput < 100.0 && self.max_batch_size > 32 {
            self.max_batch_size = (self.max_batch_size * 2) / 3;
            self.min_batch_size = self.max_batch_size / 4;
        }
    }
}

// ===== THROUGHPUT MONITOR =====

/// Monitor throughput for adaptive batching
struct ThroughputMonitor {
    window: Arc<Mutex<Vec<(Instant, usize)>>>,
    window_duration: Duration,
}

impl ThroughputMonitor {
    fn new() -> Self {
        Self {
            window: Arc::new(Mutex::new(Vec::new())),
            window_duration: Duration::from_secs(1),
        }
    }
    
    fn record_batch(&self, size: usize) {
        let now = Instant::now();
        let mut window = self.window.lock();
        
        // Remove old entries
        let cutoff = now - self.window_duration;
        window.retain(|(time, _)| *time > cutoff);
        
        // Add new entry
        window.push((now, size));
    }
    
    fn get_throughput(&self) -> f64 {
        let window = self.window.lock();
        if window.is_empty() {
            return 0.0;
        }
        
        let total: usize = window.iter().map(|(_, size)| size).sum();
        let duration = window.last().unwrap().0 - window.first().unwrap().0;
        
        if duration.as_secs_f64() > 0.0 {
            total as f64 / duration.as_secs_f64()
        } else {
            total as f64
        }
    }
}

// ===== CHANNEL FACTORY =====

/// Factory for creating route-specific channels
pub struct ChannelFactory;

impl ChannelFactory {
    /// Create a channel optimized for CacheOnly route (2ms target)
    pub fn create_cache_only(capacity: usize) -> CacheOnlyChannel {
        CacheOnlyChannel::new(capacity.max(1000))
    }
    
    /// Create a channel optimized for SmartRouting (15ms target)
    pub fn create_smart_routing(buffer_size: usize) -> SmartRoutingChannel {
        SmartRoutingChannel::new(buffer_size.max(100))
    }
    
    /// Create an adaptive batcher for FullPipeline (40ms target)
    pub fn create_full_pipeline() -> AdaptiveBatcher {
        AdaptiveBatcher::new(
            8,   // min batch size
            64,  // max batch size  
            Duration::from_millis(20), // max wait time
        )
    }
    
    /// Create an adaptive batcher for MaximumIntelligence (45ms target)
    pub fn create_max_intelligence() -> AdaptiveBatcher {
        AdaptiveBatcher::new(
            16,  // min batch size (larger for better throughput)
            128, // max batch size
            Duration::from_millis(25), // max wait time
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use crate::core::{PipelineStage, MessagePayload, QueryPayload};
    
    #[test]
    fn test_cache_only_channel() {
        let channel = ChannelFactory::create_cache_only(10);
        
        let msg = ZeroCopyMessage {
            id: Uuid::new_v4(),
            stage: PipelineStage::Router,
            payload: MessagePayload::Error("test".to_string()),
            timestamp: 0,
            trace_id: None,
        };
        
        // Test send and receive
        channel.try_send(msg.clone()).unwrap();
        assert_eq!(channel.len(), 1);
        
        let received = channel.try_recv().unwrap();
        assert_eq!(received.id, msg.id);
        assert!(channel.is_empty());
    }
    
    #[tokio::test]
    async fn test_smart_routing_channel() {
        let channel = ChannelFactory::create_smart_routing(10);
        
        let msg = ZeroCopyMessage {
            id: Uuid::new_v4(),
            stage: PipelineStage::Search,
            payload: MessagePayload::Error("test".to_string()),
            timestamp: 0,
            trace_id: None,
        };
        
        // Test async send and receive
        channel.send(msg.clone()).await.unwrap();
        let received = channel.recv().await.unwrap();
        assert_eq!(received.id, msg.id);
    }
    
    #[test]
    fn test_adaptive_batcher() {
        let mut batcher = ChannelFactory::create_full_pipeline();
        
        // Add messages below min batch size
        for i in 0..7 {
            let msg = ZeroCopyMessage {
                id: Uuid::new_v4(),
                stage: PipelineStage::Fusion,
                payload: MessagePayload::Error(format!("test {}", i)),
                timestamp: i,
                trace_id: None,
            };
            assert!(batcher.add(msg).is_none());
        }
        
        // Adding the 8th message should trigger a batch
        let msg = ZeroCopyMessage {
            id: Uuid::new_v4(),
            stage: PipelineStage::Fusion,
            payload: MessagePayload::Error("trigger".to_string()),
            timestamp: 8,
            trace_id: None,
        };
        
        let batch = batcher.add(msg).expect("Should return batch");
        assert_eq!(batch.len(), 8);
    }
}