//! Execution Manager
//! 
//! Manages pipeline execution with thread pools and resource control

use crate::packages::{Pipeline, PipelineOutput, PipelineError};
use std::sync::Arc;
use tokio::sync::{Semaphore, RwLock};
use dashmap::DashMap;
use std::time::Duration;

/// Execution manager with thread pool and resource control
pub struct ExecutionManager {
    thread_pool: Arc<ThreadPool>,
    resource_manager: Arc<ResourceManager>,
    execution_mode: RwLock<ExecutionMode>,
    metrics: Arc<ExecutionMetrics>,
}

impl ExecutionManager {
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            thread_pool: Arc::new(ThreadPool::new(config.num_threads)),
            resource_manager: Arc::new(ResourceManager::new(config.max_concurrent)),
            execution_mode: RwLock::new(config.mode),
            metrics: Arc::new(ExecutionMetrics::new()),
        }
    }
    
    /// Execute pipeline with resource management
    pub async fn execute(
        &self,
        pipeline: Arc<dyn Pipeline>,
        input: &str,
    ) -> Result<PipelineOutput, PipelineError> {
        // Acquire resource permit
        let _permit = self.resource_manager.acquire().await?;
        
        // Record start
        self.metrics.record_start();
        
        // Execute based on mode
        let mode = *self.execution_mode.read().await;
        let result = match mode {
            ExecutionMode::Synchronous => {
                pipeline.execute(input).await
            }
            ExecutionMode::Asynchronous => {
                self.execute_async(pipeline, input).await
            }
            ExecutionMode::Batched(size) => {
                self.execute_batched(pipeline, input, size).await
            }
            ExecutionMode::Streaming => {
                self.execute_streaming(pipeline, input).await
            }
        };
        
        // Record completion
        self.metrics.record_completion(result.is_ok());
        
        result
    }
    
    /// Execute asynchronously
    async fn execute_async(
        &self,
        pipeline: Arc<dyn Pipeline>,
        input: &str,
    ) -> Result<PipelineOutput, PipelineError> {
        let input = input.to_string();
        let handle = tokio::spawn(async move {
            pipeline.execute(&input).await
        });
        
        handle.await
            .map_err(|e| PipelineError::ExecutionFailed(e.to_string()))?
    }
    
    /// Execute in batch mode
    async fn execute_batched(
        &self,
        pipeline: Arc<dyn Pipeline>,
        input: &str,
        _batch_size: usize,
    ) -> Result<PipelineOutput, PipelineError> {
        // Simplified - just execute normally
        pipeline.execute(input).await
    }
    
    /// Execute in streaming mode
    async fn execute_streaming(
        &self,
        pipeline: Arc<dyn Pipeline>,
        input: &str,
    ) -> Result<PipelineOutput, PipelineError> {
        // Simplified - just execute normally
        pipeline.execute(input).await
    }
    
    /// Set execution mode
    pub async fn set_mode(&self, mode: ExecutionMode) {
        *self.execution_mode.write().await = mode;
    }
    
    /// Get execution metrics
    pub fn metrics(&self) -> Arc<ExecutionMetrics> {
        self.metrics.clone()
    }
    
    /// Execute multiple pipelines in parallel
    pub async fn execute_parallel(
        &self,
        pipelines: Vec<Arc<dyn Pipeline>>,
        input: &str,
    ) -> Vec<Result<PipelineOutput, PipelineError>> {
        let mut handles = Vec::new();
        
        for pipeline in pipelines {
            let input = input.to_string();
            let handle = tokio::spawn(async move {
                pipeline.execute(&input).await
            });
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(PipelineError::ExecutionFailed(e.to_string()))),
            }
        }
        
        results
    }
}

/// Execution mode
#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
    Batched(usize),
    Streaming,
}

/// Execution configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    pub num_threads: usize,
    pub max_concurrent: usize,
    pub mode: ExecutionMode,
    pub timeout: Duration,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            max_concurrent: 100,
            mode: ExecutionMode::Asynchronous,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Thread pool for execution
struct ThreadPool {
    workers: Vec<Worker>,
    sender: tokio::sync::mpsc::UnboundedSender<Job>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let receiver = Arc::new(tokio::sync::Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(id, receiver.clone()));
        }
        
        Self { workers, sender }
    }
    
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job).unwrap();
    }
}

/// Worker thread
struct Worker {
    id: usize,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Job>>>) -> Self {
        let handle = tokio::spawn(async move {
            loop {
                let job = {
                    let mut receiver = receiver.lock().await;
                    receiver.recv().await
                };
                
                match job {
                    Some(job) => {
                        tracing::debug!("Worker {} executing job", id);
                        job();
                    }
                    None => {
                        tracing::debug!("Worker {} shutting down", id);
                        break;
                    }
                }
            }
        });
        
        Self {
            id,
            handle: Some(handle),
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Resource manager
struct ResourceManager {
    semaphore: Arc<Semaphore>,
    active_executions: DashMap<uuid::Uuid, ExecutionInfo>,
}

impl ResourceManager {
    fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            active_executions: DashMap::new(),
        }
    }
    
    async fn acquire(&self) -> Result<ResourcePermit, PipelineError> {
        let permit = self.semaphore.clone().acquire_owned().await
            .map_err(|_| PipelineError::ExecutionFailed("Failed to acquire resource".into()))?;
        
        let id = uuid::Uuid::new_v4();
        self.active_executions.insert(id, ExecutionInfo {
            started_at: std::time::Instant::now(),
        });
        
        Ok(ResourcePermit {
            id,
            _permit: permit,
            manager: self as *const _ as *mut ResourceManager,
        })
    }
}

/// Resource permit
struct ResourcePermit {
    id: uuid::Uuid,
    _permit: tokio::sync::OwnedSemaphorePermit,
    manager: *mut ResourceManager,
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        unsafe {
            if let Some(manager) = self.manager.as_ref() {
                manager.active_executions.remove(&self.id);
            }
        }
    }
}

/// Execution info
#[derive(Debug)]
struct ExecutionInfo {
    started_at: std::time::Instant,
}

/// Execution metrics
pub struct ExecutionMetrics {
    total_executions: Arc<RwLock<u64>>,
    successful_executions: Arc<RwLock<u64>>,
    failed_executions: Arc<RwLock<u64>>,
    total_latency_ms: Arc<RwLock<f64>>,
}

impl ExecutionMetrics {
    fn new() -> Self {
        Self {
            total_executions: Arc::new(RwLock::new(0)),
            successful_executions: Arc::new(RwLock::new(0)),
            failed_executions: Arc::new(RwLock::new(0)),
            total_latency_ms: Arc::new(RwLock::new(0.0)),
        }
    }
    
    fn record_start(&self) {
        tokio::spawn({
            let total = self.total_executions.clone();
            async move {
                *total.write().await += 1;
            }
        });
    }
    
    fn record_completion(&self, success: bool) {
        tokio::spawn({
            let successful = self.successful_executions.clone();
            let failed = self.failed_executions.clone();
            async move {
                if success {
                    *successful.write().await += 1;
                } else {
                    *failed.write().await += 1;
                }
            }
        });
    }
    
    pub async fn get_stats(&self) -> ExecutionStats {
        ExecutionStats {
            total: *self.total_executions.read().await,
            successful: *self.successful_executions.read().await,
            failed: *self.failed_executions.read().await,
            avg_latency_ms: if *self.total_executions.read().await > 0 {
                *self.total_latency_ms.read().await / *self.total_executions.read().await as f64
            } else {
                0.0
            },
        }
    }
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total: u64,
    pub successful: u64,
    pub failed: u64,
    pub avg_latency_ms: f64,
}