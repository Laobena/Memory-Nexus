//! Optimized Tokio runtime configuration
//! 
//! Configures Tokio with 512 max blocking threads and optimal worker settings
//! based on Discord and Cloudflare production patterns.

use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use tracing::{info, warn};

/// Create optimized Tokio runtime
pub fn create_optimized_runtime() -> Result<Runtime, std::io::Error> {
    let num_cpus = num_cpus::get();
    
    // Calculate optimal worker threads
    // Discord uses 2x CPU cores for I/O heavy workloads
    let worker_threads = if is_io_heavy() {
        (num_cpus * 2).min(128) // Cap at 128 workers
    } else {
        num_cpus
    };
    
    // Max blocking threads - critical for preventing thread starvation
    let max_blocking_threads = 512;
    
    // Thread keep-alive duration
    let thread_keep_alive = Duration::from_secs(60);
    
    info!(
        worker_threads,
        max_blocking_threads,
        num_cpus,
        "Creating optimized Tokio runtime"
    );
    
    let mut builder = Builder::new_multi_thread();
    
    builder
        .worker_threads(worker_threads)
        .max_blocking_threads(max_blocking_threads)
        .thread_keep_alive(thread_keep_alive)
        .thread_name("nexus-worker")
        .enable_all();
    
    // Set thread stack size for deep async call stacks
    builder.thread_stack_size(4 * 1024 * 1024); // 4MB stack
    
    // Enable time and I/O drivers
    builder.enable_time();
    builder.enable_io();
    
    // Custom panic handler for worker threads
    builder.on_thread_start(|| {
        // Thread-local initialization
        #[cfg(feature = "jemalloc")]
        {
            // Initialize thread-local jemalloc cache
            let _ = tikv_jemallocator::Jemalloc;
        }
    });
    
    builder.on_thread_stop(|| {
        // Thread cleanup
        tracing::debug!("Worker thread stopping");
    });
    
    // Build runtime
    let runtime = builder.build()?;
    
    // Spawn background tasks
    spawn_background_tasks(&runtime);
    
    Ok(runtime)
}

/// Check if workload is I/O heavy (heuristic)
fn is_io_heavy() -> bool {
    // Check environment variable
    std::env::var("NEXUS_IO_HEAVY")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true) // Default to I/O heavy
}

/// Spawn background maintenance tasks
fn spawn_background_tasks(runtime: &Runtime) {
    // Allocator stats reporter
    runtime.spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            crate::core::allocator::report_allocator_stats();
        }
    });
    
    // Health check task
    runtime.spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            // Health checks would go here
            tracing::trace!("Background health check");
        }
    });
}

/// Runtime metrics collector
pub struct RuntimeMetrics {
    runtime: tokio::runtime::Handle,
}

impl RuntimeMetrics {
    pub fn new(runtime: tokio::runtime::Handle) -> Self {
        Self { runtime }
    }
    
    pub fn collect(&self) -> RuntimeStats {
        let metrics = self.runtime.metrics();
        
        RuntimeStats {
            workers_count: metrics.num_workers(),
            blocking_threads_count: metrics.num_workers(), // num_blocking_threads not available
            active_tasks_count: metrics.num_alive_tasks(),
            injection_queue_depth: metrics.global_queue_depth(),
            worker_local_queue_depth: 0, // worker_local_queue_depth not available per-worker
            blocking_queue_depth: 0, // blocking_queue_depth not available  
            total_park_count: (0..metrics.num_workers())
                .map(|i| metrics.worker_park_count(i))
                .sum(),
            total_busy_duration_us: (0..metrics.num_workers())
                .map(|i| metrics.worker_total_busy_duration(i).as_micros() as u64)
                .sum(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub workers_count: usize,
    pub blocking_threads_count: usize,
    pub active_tasks_count: usize,
    pub injection_queue_depth: usize,
    pub worker_local_queue_depth: usize,
    pub blocking_queue_depth: usize,
    pub total_park_count: u64,
    pub total_busy_duration_us: u64,
}

/// Custom task spawner with priorities
pub struct PrioritySpawner {
    high_priority: tokio::sync::mpsc::UnboundedSender<BoxedFuture>,
    normal_priority: tokio::sync::mpsc::UnboundedSender<BoxedFuture>,
    low_priority: tokio::sync::mpsc::UnboundedSender<BoxedFuture>,
}

type BoxedFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

impl PrioritySpawner {
    pub fn new(runtime: &Runtime) -> Self {
        let (high_tx, mut high_rx) = tokio::sync::mpsc::unbounded_channel::<BoxedFuture>();
        let (normal_tx, mut normal_rx) = tokio::sync::mpsc::unbounded_channel::<BoxedFuture>();
        let (low_tx, mut low_rx) = tokio::sync::mpsc::unbounded_channel::<BoxedFuture>();
        
        // Spawn executor tasks
        runtime.spawn(async move {
            while let Some(task) = high_rx.recv().await {
                tokio::spawn(task);
            }
        });
        
        runtime.spawn(async move {
            while let Some(task) = normal_rx.recv().await {
                tokio::spawn(task);
                // Add small yield for lower priority
                tokio::task::yield_now().await;
            }
        });
        
        runtime.spawn(async move {
            while let Some(task) = low_rx.recv().await {
                tokio::spawn(task);
                // Add larger yield for lowest priority
                tokio::time::sleep(Duration::from_micros(10)).await;
            }
        });
        
        Self {
            high_priority: high_tx,
            normal_priority: normal_tx,
            low_priority: low_tx,
        }
    }
    
    pub fn spawn_high<F>(&self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let _ = self.high_priority.send(Box::pin(future));
    }
    
    pub fn spawn_normal<F>(&self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let _ = self.normal_priority.send(Box::pin(future));
    }
    
    pub fn spawn_low<F>(&self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let _ = self.low_priority.send(Box::pin(future));
    }
}

/// Blocking task pool for CPU-intensive operations
pub struct BlockingPool {
    pool: rayon::ThreadPool,
}

impl BlockingPool {
    pub fn new() -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .thread_name(|i| format!("nexus-blocking-{}", i))
            .panic_handler(|panic| {
                if let Some(msg) = panic.downcast_ref::<&str>() {
                    tracing::error!("Blocking pool panic: {}", msg);
                } else if let Some(msg) = panic.downcast_ref::<String>() {
                    tracing::error!("Blocking pool panic: {}", msg);
                } else {
                    tracing::error!("Blocking pool panic: unknown");
                }
            })
            .build()
            .expect("Failed to create blocking pool");
        
        Self { pool }
    }
    
    pub async fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        self.pool.spawn(move || {
            let result = f();
            let _ = tx.send(result);
        });
        
        rx.await.expect("Blocking task failed")
    }
}

/// Local task set for single-threaded execution
pub struct LocalExecutor {
    local: tokio::task::LocalSet,
}

impl LocalExecutor {
    pub fn new() -> Self {
        Self {
            local: tokio::task::LocalSet::new(),
        }
    }
    
    pub fn spawn_local<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + 'static,
        F::Output: 'static,
    {
        self.local.spawn_local(future)
    }
    
    pub async fn run(&mut self) {
        // LocalSet needs to be run within a runtime context
        self.local.run_until(std::future::pending::<()>()).await;
    }
}

/// Shutdown coordinator for graceful shutdown
pub struct ShutdownCoordinator {
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
    tasks: Arc<parking_lot::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl ShutdownCoordinator {
    pub fn new() -> Self {
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
        
        Self {
            shutdown_tx,
            tasks: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }
    
    pub fn register_task(&self, handle: tokio::task::JoinHandle<()>) {
        self.tasks.lock().push(handle);
    }
    
    pub fn shutdown_signal(&self) -> tokio::sync::broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }
    
    pub async fn shutdown(self, timeout: Duration) {
        info!("Initiating graceful shutdown");
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Wait for tasks with timeout
        let tasks = self.tasks.lock().drain(..).collect::<Vec<_>>();
        
        let shutdown_future = async {
            for task in tasks {
                let _ = task.await;
            }
        };
        
        if tokio::time::timeout(timeout, shutdown_future).await.is_err() {
            warn!("Some tasks did not complete within shutdown timeout");
        }
        
        info!("Shutdown complete");
    }
}

use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_creation() {
        let runtime = create_optimized_runtime().unwrap();
        assert!(runtime.metrics().num_workers() > 0);
    }
    
    #[tokio::test]
    async fn test_priority_spawner() {
        let runtime = create_optimized_runtime().unwrap();
        let spawner = PrioritySpawner::new(&runtime);
        
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        spawner.spawn_high(async move {
            counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_blocking_pool() {
        let pool = BlockingPool::new();
        
        let result = pool.run(|| {
            // CPU-intensive operation
            let mut sum = 0u64;
            for i in 0..1000000 {
                sum += i;
            }
            sum
        }).await;
        
        assert_eq!(result, 499999500000);
    }
}