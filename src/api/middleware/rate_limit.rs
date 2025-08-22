use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Rate limiting middleware
#[derive(Clone)]
pub struct RateLimitLayer {
    limits: Arc<DashMap<String, RateLimit>>,
    max_requests: u32,
    window: Duration,
}

struct RateLimit {
    count: u32,
    window_start: Instant,
}

impl RateLimitLayer {
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            limits: Arc::new(DashMap::new()),
            max_requests,
            window,
        }
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitMiddleware<S>;
    
    fn layer(&self, inner: S) -> Self::Service {
        RateLimitMiddleware {
            inner,
            limits: self.limits.clone(),
            max_requests: self.max_requests,
            window: self.window,
        }
    }
}

#[derive(Clone)]
pub struct RateLimitMiddleware<S> {
    inner: S,
    limits: Arc<DashMap<String, RateLimit>>,
    max_requests: u32,
    window: Duration,
}

impl<S> Service<Request<Body>> for RateLimitMiddleware<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
    
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }
    
    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let limits = self.limits.clone();
        let max_requests = self.max_requests;
        let window = self.window;
        let mut inner = self.inner.clone();
        
        Box::pin(async move {
            // Extract client identifier (IP address or API key)
            let client_id = extract_client_id(&req);
            
            // Check rate limit
            let now = Instant::now();
            let mut rate_limit_exceeded = false;
            
            limits.alter(&client_id, |_, mut limit| {
                match limit {
                    Some(ref mut l) => {
                        if now.duration_since(l.window_start) > window {
                            // Reset window
                            l.count = 1;
                            l.window_start = now;
                        } else if l.count >= max_requests {
                            rate_limit_exceeded = true;
                        } else {
                            l.count += 1;
                        }
                    }
                    None => {
                        limit = Some(RateLimit {
                            count: 1,
                            window_start: now,
                        });
                    }
                }
                limit
            });
            
            if rate_limit_exceeded {
                let response = Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .header("Retry-After", window.as_secs().to_string())
                    .body(Body::from(r#"{"error":"Rate limit exceeded"}"#))
                    .unwrap();
                
                Ok(response)
            } else {
                inner.call(req).await
            }
        })
    }
}

fn extract_client_id(req: &Request<Body>) -> String {
    // Try to get from X-Forwarded-For header
    if let Some(forwarded) = req.headers().get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(ip) = forwarded_str.split(',').next() {
                return ip.trim().to_string();
            }
        }
    }
    
    // Try to get from X-Real-IP header
    if let Some(real_ip) = req.headers().get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }
    
    // Default to a generic identifier
    "unknown".to_string()
}