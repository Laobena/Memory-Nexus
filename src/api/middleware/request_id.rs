use axum::{
    body::Body,
    http::{Request, Response, HeaderValue},
};
use tower::{Layer, Service};
use uuid::Uuid;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Request ID middleware for tracing
#[derive(Clone)]
pub struct RequestIdLayer;

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;
    
    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for RequestIdMiddleware<S>
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
    
    fn call(&mut self, mut req: Request<Body>) -> Self::Future {
        // Generate or extract request ID
        let request_id = if let Some(existing_id) = req.headers().get("x-request-id") {
            existing_id.to_str().unwrap_or("invalid").to_string()
        } else {
            Uuid::new_v4().to_string()
        };
        
        // Add request ID to headers
        req.headers_mut().insert(
            "x-request-id",
            HeaderValue::from_str(&request_id).unwrap(),
        );
        
        // Set request ID in tracing span
        let span = tracing::info_span!("request", request_id = %request_id);
        let _guard = span.enter();
        
        let mut inner = self.inner.clone();
        
        Box::pin(async move {
            let mut response = inner.call(req).await?;
            
            // Add request ID to response headers
            response.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&request_id).unwrap(),
            );
            
            Ok(response)
        })
    }
}