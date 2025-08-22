use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
    middleware::Next,
};
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Authentication middleware
#[derive(Clone)]
pub struct AuthLayer {
    require_auth: bool,
}

impl AuthLayer {
    pub fn new() -> Self {
        Self {
            require_auth: false, // Disabled by default
        }
    }
    
    pub fn with_auth_required(mut self) -> Self {
        self.require_auth = true;
        self
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthMiddleware<S>;
    
    fn layer(&self, inner: S) -> Self::Service {
        AuthMiddleware {
            inner,
            require_auth: self.require_auth,
        }
    }
}

#[derive(Clone)]
pub struct AuthMiddleware<S> {
    inner: S,
    require_auth: bool,
}

impl<S> Service<Request<Body>> for AuthMiddleware<S>
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
        let require_auth = self.require_auth;
        let mut inner = self.inner.clone();
        
        Box::pin(async move {
            if require_auth {
                // Check for API key in header
                if let Some(auth_header) = req.headers().get("authorization") {
                    if let Ok(auth_str) = auth_header.to_str() {
                        if auth_str.starts_with("Bearer ") {
                            let token = &auth_str[7..];
                            if validate_token(token).await {
                                return inner.call(req).await;
                            }
                        }
                    }
                }
                
                // No valid auth, return 401
                let response = Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .body(Body::from(r#"{"error":"Unauthorized"}"#))
                    .unwrap();
                
                Ok(response)
            } else {
                // Auth not required, proceed
                inner.call(req).await
            }
        })
    }
}

async fn validate_token(token: &str) -> bool {
    // Placeholder token validation
    // In production, this would validate against a database or auth service
    !token.is_empty()
}