//! Fallback routing mechanisms for resilience

use crate::core::{BlockError, BlockResult, BlockInput};
use super::RoutingDecision;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, warn};

/// Fallback router for when primary routing fails
pub struct FallbackRouter {
    /// Static routing rules
    static_routes: Arc<RwLock<HashMap<String, String>>>,
    /// Default route when no rules match
    default_route: String,
    /// Health status
    health: Arc<RwLock<RouterHealth>>,
}

/// Router health information
#[derive(Debug, Clone)]
pub struct RouterHealth {
    pub healthy_routes: Vec<String>,
    pub unhealthy_routes: Vec<String>,
    pub last_check: std::time::Instant,
}

impl Default for RouterHealth {
    fn default() -> Self {
        Self {
            healthy_routes: vec![
                "cache".to_string(),
                "search".to_string(),
                "storage".to_string(),
                "fusion".to_string(),
            ],
            unhealthy_routes: Vec::new(),
            last_check: std::time::Instant::now(),
        }
    }
}

impl FallbackRouter {
    pub fn new() -> Self {
        Self {
            static_routes: Arc::new(RwLock::new(Self::default_routes())),
            default_route: "cache".to_string(),
            health: Arc::new(RwLock::new(RouterHealth::default())),
        }
    }
    
    /// Get default routing rules
    fn default_routes() -> HashMap<String, String> {
        let mut routes = HashMap::new();
        
        // Pattern-based static routes
        routes.insert("get_*".to_string(), "cache".to_string());
        routes.insert("search_*".to_string(), "search".to_string());
        routes.insert("store_*".to_string(), "storage".to_string());
        routes.insert("query_*".to_string(), "search".to_string());
        routes.insert("save_*".to_string(), "storage".to_string());
        routes.insert("fetch_*".to_string(), "cache".to_string());
        routes.insert("find_*".to_string(), "search".to_string());
        routes.insert("merge_*".to_string(), "fusion".to_string());
        
        routes
    }
    
    /// Route using fallback mechanisms
    pub async fn route(&self, input: &BlockInput) -> BlockResult<RoutingDecision> {
        let start = std::time::Instant::now();
        
        // Try pattern matching first
        let route = match input {
            BlockInput::Text(text) => self.route_by_pattern(text),
            BlockInput::Query(query) => self.route_by_pattern(&query.query),
            _ => self.get_healthy_route(),
        };
        
        let latency_us = start.elapsed().as_micros() as u64;
        
        info!("Fallback routing to: {} ({}μs)", route, latency_us);
        
        Ok(RoutingDecision {
            route,
            confidence: 0.7, // Lower confidence for fallback
            fallback_available: false, // Already in fallback
            latency_us,
        })
    }
    
    /// Route based on pattern matching
    fn route_by_pattern(&self, text: &str) -> String {
        let routes = self.static_routes.read();
        let lower = text.to_lowercase();
        
        // Check each pattern
        for (pattern, route) in routes.iter() {
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                if lower.starts_with(prefix) {
                    return self.validate_route(route.clone());
                }
            } else if lower.contains(pattern) {
                return self.validate_route(route.clone());
            }
        }
        
        // Use round-robin among healthy routes
        self.get_healthy_route()
    }
    
    /// Validate route is healthy
    fn validate_route(&self, route: String) -> String {
        let health = self.health.read();
        
        if health.healthy_routes.contains(&route) {
            route
        } else {
            // Return first healthy route
            health.healthy_routes
                .first()
                .cloned()
                .unwrap_or(self.default_route.clone())
        }
    }
    
    /// Get a healthy route using round-robin
    fn get_healthy_route(&self) -> String {
        let health = self.health.read();
        
        if health.healthy_routes.is_empty() {
            warn!("No healthy routes available, using default");
            return self.default_route.clone();
        }
        
        // Simple round-robin based on timestamp
        let index = (health.last_check.elapsed().as_secs() as usize) % health.healthy_routes.len();
        health.healthy_routes[index].clone()
    }
    
    /// Get default route
    pub fn get_default_route(&self) -> String {
        self.default_route.clone()
    }
    
    /// Update route health
    pub fn update_health(&self, route: String, healthy: bool) {
        let mut health = self.health.write();
        
        if healthy {
            // Move from unhealthy to healthy
            health.unhealthy_routes.retain(|r| r != &route);
            if !health.healthy_routes.contains(&route) {
                health.healthy_routes.push(route);
            }
        } else {
            // Move from healthy to unhealthy
            health.healthy_routes.retain(|r| r != &route);
            if !health.unhealthy_routes.contains(&route) {
                health.unhealthy_routes.push(route);
            }
        }
        
        health.last_check = std::time::Instant::now();
    }
    
    /// Add custom routing rule
    pub fn add_rule(&self, pattern: String, route: String) {
        let mut routes = self.static_routes.write();
        routes.insert(pattern, route);
    }
    
    /// Remove routing rule
    pub fn remove_rule(&self, pattern: &str) {
        let mut routes = self.static_routes.write();
        routes.remove(pattern);
    }
    
    /// Get current health status
    pub fn health_status(&self) -> RouterHealth {
        self.health.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fallback_routing() {
        let router = FallbackRouter::new();
        
        // Test pattern-based routing
        let input = BlockInput::Text("get_user_profile".to_string());
        let decision = router.route(&input).await.unwrap();
        assert_eq!(decision.route, "cache");
        
        let input = BlockInput::Text("search_documents".to_string());
        let decision = router.route(&input).await.unwrap();
        assert_eq!(decision.route, "search");
    }
    
    #[test]
    fn test_health_updates() {
        let router = FallbackRouter::new();
        
        // Mark cache as unhealthy
        router.update_health("cache".to_string(), false);
        
        let health = router.health_status();
        assert!(!health.healthy_routes.contains(&"cache".to_string()));
        assert!(health.unhealthy_routes.contains(&"cache".to_string()));
        
        // Mark cache as healthy again
        router.update_health("cache".to_string(), true);
        
        let health = router.health_status();
        assert!(health.healthy_routes.contains(&"cache".to_string()));
        assert!(!health.unhealthy_routes.contains(&"cache".to_string()));
    }
}