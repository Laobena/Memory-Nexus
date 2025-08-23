//! Contextual Boosting for Personalized Search Results
//! 
//! Boosts search results based on:
//! - User's tech stack
//! - Expertise level
//! - Current project context
//! - Recent query history

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// User context for personalized boosting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// Technologies the user is working with (e.g., ["Python", "Django", "PostgreSQL"])
    pub tech_stack: Vec<String>,
    /// Current project or domain (e.g., "e-commerce", "data-science")
    pub project_context: Option<String>,
    /// User expertise level (0.0=beginner, 0.5=intermediate, 1.0=expert)
    pub expertise_level: f32,
    /// Recent search queries for context awareness
    pub recent_queries: Vec<String>,
    /// User preferences (e.g., prefers examples, avoids theory)
    pub preferences: UserPreferences,
}

impl Default for UserContext {
    fn default() -> Self {
        Self {
            tech_stack: Vec::new(),
            project_context: None,
            expertise_level: 0.5,
            recent_queries: Vec::new(),
            preferences: UserPreferences::default(),
        }
    }
}

/// User preferences for content style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub prefers_examples: bool,
    pub prefers_theory: bool,
    pub prefers_visual: bool,
    pub prefers_concise: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            prefers_examples: true,
            prefers_theory: false,
            prefers_visual: false,
            prefers_concise: true,
        }
    }
}

/// Contextual booster for search results
pub struct ContextBooster {
    user_context: UserContext,
    tech_stack_set: HashSet<String>,
}

impl ContextBooster {
    pub fn new(user_context: UserContext) -> Self {
        let tech_stack_set: HashSet<String> = user_context.tech_stack
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        
        Self {
            user_context,
            tech_stack_set,
        }
    }
    
    /// Calculate contextual boost for a search result
    pub fn calculate_boost(&self, content: &str, metadata: &serde_json::Value) -> f32 {
        let mut boost = 1.0;
        
        // 1. Tech stack boost (up to 1.5x)
        boost *= self.calculate_tech_stack_boost(content);
        
        // 2. Expertise level adjustment (0.8x to 1.2x)
        boost *= self.calculate_expertise_adjustment(content, metadata);
        
        // 3. Project context boost (up to 1.3x)
        boost *= self.calculate_project_boost(content);
        
        // 4. Preference alignment (up to 1.2x)
        boost *= self.calculate_preference_boost(content, metadata);
        
        // 5. Recent query relevance (up to 1.1x)
        boost *= self.calculate_recency_boost(content);
        
        // Cap maximum boost at 2x to prevent over-amplification
        boost.min(2.0).max(0.5)
    }
    
    /// Calculate tech stack relevance boost
    fn calculate_tech_stack_boost(&self, content: &str) -> f32 {
        if self.tech_stack_set.is_empty() {
            return 1.0;
        }
        
        let content_lower = content.to_lowercase();
        let mut matches = 0;
        let mut strong_matches = 0;
        
        for tech in &self.tech_stack_set {
            if content_lower.contains(tech) {
                matches += 1;
                
                // Strong match if tech appears multiple times or in title
                let count = content_lower.matches(tech).count();
                if count > 2 {
                    strong_matches += 1;
                }
            }
        }
        
        // Calculate boost based on matches
        let match_ratio = matches as f32 / self.tech_stack_set.len() as f32;
        let strong_ratio = strong_matches as f32 / self.tech_stack_set.len().max(1) as f32;
        
        // Base boost from matches
        let base_boost = 1.0 + (match_ratio * 0.3);  // Up to 1.3x
        
        // Additional boost for strong matches
        let strong_boost = 1.0 + (strong_ratio * 0.2);  // Up to 1.2x
        
        base_boost * strong_boost
    }
    
    /// Adjust score based on content complexity vs user expertise
    fn calculate_expertise_adjustment(&self, content: &str, metadata: &serde_json::Value) -> f32 {
        // Estimate content complexity
        let complexity = self.estimate_complexity(content, metadata);
        
        let expertise = self.user_context.expertise_level;
        
        // If content matches expertise level, boost slightly
        let diff = (expertise - complexity).abs();
        
        if diff < 0.2 {
            // Perfect match
            1.2
        } else if diff < 0.4 {
            // Good match
            1.1
        } else if complexity > expertise + 0.5 {
            // Too complex for user
            0.8
        } else if complexity < expertise - 0.5 {
            // Too simple for user
            0.9
        } else {
            // Neutral
            1.0
        }
    }
    
    /// Estimate content complexity (0.0=beginner, 1.0=expert)
    fn estimate_complexity(&self, content: &str, metadata: &serde_json::Value) -> f32 {
        let mut complexity = 0.5;  // Default: intermediate
        
        // Check metadata for explicit complexity
        if let Some(level) = metadata.get("complexity").and_then(|v| v.as_f64()) {
            return level as f32;
        }
        
        // Heuristic: technical terms increase complexity
        let technical_terms = [
            "algorithm", "optimization", "architecture", "implementation",
            "abstract", "polymorphism", "concurrency", "distributed",
            "kernel", "compilation", "bytecode", "assembly"
        ];
        
        let term_count = technical_terms.iter()
            .filter(|&&term| content.to_lowercase().contains(term))
            .count();
        
        complexity += (term_count as f32 * 0.05).min(0.3);
        
        // Code blocks suggest higher complexity
        if content.contains("```") || content.contains("fn ") || content.contains("def ") {
            complexity += 0.1;
        }
        
        // Math/formulas suggest higher complexity
        if content.contains("Σ") || content.contains("∫") || content.contains("≈") {
            complexity += 0.2;
        }
        
        complexity.min(1.0)
    }
    
    /// Calculate project context relevance boost
    fn calculate_project_boost(&self, content: &str) -> f32 {
        if let Some(project) = &self.user_context.project_context {
            let content_lower = content.to_lowercase();
            let project_lower = project.to_lowercase();
            
            // Direct mention of project type
            if content_lower.contains(&project_lower) {
                return 1.3;
            }
            
            // Related domains
            let related = match project_lower.as_str() {
                "web" | "frontend" => ["html", "css", "javascript", "react", "vue"],
                "backend" => ["api", "database", "server", "rest", "graphql"],
                "data-science" => ["pandas", "numpy", "scikit", "tensorflow", "pytorch"],
                "mobile" => ["ios", "android", "swift", "kotlin", "flutter"],
                "devops" => ["docker", "kubernetes", "ci/cd", "aws", "azure"],
                _ => return 1.0,
            };
            
            let matches = related.iter()
                .filter(|&&term| content_lower.contains(term))
                .count();
            
            if matches > 0 {
                1.0 + (matches as f32 * 0.1).min(0.2)
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
    
    /// Calculate preference alignment boost
    fn calculate_preference_boost(&self, content: &str, metadata: &serde_json::Value) -> f32 {
        let mut boost = 1.0;
        let content_lower = content.to_lowercase();
        
        // Check for examples
        if self.user_context.preferences.prefers_examples {
            if content_lower.contains("example") || content_lower.contains("```") {
                boost *= 1.1;
            }
        }
        
        // Check for theory
        if self.user_context.preferences.prefers_theory {
            if content_lower.contains("theory") || content_lower.contains("concept") {
                boost *= 1.1;
            }
        } else if content_lower.contains("theory") || content_lower.contains("abstract") {
            boost *= 0.9;  // Penalize theory if not preferred
        }
        
        // Check for visual content
        if self.user_context.preferences.prefers_visual {
            if content.contains("![") || metadata.get("has_images").is_some() {
                boost *= 1.1;
            }
        }
        
        // Check for conciseness
        if self.user_context.preferences.prefers_concise {
            let word_count = content.split_whitespace().count();
            if word_count < 200 {
                boost *= 1.1;
            } else if word_count > 1000 {
                boost *= 0.9;
            }
        }
        
        boost
    }
    
    /// Calculate boost based on recent query relevance
    fn calculate_recency_boost(&self, content: &str) -> f32 {
        if self.user_context.recent_queries.is_empty() {
            return 1.0;
        }
        
        let content_lower = content.to_lowercase();
        let mut relevance = 0.0;
        
        // Check last 5 queries (more recent = higher weight)
        for (i, query) in self.user_context.recent_queries.iter().rev().take(5).enumerate() {
            let weight = 1.0 / (i as f32 + 1.0);  // Recent queries weighted more
            
            // Check if content relates to recent query
            let query_terms: Vec<&str> = query.to_lowercase().split_whitespace().collect();
            let matches = query_terms.iter()
                .filter(|&&term| content_lower.contains(term))
                .count();
            
            if matches > 0 {
                let match_ratio = matches as f32 / query_terms.len() as f32;
                relevance += match_ratio * weight;
            }
        }
        
        // Convert relevance to boost (max 1.1x)
        1.0 + (relevance * 0.1).min(0.1)
    }
    
    /// Create from user ID (would load from database in production)
    pub fn from_user_id(user_id: &str) -> Self {
        // In production, load user context from database
        // For now, return default or mock data
        
        let mock_context = match user_id {
            "developer" => UserContext {
                tech_stack: vec!["Rust".to_string(), "Python".to_string(), "Docker".to_string()],
                project_context: Some("backend".to_string()),
                expertise_level: 0.8,
                recent_queries: vec!["error handling".to_string(), "async rust".to_string()],
                preferences: UserPreferences {
                    prefers_examples: true,
                    prefers_theory: false,
                    prefers_visual: false,
                    prefers_concise: true,
                },
            },
            _ => UserContext::default(),
        };
        
        Self::new(mock_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tech_stack_boost() {
        let context = UserContext {
            tech_stack: vec!["rust".to_string(), "python".to_string()],
            ..Default::default()
        };
        let booster = ContextBooster::new(context);
        
        // Content with tech stack should get boost
        let boost1 = booster.calculate_tech_stack_boost("Rust async programming guide");
        assert!(boost1 > 1.0);
        
        // Content without tech stack should be neutral
        let boost2 = booster.calculate_tech_stack_boost("JavaScript tutorial");
        assert_eq!(boost2, 1.0);
    }
    
    #[test]
    fn test_expertise_adjustment() {
        let context = UserContext {
            expertise_level: 0.3,  // Beginner
            ..Default::default()
        };
        let booster = ContextBooster::new(context);
        
        // Simple content should get boost for beginner
        let simple_content = "Introduction to programming basics";
        let boost = booster.calculate_expertise_adjustment(simple_content, &serde_json::json!({}));
        assert!(boost >= 1.0);
        
        // Complex content should get penalty for beginner
        let complex_content = "Advanced kernel programming with assembly optimization";
        let penalty = booster.calculate_expertise_adjustment(complex_content, &serde_json::json!({}));
        assert!(penalty < 1.0);
    }
}