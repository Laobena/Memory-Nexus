//! Production Ollama HTTP Client Implementation
//!
//! This file shows the correct implementation for Linux/Docker environments
//! where full HTTP dependencies are available.

#[cfg(not(target_os = "windows"))]
use serde::{Deserialize, Serialize};

/// Ollama API request structure for embeddings (CORRECT FORMAT)
#[derive(Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

/// Ollama API response structure for embeddings (CORRECT FORMAT)
/// Note: Ollama returns f64, not f32 in the actual API
#[derive(Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f64>,  // IMPORTANT: f64 not f32!
}

#[cfg(not(target_os = "windows"))]
impl super::LocalAIEngine {
    /// Generate embedding from Ollama API (PRODUCTION IMPLEMENTATION)
    pub async fn generate_embedding_from_ollama_production(&self, content: &str) -> super::AIResult<Vec<f32>> {
        use reqwest::Client;

        let request = EmbeddingRequest {
            model: self.model.clone(),
            prompt: content.to_string(),
        };

        let url = "http://localhost:11434/api/embeddings";

        let client = Client::new();
        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| super::AIError::HttpError(format!("Failed to send request: {}", e)))?;

        // Handle specific HTTP status codes
        match response.status().as_u16() {
            200 => {
                // Success - parse response
                let embedding_response: EmbeddingResponse = response.json().await
                    .map_err(|e| super::AIError::InvalidResponse(format!("Failed to parse response: {}", e)))?;

                // Convert f64 to f32 for our internal representation
                let embedding: Vec<f32> = embedding_response.embedding
                    .into_iter()
                    .map(|x| x as f32)
                    .collect();

                // Verify dimension
                if embedding.len() != Self::EXPECTED_EMBEDDING_DIM {
                    return Err(super::AIError::EmbeddingError(format!(
                        "Unexpected embedding dimension: {} (expected {})",
                        embedding.len(),
                        Self::EXPECTED_EMBEDDING_DIM
                    )));
                }

                Ok(embedding)
            },
            404 => {
                Err(super::AIError::ModelNotAvailable(format!(
                    "Model '{}' not found. Please run: ollama pull {}",
                    self.model, self.model
                )))
            },
            500 => {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown server error".to_string());
                Err(super::AIError::EmbeddingError(format!(
                    "Ollama server error: {}", error_text
                )))
            },
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                Err(super::AIError::HttpError(format!(
                    "Ollama API error {}: {}", status, error_text
                )))
            }
        }
    }

    /// Test connection to Ollama service (PRODUCTION IMPLEMENTATION)
    pub async fn test_connection_production(&self) -> super::AIResult<()> {
        use reqwest::Client;

        let client = Client::new();
        let url = "http://localhost:11434/api/tags";

        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| super::AIError::ConnectionError(format!("Failed to connect to Ollama: {}", e)))?;

        if !response.status().is_success() {
            return Err(super::AIError::ConnectionError(format!(
                "Ollama responded with status: {}",
                response.status()
            )));
        }

        // Check if our model is available
        let text = response.text().await
            .map_err(|e| super::AIError::HttpError(format!("Failed to read response: {}", e)))?;

        if !text.contains(&self.model) {
            return Err(super::AIError::ModelNotAvailable(format!(
                "Model '{}' not found. Please run: ollama pull {}",
                self.model, self.model
            )));
        }

        println!("✅ Ollama connection verified - model '{}' is available", self.model);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_serialization() {
        let request = EmbeddingRequest {
            model: "mxbai-embed-large".to_string(),
            prompt: "Test content".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("mxbai-embed-large"));
        assert!(json.contains("Test content"));
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{"embedding": [0.1, 0.2, 0.3]}"#;
        let response: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.embedding, vec![0.1, 0.2, 0.3]);
    }
}

/// Docker Compose Configuration for Development
///
/// Create docker-compose.yml in project root:
/// ```yaml
/// version: '3.8'
/// services:
///   ollama:
///     image: ollama/ollama:latest
///     ports:
///       - "11434:11434"
///     volumes:
///       - ollama_data:/root/.ollama
///     environment:
///       - OLLAMA_HOST=0.0.0.0
///     command: ["ollama", "serve"]
///
///   localmind:
///     build: .
///     depends_on:
///       - ollama
///     environment:
///       - OLLAMA_URL=http://ollama:11434
///     volumes:
///       - .:/app
///     working_dir: /app
///
/// volumes:
///   ollama_data:
/// ```
///
/// Usage:
/// ```bash
/// docker-compose up -d ollama
/// docker-compose exec ollama ollama pull mxbai-embed-large
/// docker-compose up localmind
/// ```