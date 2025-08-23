//! Pipeline Orchestrator
//! 
//! Orchestrates complex workflows with multiple pipelines

use crate::packages::{Pipeline, PipelineOutput, PipelineError};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Orchestrator for complex workflows
pub struct Orchestrator {
    workflows: Arc<RwLock<HashMap<String, Workflow>>>,
    pipelines: Arc<RwLock<HashMap<String, Arc<dyn Pipeline>>>>,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            pipelines: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a pipeline
    pub async fn register_pipeline(&self, name: String, pipeline: Arc<dyn Pipeline>) {
        self.pipelines.write().await.insert(name, pipeline);
    }
    
    /// Register a workflow
    pub async fn register_workflow(&self, workflow: Workflow) {
        self.workflows.write().await.insert(workflow.name.clone(), workflow);
    }
    
    /// Execute a workflow
    pub async fn execute_workflow(
        &self,
        workflow_name: &str,
        input: &str,
    ) -> Result<WorkflowOutput, OrchestrationError> {
        let workflows = self.workflows.read().await;
        let workflow = workflows.get(workflow_name)
            .ok_or_else(|| OrchestrationError::WorkflowNotFound(workflow_name.to_string()))?;
        
        let mut context = WorkflowContext::new();
        context.set_input(input.to_string());
        
        for step in &workflow.steps {
            context = self.execute_step(step, context).await?;
        }
        
        Ok(WorkflowOutput {
            result: context.get_output(),
            steps_executed: workflow.steps.len(),
            total_latency_ms: context.total_latency_ms,
        })
    }
    
    /// Execute a workflow step
    async fn execute_step(
        &self,
        step: &WorkflowStep,
        mut context: WorkflowContext,
    ) -> Result<WorkflowContext, OrchestrationError> {
        let pipelines = self.pipelines.read().await;
        
        match &step.step_type {
            StepType::Pipeline(name) => {
                let pipeline = pipelines.get(name)
                    .ok_or_else(|| OrchestrationError::PipelineNotFound(name.clone()))?;
                
                let input = context.get_step_input(&step.input_mapping);
                let start = std::time::Instant::now();
                
                let output = pipeline.execute(&input).await
                    .map_err(|e| OrchestrationError::PipelineError(e))?;
                
                context.add_step_output(step.name.clone(), output);
                context.add_latency(start.elapsed().as_secs_f64() * 1000.0);
                
                Ok(context)
            }
            StepType::Parallel(names) => {
                let mut handles = Vec::new();
                
                for name in names {
                    if let Some(pipeline) = pipelines.get(name) {
                        let pipeline = pipeline.clone();
                        let input = context.get_step_input(&step.input_mapping);
                        
                        let handle = tokio::spawn(async move {
                            pipeline.execute(&input).await
                        });
                        handles.push((name.clone(), handle));
                    }
                }
                
                let start = std::time::Instant::now();
                
                for (name, handle) in handles {
                    match handle.await {
                        Ok(Ok(output)) => {
                            context.add_step_output(format!("{}-{}", step.name, name), output);
                        }
                        Ok(Err(e)) => {
                            return Err(OrchestrationError::PipelineError(e));
                        }
                        Err(e) => {
                            return Err(OrchestrationError::ExecutionError(e.to_string()));
                        }
                    }
                }
                
                context.add_latency(start.elapsed().as_secs_f64() * 1000.0);
                Ok(context)
            }
            StepType::Conditional { condition, if_true, if_false } => {
                let should_execute = self.evaluate_condition(condition, &context);
                
                let pipeline_name = if should_execute {
                    if_true
                } else {
                    if_false
                };
                
                let pipeline = pipelines.get(pipeline_name)
                    .ok_or_else(|| OrchestrationError::PipelineNotFound(pipeline_name.clone()))?;
                
                let input = context.get_step_input(&step.input_mapping);
                let start = std::time::Instant::now();
                
                let output = pipeline.execute(&input).await
                    .map_err(|e| OrchestrationError::PipelineError(e))?;
                
                context.add_step_output(step.name.clone(), output);
                context.add_latency(start.elapsed().as_secs_f64() * 1000.0);
                
                Ok(context)
            }
            StepType::Loop { pipeline, max_iterations } => {
                let pipeline = pipelines.get(pipeline)
                    .ok_or_else(|| OrchestrationError::PipelineNotFound(pipeline.clone()))?;
                
                let mut iteration = 0;
                let start = std::time::Instant::now();
                
                while iteration < *max_iterations {
                    let input = context.get_step_input(&step.input_mapping);
                    
                    let output = pipeline.execute(&input).await
                        .map_err(|e| OrchestrationError::PipelineError(e))?;
                    
                    context.add_step_output(format!("{}-{}", step.name, iteration), output);
                    
                    // Check loop condition (simplified - always continue)
                    iteration += 1;
                }
                
                context.add_latency(start.elapsed().as_secs_f64() * 1000.0);
                Ok(context)
            }
        }
    }
    
    /// Evaluate condition
    fn evaluate_condition(&self, condition: &str, context: &WorkflowContext) -> bool {
        // Simplified condition evaluation
        match condition {
            "always" => true,
            "never" => false,
            "has_output" => context.has_output(),
            _ => false,
        }
    }
}

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub name: String,
    pub description: String,
    pub steps: Vec<WorkflowStep>,
}

/// Workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub name: String,
    pub step_type: StepType,
    pub input_mapping: InputMapping,
    pub retry_policy: Option<RetryPolicy>,
}

/// Step type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    Pipeline(String),
    Parallel(Vec<String>),
    Conditional {
        condition: String,
        if_true: String,
        if_false: String,
    },
    Loop {
        pipeline: String,
        max_iterations: usize,
    },
}

/// Input mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMapping {
    pub source: InputSource,
    pub transform: Option<String>,
}

/// Input source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputSource {
    WorkflowInput,
    StepOutput(String),
    Constant(String),
}

/// Retry policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub backoff_ms: u64,
}

/// Workflow context
struct WorkflowContext {
    input: String,
    outputs: HashMap<String, PipelineOutput>,
    total_latency_ms: f64,
}

impl WorkflowContext {
    fn new() -> Self {
        Self {
            input: String::new(),
            outputs: HashMap::new(),
            total_latency_ms: 0.0,
        }
    }
    
    fn set_input(&mut self, input: String) {
        self.input = input;
    }
    
    fn get_step_input(&self, mapping: &InputMapping) -> String {
        match &mapping.source {
            InputSource::WorkflowInput => self.input.clone(),
            InputSource::StepOutput(name) => {
                if let Some(output) = self.outputs.get(name) {
                    format!("{:?}", output.result)
                } else {
                    String::new()
                }
            }
            InputSource::Constant(value) => value.clone(),
        }
    }
    
    fn add_step_output(&mut self, name: String, output: PipelineOutput) {
        self.outputs.insert(name, output);
    }
    
    fn add_latency(&mut self, latency_ms: f64) {
        self.total_latency_ms += latency_ms;
    }
    
    fn has_output(&self) -> bool {
        !self.outputs.is_empty()
    }
    
    fn get_output(&self) -> String {
        // Return last output as string
        if let Some((_, output)) = self.outputs.iter().last() {
            format!("{:?}", output.result)
        } else {
            String::new()
        }
    }
}

/// Workflow output
#[derive(Debug, Clone)]
pub struct WorkflowOutput {
    pub result: String,
    pub steps_executed: usize,
    pub total_latency_ms: f64,
}

/// Orchestration errors
#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    #[error("Workflow not found: {0}")]
    WorkflowNotFound(String),
    
    #[error("Pipeline not found: {0}")]
    PipelineNotFound(String),
    
    #[error("Pipeline error: {0}")]
    PipelineError(#[from] PipelineError),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

/// Workflow builder
pub struct WorkflowBuilder {
    name: String,
    description: String,
    steps: Vec<WorkflowStep>,
}

impl WorkflowBuilder {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: String::new(),
            steps: Vec::new(),
        }
    }
    
    pub fn description(mut self, desc: String) -> Self {
        self.description = desc;
        self
    }
    
    pub fn add_step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }
    
    pub fn add_pipeline_step(mut self, name: String, pipeline: String) -> Self {
        self.steps.push(WorkflowStep {
            name,
            step_type: StepType::Pipeline(pipeline),
            input_mapping: InputMapping {
                source: InputSource::WorkflowInput,
                transform: None,
            },
            retry_policy: None,
        });
        self
    }
    
    pub fn add_parallel_step(mut self, name: String, pipelines: Vec<String>) -> Self {
        self.steps.push(WorkflowStep {
            name,
            step_type: StepType::Parallel(pipelines),
            input_mapping: InputMapping {
                source: InputSource::WorkflowInput,
                transform: None,
            },
            retry_policy: None,
        });
        self
    }
    
    pub fn build(self) -> Workflow {
        Workflow {
            name: self.name,
            description: self.description,
            steps: self.steps,
        }
    }
}