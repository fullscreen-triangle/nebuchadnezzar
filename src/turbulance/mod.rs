//! # Turbulance Language Parser and Compiler
//!
//! This module provides a complete parser and compiler for the Turbulance domain-specific language,
//! designed for scientific reasoning, pattern analysis, and evidence-based thinking.
//!
//! ## Components
//!
//! - **Lexer**: Tokenizes Turbulance source code
//! - **Parser**: Creates Abstract Syntax Tree (AST) from tokens
//! - **AST**: Represents Turbulance program structure
//! - **Compiler**: Compiles Turbulance code to Nebuchadnezzar operations
//! - **Runtime**: Executes compiled Turbulance programs

pub mod lexer;
pub mod parser;
pub mod ast;
pub mod compiler;
pub mod runtime;
pub mod integration;

pub use lexer::{Lexer, Token, TokenType};
pub use parser::{Parser, ParseError};
pub use ast::{TurbulanceAst, AstNode};
pub use compiler::{TurbulanceCompiler, CompilerError};
pub use runtime::{TurbulanceRuntime, RuntimeError};
pub use integration::NebuIntegration;

use crate::error::{NebuchadnezzarError, Result};
use std::collections::HashMap;

/// Main interface for the Turbulance language system
pub struct TurbulanceEngine {
    lexer: Lexer,
    parser: Parser,
    compiler: TurbulanceCompiler,
    runtime: TurbulanceRuntime,
    integration: NebuIntegration,
}

impl TurbulanceEngine {
    /// Create a new Turbulance engine instance
    pub fn new() -> Self {
        Self {
            lexer: Lexer::new(),
            parser: Parser::new(),
            compiler: TurbulanceCompiler::new(),
            runtime: TurbulanceRuntime::new(),
            integration: NebuIntegration::new(),
        }
    }

    /// Parse and execute Turbulance source code
    pub fn execute(&mut self, source: &str) -> Result<TurbulanceResult> {
        // Tokenize
        let tokens = self.lexer.tokenize(source)
            .map_err(|e| NebuchadnezzarError::parse_error(format!("Lexer error: {}", e)))?;

        // Parse to AST
        let ast = self.parser.parse(tokens)
            .map_err(|e| NebuchadnezzarError::parse_error(format!("Parser error: {}", e)))?;

        // Compile to bytecode/operations
        let program = self.compiler.compile(ast)
            .map_err(|e| NebuchadnezzarError::parse_error(format!("Compiler error: {}", e)))?;

        // Execute
        let result = self.runtime.execute(program)
            .map_err(|e| NebuchadnezzarError::execution_error(format!("Runtime error: {}", e)))?;

        Ok(result)
    }

    /// Execute a Turbulance file
    pub fn execute_file(&mut self, path: &str) -> Result<TurbulanceResult> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| NebuchadnezzarError::IoError(e))?;
        self.execute(&source)
    }

    /// Get the current integration context
    pub fn integration(&self) -> &NebuIntegration {
        &self.integration
    }

    /// Get mutable access to the integration context
    pub fn integration_mut(&mut self) -> &mut NebuIntegration {
        &mut self.integration
    }
}

impl Default for TurbulanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of executing Turbulance code
#[derive(Debug, Clone)]
pub struct TurbulanceResult {
    pub propositions: Vec<PropositionResult>,
    pub goals: Vec<GoalResult>,
    pub evidence: Vec<EvidenceResult>,
    pub variables: HashMap<String, TurbulanceValue>,
    pub metrics: ExecutionMetrics,
}

/// Result of evaluating a proposition
#[derive(Debug, Clone)]
pub struct PropositionResult {
    pub name: String,
    pub motions: Vec<MotionResult>,
    pub overall_support: f64,
    pub confidence: f64,
    pub evidence_count: usize,
}

/// Result of evaluating a motion
#[derive(Debug, Clone)]
pub struct MotionResult {
    pub name: String,
    pub description: String,
    pub support_level: f64,
    pub evidence: Vec<String>,
    pub status: MotionStatus,
}

/// Status of a motion evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum MotionStatus {
    Supported,
    Contradicted,
    Inconclusive,
    InsufficientEvidence,
}

/// Result of goal tracking
#[derive(Debug, Clone)]
pub struct GoalResult {
    pub name: String,
    pub progress: f64,
    pub status: GoalStatus,
    pub metrics: HashMap<String, f64>,
    pub suggestions: Vec<String>,
}

/// Status of a goal
#[derive(Debug, Clone, PartialEq)]
pub enum GoalStatus {
    Active,
    Paused,
    Completed,
    Failed,
}

/// Result of evidence collection
#[derive(Debug, Clone)]
pub struct EvidenceResult {
    pub source: String,
    pub data_type: String,
    pub quality: f64,
    pub relevance: f64,
    pub patterns: Vec<String>,
}

/// Values that can be stored and manipulated in Turbulance
#[derive(Debug, Clone)]
pub enum TurbulanceValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<TurbulanceValue>),
    Object(HashMap<String, TurbulanceValue>),
    BiologicalData(BiologicalDataValue),
}

/// Biological data types specific to Nebuchadnezzar
#[derive(Debug, Clone)]
pub enum BiologicalDataValue {
    AtpPool(crate::AtpPool),
    OscillationState(crate::OscillationState),
    QuantumMembrane(crate::QuantumMembrane),
    CircuitGrid(crate::CircuitGrid),
    TimeSeries(Vec<(f64, f64)>), // (time, value) pairs
    Sequence(String), // DNA/RNA/Protein sequences
    Pattern(PatternData),
}

/// Pattern data for pattern matching
#[derive(Debug, Clone)]
pub struct PatternData {
    pub pattern_type: String,
    pub confidence: f64,
    pub matches: Vec<PatternMatch>,
}

/// Individual pattern match
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub location: usize,
    pub length: usize,
    pub score: f64,
    pub context: String,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub parse_time_ms: u64,
    pub compile_time_ms: u64,
    pub execute_time_ms: u64,
    pub memory_usage_kb: u64,
    pub operations_count: u64,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            parse_time_ms: 0,
            compile_time_ms: 0,
            execute_time_ms: 0,
            memory_usage_kb: 0,
            operations_count: 0,
        }
    }
} 