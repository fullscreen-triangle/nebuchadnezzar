//! # Turbulance Abstract Syntax Tree
//!
//! Defines the AST nodes for representing Turbulance programs in memory.

use std::collections::HashMap;

/// Root AST node for a Turbulance program
#[derive(Debug, Clone)]
pub struct TurbulanceAst {
    pub statements: Vec<AstNode>,
}

impl TurbulanceAst {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }

    pub fn add_statement(&mut self, statement: AstNode) {
        self.statements.push(statement);
    }
}

impl Default for TurbulanceAst {
    fn default() -> Self {
        Self::new()
    }
}

/// AST node types
#[derive(Debug, Clone)]
pub enum AstNode {
    // Basic language constructs
    VariableDeclaration {
        name: String,
        value: Option<Box<AstNode>>,
        type_annotation: Option<String>,
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<Parameter>,
        return_type: Option<String>,
        body: Vec<AstNode>,
    },
    Assignment {
        target: String,
        value: Box<AstNode>,
    },
    Return {
        value: Option<Box<AstNode>>,
    },

    // Control flow
    Given {
        condition: Box<AstNode>,
        then_branch: Vec<AstNode>,
        else_branch: Option<Vec<AstNode>>,
    },
    Within {
        scope: Box<AstNode>,
        patterns: Vec<PatternMatch>,
        body: Vec<AstNode>,
    },
    Considering {
        iterator: String,
        collection: Box<AstNode>,
        condition: Option<Box<AstNode>>,
        body: Vec<AstNode>,
    },
    ForEach {
        iterator: String,
        collection: Box<AstNode>,
        body: Vec<AstNode>,
    },
    While {
        condition: Box<AstNode>,
        body: Vec<AstNode>,
    },

    // Scientific reasoning constructs
    Proposition {
        name: String,
        context: Option<Box<AstNode>>,
        motions: Vec<MotionDeclaration>,
        evaluation_blocks: Vec<EvaluationBlock>,
    },
    Motion {
        name: String,
        description: String,
        requirements: Option<Requirements>,
        criteria: Option<Criteria>,
        patterns: Vec<String>,
    },
    Evidence {
        name: String,
        sources: Vec<EvidenceSource>,
        collection: Option<CollectionSpec>,
        processing: Vec<ProcessingStep>,
        storage: Option<StorageSpec>,
    },
    Support {
        motion: String,
        weight: Option<Box<AstNode>>,
    },
    Contradict {
        motion: String,
        weight: Option<Box<AstNode>>,
    },

    // Goal system
    Goal {
        name: String,
        description: String,
        success_threshold: Option<f64>,
        keywords: Vec<String>,
        domain: Option<String>,
        audience: Option<String>,
        priority: Option<Priority>,
        deadline: Option<String>,
        metrics: Option<HashMap<String, f64>>,
    },

    // Pattern matching
    PatternRegistry {
        name: String,
        categories: Vec<PatternCategory>,
        matching_rules: Option<MatchingRules>,
        relationships: Vec<PatternRelationship>,
    },

    // Temporal constructs
    Temporal {
        name: String,
        scope: TimeScope,
        patterns: Vec<TemporalPattern>,
        operations: Vec<TemporalOperation>,
    },

    // Metacognitive constructs
    Metacognitive {
        name: String,
        tracking: Vec<TrackingSpec>,
        evaluation: Vec<EvaluationSpec>,
        adaptation: Vec<AdaptationRule>,
    },

    // Expressions
    BinaryOperation {
        left: Box<AstNode>,
        operator: BinaryOperator,
        right: Box<AstNode>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<AstNode>,
    },
    FunctionCall {
        function: String,
        arguments: Vec<AstNode>,
    },
    MethodCall {
        object: Box<AstNode>,
        method: String,
        arguments: Vec<AstNode>,
    },
    Identifier {
        name: String,
    },
    Literal {
        value: LiteralValue,
    },
    Array {
        elements: Vec<AstNode>,
    },
    Object {
        fields: Vec<(String, AstNode)>,
    },

    // Pattern matching
    PatternMatch {
        pattern: String,
        variables: Vec<String>,
    },

    // Parallel processing
    Parallel {
        name: String,
        workers: Option<usize>,
        load_balancing: Option<String>,
        body: Vec<AstNode>,
    },
    Async {
        body: Box<AstNode>,
    },
    Await {
        expression: Box<AstNode>,
    },

    // Error handling
    TryBlock {
        try_body: Vec<AstNode>,
        catch_blocks: Vec<CatchBlock>,
        finally_block: Option<Vec<AstNode>>,
    },

    // Import/Export
    Import {
        module: String,
        items: Option<Vec<String>>,
        alias: Option<String>,
    },
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default_value: Option<AstNode>,
}

/// Pattern matching in within blocks
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: String,
    pub condition: Option<AstNode>,
    pub variables: Vec<String>,
}

/// Motion declaration
#[derive(Debug, Clone)]
pub struct MotionDeclaration {
    pub name: String,
    pub description: String,
    pub requirements: Option<Requirements>,
    pub criteria: Option<Criteria>,
    pub patterns: Vec<String>,
}

/// Motion requirements
#[derive(Debug, Clone)]
pub struct Requirements {
    pub data_types: Vec<String>,
    pub min_data_points: Option<usize>,
    pub quality_threshold: Option<f64>,
}

/// Motion criteria
#[derive(Debug, Clone)]
pub struct Criteria {
    pub conditions: Vec<AstNode>,
    pub thresholds: HashMap<String, f64>,
}

/// Evidence evaluation block
#[derive(Debug, Clone)]
pub struct EvaluationBlock {
    pub scope: Option<AstNode>,
    pub conditions: Vec<EvaluationCondition>,
}

/// Individual evaluation condition
#[derive(Debug, Clone)]
pub struct EvaluationCondition {
    pub condition: AstNode,
    pub action: EvaluationAction,
}

/// Action to take when evaluation condition is met
#[derive(Debug, Clone)]
pub enum EvaluationAction {
    Support { motion: String, weight: Option<AstNode> },
    Contradict { motion: String, weight: Option<AstNode> },
    Collect { evidence_type: String },
    Flag { uncertainty: String },
}

/// Evidence source specification
#[derive(Debug, Clone)]
pub struct EvidenceSource {
    pub name: String,
    pub source_type: String,
    pub location: String,
    pub validation: Option<String>,
}

/// Data collection specification
#[derive(Debug, Clone)]
pub struct CollectionSpec {
    pub frequency: Option<String>,
    pub duration: Option<String>,
    pub validation: Option<String>,
    pub quality_threshold: Option<f64>,
}

/// Processing step for evidence
#[derive(Debug, Clone)]
pub struct ProcessingStep {
    pub name: String,
    pub operation: String,
    pub parameters: HashMap<String, AstNode>,
}

/// Storage specification
#[derive(Debug, Clone)]
pub struct StorageSpec {
    pub format: String,
    pub compression: Option<String>,
    pub indexing: Option<String>,
}

/// Goal priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Pattern category in pattern registry
#[derive(Debug, Clone)]
pub struct PatternCategory {
    pub name: String,
    pub patterns: Vec<PatternDefinition>,
}

/// Individual pattern definition
#[derive(Debug, Clone)]
pub struct PatternDefinition {
    pub name: String,
    pub pattern_type: String,
    pub confidence_threshold: Option<f64>,
}

/// Pattern matching rules
#[derive(Debug, Clone)]
pub struct MatchingRules {
    pub threshold: f64,
    pub context_window: Option<usize>,
    pub overlap_policy: String,
    pub confidence_level: f64,
}

/// Relationship between patterns
#[derive(Debug, Clone)]
pub struct PatternRelationship {
    pub from_pattern: String,
    pub to_pattern: String,
    pub relationship_type: String,
    pub strength: Option<f64>,
}

/// Time scope for temporal constructs
#[derive(Debug, Clone)]
pub struct TimeScope {
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub resolution: Option<String>,
    pub time_zone: Option<String>,
}

/// Temporal pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub name: String,
    pub pattern_type: TemporalPatternType,
    pub parameters: HashMap<String, AstNode>,
}

/// Types of temporal patterns
#[derive(Debug, Clone)]
pub enum TemporalPatternType {
    Periodic,
    Trending,
    Seasonal,
    Anomalous,
}

/// Temporal operation
#[derive(Debug, Clone)]
pub struct TemporalOperation {
    pub name: String,
    pub operation_type: String,
    pub parameters: HashMap<String, AstNode>,
}

/// Tracking specification for metacognitive constructs
#[derive(Debug, Clone)]
pub struct TrackingSpec {
    pub name: String,
    pub data_type: String,
    pub collection_method: String,
}

/// Evaluation specification for metacognitive constructs
#[derive(Debug, Clone)]
pub struct EvaluationSpec {
    pub name: String,
    pub method: String,
    pub criteria: Vec<AstNode>,
}

/// Adaptation rule for metacognitive constructs
#[derive(Debug, Clone)]
pub struct AdaptationRule {
    pub condition: AstNode,
    pub actions: Vec<AstNode>,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    
    // Comparison
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    
    // Logical
    And,
    Or,
    
    // Pattern matching
    Matches,
    Contains,
    
    // Assignment
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not,
    Minus,
    Plus,
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Integer(i64),
    Float(f64),
    String(String),
    Char(char),
    Boolean(bool),
}

/// Catch block for error handling
#[derive(Debug, Clone)]
pub struct CatchBlock {
    pub exception_type: Option<String>,
    pub variable: Option<String>,
    pub body: Vec<AstNode>,
}

impl TurbulanceAst {
    /// Pretty print the AST for debugging
    pub fn pretty_print(&self, indent: usize) -> String {
        let mut result = String::new();
        let indent_str = "  ".repeat(indent);
        
        result.push_str(&format!("{}TurbulanceAst {{\n", indent_str));
        for statement in &self.statements {
            result.push_str(&statement.pretty_print(indent + 1));
            result.push('\n');
        }
        result.push_str(&format!("{}}}\n", indent_str));
        
        result
    }
}

impl AstNode {
    /// Pretty print an AST node
    pub fn pretty_print(&self, indent: usize) -> String {
        let indent_str = "  ".repeat(indent);
        
        match self {
            AstNode::VariableDeclaration { name, value, type_annotation } => {
                let mut result = format!("{}VariableDeclaration {{ name: {}", indent_str, name);
                if let Some(typ) = type_annotation {
                    result.push_str(&format!(", type: {}", typ));
                }
                if let Some(val) = value {
                    result.push_str(&format!(",\n{}", val.pretty_print(indent + 1)));
                }
                result.push_str(" }");
                result
            }
            AstNode::FunctionDeclaration { name, parameters, return_type, body } => {
                let mut result = format!("{}FunctionDeclaration {{ name: {}", indent_str, name);
                if let Some(ret_type) = return_type {
                    result.push_str(&format!(", return_type: {}", ret_type));
                }
                result.push_str(&format!(",\n{}  parameters: [", indent_str));
                for param in parameters {
                    result.push_str(&format!("\n{}    {}", indent_str, param.name));
                }
                result.push_str(&format!("\n{}  ],", indent_str));
                result.push_str(&format!("\n{}  body: [", indent_str));
                for stmt in body {
                    result.push_str(&format!("\n{}", stmt.pretty_print(indent + 2)));
                }
                result.push_str(&format!("\n{}  ]", indent_str));
                result.push_str(" }");
                result
            }
            AstNode::Proposition { name, motions, .. } => {
                let mut result = format!("{}Proposition {{ name: {}", indent_str, name);
                result.push_str(&format!(",\n{}  motions: [", indent_str));
                for motion in motions {
                    result.push_str(&format!("\n{}    Motion {{ name: {}, description: {} }}", 
                                           indent_str, motion.name, motion.description));
                }
                result.push_str(&format!("\n{}  ]", indent_str));
                result.push_str(" }");
                result
            }
            AstNode::Literal { value } => {
                match value {
                    LiteralValue::Integer(i) => format!("{}Literal({})", indent_str, i),
                    LiteralValue::Float(f) => format!("{}Literal({})", indent_str, f),
                    LiteralValue::String(s) => format!("{}Literal(\"{}\")", indent_str, s),
                    LiteralValue::Boolean(b) => format!("{}Literal({})", indent_str, b),
                    LiteralValue::Char(c) => format!("{}Literal('{}')", indent_str, c),
                }
            }
            AstNode::Identifier { name } => {
                format!("{}Identifier({})", indent_str, name)
            }
            _ => format!("{}[Other AST Node]", indent_str),
        }
    }
} 