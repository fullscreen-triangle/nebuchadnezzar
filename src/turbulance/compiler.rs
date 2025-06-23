//! # Turbulance Compiler  
//!
//! Compiles Turbulance AST to executable operations that can interact with Nebuchadnezzar.

use crate::turbulance::ast::{TurbulanceAst, AstNode, BinaryOperator, UnaryOperator, LiteralValue};
use std::collections::HashMap;
use std::fmt;

/// Turbulance compiler
pub struct TurbulanceCompiler {
    /// Current scope for variable resolution
    scopes: Vec<HashMap<String, VariableInfo>>,
    
    /// Function definitions
    functions: HashMap<String, FunctionInfo>,
    
    /// Proposition definitions
    propositions: HashMap<String, PropositionInfo>,
    
    /// Goal definitions
    goals: HashMap<String, GoalInfo>,
}

impl TurbulanceCompiler {
    /// Create a new compiler instance
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()], // Global scope
            functions: HashMap::new(),
            propositions: HashMap::new(),
            goals: HashMap::new(),
        }
    }

    /// Compile an AST to a program
    pub fn compile(&mut self, ast: TurbulanceAst) -> Result<TurbulanceProgram, CompilerError> {
        let mut program = TurbulanceProgram::new();
        
        // First pass: collect declarations
        for statement in &ast.statements {
            self.collect_declarations(statement, &mut program)?;
        }
        
        // Second pass: compile statements
        for statement in &ast.statements {
            let instruction = self.compile_statement(statement)?;
            program.add_instruction(instruction);
        }
        
        Ok(program)
    }
    
    /// Collect function, proposition, and goal declarations
    fn collect_declarations(&mut self, node: &AstNode, program: &mut TurbulanceProgram) -> Result<(), CompilerError> {
        match node {
            AstNode::FunctionDeclaration { name, parameters, return_type, .. } => {
                let func_info = FunctionInfo {
                    name: name.clone(),
                    parameter_types: parameters.iter().map(|p| p.type_annotation.clone()).collect(),
                    return_type: return_type.clone(),
                    is_builtin: false,
                };
                self.functions.insert(name.clone(), func_info);
                program.add_function_declaration(name.clone());
            }
            AstNode::Proposition { name, motions, .. } => {
                let prop_info = PropositionInfo {
                    name: name.clone(),
                    motion_names: motions.iter().map(|m| m.name.clone()).collect(),
                };
                self.propositions.insert(name.clone(), prop_info);
                program.add_proposition_declaration(name.clone());
            }
            AstNode::Goal { name, description, .. } => {
                let goal_info = GoalInfo {
                    name: name.clone(),
                    description: description.clone(),
                };
                self.goals.insert(name.clone(), goal_info);
                program.add_goal_declaration(name.clone());
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Compile a statement to an instruction
    fn compile_statement(&mut self, node: &AstNode) -> Result<Instruction, CompilerError> {
        match node {
            AstNode::VariableDeclaration { name, value, type_annotation } => {
                let var_info = VariableInfo {
                    name: name.clone(),
                    var_type: type_annotation.clone(),
                    is_mutable: true,
                };
                
                self.current_scope_mut().insert(name.clone(), var_info);
                
                let value_instr = if let Some(val) = value {
                    self.compile_expression(val)?
                } else {
                    Operation::LoadConstant(Value::Null)
                };
                
                Ok(Instruction::VariableDeclaration {
                    name: name.clone(),
                    value: Box::new(value_instr),
                })
            }
            AstNode::Assignment { target, value } => {
                let value_instr = self.compile_expression(value)?;
                Ok(Instruction::Assignment {
                    target: target.clone(),
                    value: Box::new(value_instr),
                })
            }
            AstNode::FunctionDeclaration { name, parameters, body, .. } => {
                self.push_scope(); // Function scope
                
                // Add parameters to scope
                for param in parameters {
                    let var_info = VariableInfo {
                        name: param.name.clone(),
                        var_type: param.type_annotation.clone(),
                        is_mutable: false,
                    };
                    self.current_scope_mut().insert(param.name.clone(), var_info);
                }
                
                let mut body_instructions = Vec::new();
                for stmt in body {
                    body_instructions.push(self.compile_statement(stmt)?);
                }
                
                self.pop_scope();
                
                Ok(Instruction::FunctionDeclaration {
                    name: name.clone(),
                    parameter_names: parameters.iter().map(|p| p.name.clone()).collect(),
                    body: body_instructions,
                })
            }
            AstNode::Proposition { name, motions, evaluation_blocks, .. } => {
                let mut motion_instructions = Vec::new();
                for motion in motions {
                    motion_instructions.push(Instruction::MotionDeclaration {
                        name: motion.name.clone(),
                        description: motion.description.clone(),
                    });
                }
                
                let mut evaluation_instructions = Vec::new();
                for eval_block in evaluation_blocks {
                    // Compile evaluation blocks
                    for condition in &eval_block.conditions {
                        let condition_instr = self.compile_expression(&condition.condition)?;
                        evaluation_instructions.push(Instruction::EvaluationCondition {
                            condition: Box::new(condition_instr),
                            action: self.compile_evaluation_action(&condition.action)?,
                        });
                    }
                }
                
                Ok(Instruction::PropositionDeclaration {
                    name: name.clone(),
                    motions: motion_instructions,
                    evaluations: evaluation_instructions,
                })
            }
            AstNode::Goal { name, description, success_threshold, keywords, domain, audience, priority, deadline, metrics } => {
                Ok(Instruction::GoalDeclaration {
                    name: name.clone(),
                    description: description.clone(),
                    success_threshold: *success_threshold,
                    keywords: keywords.clone(),
                    domain: domain.clone(),
                    audience: audience.clone(),
                    priority: priority.clone(),
                    deadline: deadline.clone(),
                    metrics: metrics.clone().unwrap_or_default(),
                })
            }
            AstNode::Support { motion, weight } => {
                let weight_instr = if let Some(w) = weight {
                    Some(Box::new(self.compile_expression(w)?))
                } else {
                    None
                };
                
                Ok(Instruction::Support {
                    motion: motion.clone(),
                    weight: weight_instr,
                })
            }
            AstNode::Contradict { motion, weight } => {
                let weight_instr = if let Some(w) = weight {
                    Some(Box::new(self.compile_expression(w)?))
                } else {
                    None
                };
                
                Ok(Instruction::Contradict {
                    motion: motion.clone(),
                    weight: weight_instr,
                })
            }
            AstNode::Given { condition, then_branch, else_branch } => {
                let condition_instr = self.compile_expression(condition)?;
                
                let mut then_instructions = Vec::new();
                for stmt in then_branch {
                    then_instructions.push(self.compile_statement(stmt)?);
                }
                
                let else_instructions = if let Some(else_stmts) = else_branch {
                    let mut instructions = Vec::new();
                    for stmt in else_stmts {
                        instructions.push(self.compile_statement(stmt)?);
                    }
                    Some(instructions)
                } else {
                    None
                };
                
                Ok(Instruction::Conditional {
                    condition: Box::new(condition_instr),
                    then_branch: then_instructions,
                    else_branch: else_instructions,
                })
            }
            AstNode::Return { value } => {
                let value_instr = if let Some(val) = value {
                    Some(Box::new(self.compile_expression(val)?))
                } else {
                    None
                };
                
                Ok(Instruction::Return { value: value_instr })
            }
            _ => {
                // For other statements, treat as expressions
                let operation = self.compile_expression(node)?;
                Ok(Instruction::Expression(Box::new(operation)))
            }
        }
    }
    
    /// Compile an expression to an operation
    fn compile_expression(&mut self, node: &AstNode) -> Result<Operation, CompilerError> {
        match node {
            AstNode::Literal { value } => {
                Ok(Operation::LoadConstant(self.compile_literal(value)))
            }
            AstNode::Identifier { name } => {
                if self.resolve_variable(name).is_some() {
                    Ok(Operation::LoadVariable(name.clone()))
                } else if self.functions.contains_key(name) {
                    Ok(Operation::LoadFunction(name.clone()))
                } else {
                    Err(CompilerError::UndefinedVariable(name.clone()))
                }
            }
            AstNode::BinaryOperation { left, operator, right } => {
                let left_op = self.compile_expression(left)?;
                let right_op = self.compile_expression(right)?;
                
                Ok(Operation::BinaryOperation {
                    left: Box::new(left_op),
                    operator: self.compile_binary_operator(operator),
                    right: Box::new(right_op),
                })
            }
            AstNode::UnaryOperation { operator, operand } => {
                let operand_op = self.compile_expression(operand)?;
                
                Ok(Operation::UnaryOperation {
                    operator: self.compile_unary_operator(operator),
                    operand: Box::new(operand_op),
                })
            }
            AstNode::FunctionCall { function, arguments } => {
                let mut arg_operations = Vec::new();
                for arg in arguments {
                    arg_operations.push(self.compile_expression(arg)?);
                }
                
                Ok(Operation::FunctionCall {
                    function: function.clone(),
                    arguments: arg_operations,
                })
            }
            AstNode::MethodCall { object, method, arguments } => {
                let object_op = self.compile_expression(object)?;
                let mut arg_operations = Vec::new();
                for arg in arguments {
                    arg_operations.push(self.compile_expression(arg)?);
                }
                
                Ok(Operation::MethodCall {
                    object: Box::new(object_op),
                    method: method.clone(),
                    arguments: arg_operations,
                })
            }
            AstNode::Array { elements } => {
                let mut element_operations = Vec::new();
                for element in elements {
                    element_operations.push(self.compile_expression(element)?);
                }
                
                Ok(Operation::CreateArray(element_operations))
            }
            AstNode::Object { fields } => {
                let mut field_operations = Vec::new();
                for (key, value) in fields {
                    field_operations.push((key.clone(), self.compile_expression(value)?));
                }
                
                Ok(Operation::CreateObject(field_operations))
            }
            _ => Err(CompilerError::UnsupportedExpression(format!("{:?}", node))),
        }
    }
    
    /// Compile evaluation action
    fn compile_evaluation_action(&mut self, action: &crate::turbulance::ast::EvaluationAction) -> Result<EvaluationActionInstruction, CompilerError> {
        match action {
            crate::turbulance::ast::EvaluationAction::Support { motion, weight } => {
                let weight_instr = if let Some(w) = weight {
                    Some(Box::new(self.compile_expression(w)?))
                } else {
                    None
                };
                
                Ok(EvaluationActionInstruction::Support {
                    motion: motion.clone(),
                    weight: weight_instr,
                })
            }
            crate::turbulance::ast::EvaluationAction::Contradict { motion, weight } => {
                let weight_instr = if let Some(w) = weight {
                    Some(Box::new(self.compile_expression(w)?))
                } else {
                    None
                };
                
                Ok(EvaluationActionInstruction::Contradict {
                    motion: motion.clone(),
                    weight: weight_instr,
                })
            }
            crate::turbulance::ast::EvaluationAction::Collect { evidence_type } => {
                Ok(EvaluationActionInstruction::Collect {
                    evidence_type: evidence_type.clone(),
                })
            }
            crate::turbulance::ast::EvaluationAction::Flag { uncertainty } => {
                Ok(EvaluationActionInstruction::Flag {
                    uncertainty: uncertainty.clone(),
                })
            }
        }
    }
    
    /// Compile literal value
    fn compile_literal(&self, literal: &LiteralValue) -> Value {
        match literal {
            LiteralValue::Integer(i) => Value::Integer(*i),
            LiteralValue::Float(f) => Value::Float(*f),
            LiteralValue::String(s) => Value::String(s.clone()),
            LiteralValue::Boolean(b) => Value::Boolean(*b),
            LiteralValue::Char(c) => Value::Char(*c),
        }
    }
    
    /// Compile binary operator
    fn compile_binary_operator(&self, op: &BinaryOperator) -> BinaryOp {
        match op {
            BinaryOperator::Add => BinaryOp::Add,
            BinaryOperator::Subtract => BinaryOp::Subtract,
            BinaryOperator::Multiply => BinaryOp::Multiply,
            BinaryOperator::Divide => BinaryOp::Divide,
            BinaryOperator::Modulo => BinaryOp::Modulo,
            BinaryOperator::Power => BinaryOp::Power,
            BinaryOperator::Equal => BinaryOp::Equal,
            BinaryOperator::NotEqual => BinaryOp::NotEqual,
            BinaryOperator::Less => BinaryOp::Less,
            BinaryOperator::Greater => BinaryOp::Greater,
            BinaryOperator::LessEqual => BinaryOp::LessEqual,
            BinaryOperator::GreaterEqual => BinaryOp::GreaterEqual,
            BinaryOperator::And => BinaryOp::And,
            BinaryOperator::Or => BinaryOp::Or,
            BinaryOperator::Matches => BinaryOp::Matches,
            BinaryOperator::Contains => BinaryOp::Contains,
            _ => BinaryOp::Add, // Default fallback
        }
    }
    
    /// Compile unary operator
    fn compile_unary_operator(&self, op: &UnaryOperator) -> UnaryOp {
        match op {
            UnaryOperator::Not => UnaryOp::Not,
            UnaryOperator::Minus => UnaryOp::Minus,
            UnaryOperator::Plus => UnaryOp::Plus,
        }
    }
    
    // Scope management
    
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    
    fn current_scope_mut(&mut self) -> &mut HashMap<String, VariableInfo> {
        self.scopes.last_mut().unwrap()
    }
    
    fn resolve_variable(&self, name: &str) -> Option<&VariableInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(var_info) = scope.get(name) {
                return Some(var_info);
            }
        }
        None
    }
}

impl Default for TurbulanceCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled Turbulance program
#[derive(Debug, Clone)]
pub struct TurbulanceProgram {
    /// Program instructions
    pub instructions: Vec<Instruction>,
    
    /// Function declarations
    pub functions: Vec<String>,
    
    /// Proposition declarations
    pub propositions: Vec<String>,
    
    /// Goal declarations
    pub goals: Vec<String>,
}

impl TurbulanceProgram {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            functions: Vec::new(),
            propositions: Vec::new(),
            goals: Vec::new(),
        }
    }
    
    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
    
    pub fn add_function_declaration(&mut self, name: String) {
        self.functions.push(name);
    }
    
    pub fn add_proposition_declaration(&mut self, name: String) {
        self.propositions.push(name);
    }
    
    pub fn add_goal_declaration(&mut self, name: String) {
        self.goals.push(name);
    }
}

/// Compiled instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    VariableDeclaration { name: String, value: Box<Operation> },
    Assignment { target: String, value: Box<Operation> },
    FunctionDeclaration { name: String, parameter_names: Vec<String>, body: Vec<Instruction> },
    PropositionDeclaration { name: String, motions: Vec<Instruction>, evaluations: Vec<Instruction> },
    GoalDeclaration { 
        name: String, 
        description: String,
        success_threshold: Option<f64>,
        keywords: Vec<String>,
        domain: Option<String>,
        audience: Option<String>,
        priority: Option<crate::turbulance::ast::Priority>,
        deadline: Option<String>,
        metrics: HashMap<String, f64>,
    },
    MotionDeclaration { name: String, description: String },
    Support { motion: String, weight: Option<Box<Operation>> },
    Contradict { motion: String, weight: Option<Box<Operation>> },
    Conditional { condition: Box<Operation>, then_branch: Vec<Instruction>, else_branch: Option<Vec<Instruction>> },
    EvaluationCondition { condition: Box<Operation>, action: EvaluationActionInstruction },
    Return { value: Option<Box<Operation>> },
    Expression(Box<Operation>),
}

/// Evaluation action instruction
#[derive(Debug, Clone)]
pub enum EvaluationActionInstruction {
    Support { motion: String, weight: Option<Box<Operation>> },
    Contradict { motion: String, weight: Option<Box<Operation>> },
    Collect { evidence_type: String },
    Flag { uncertainty: String },
}

/// Compiled operation
#[derive(Debug, Clone)]
pub enum Operation {
    LoadConstant(Value),
    LoadVariable(String),
    LoadFunction(String),
    BinaryOperation { left: Box<Operation>, operator: BinaryOp, right: Box<Operation> },
    UnaryOperation { operator: UnaryOp, operand: Box<Operation> },
    FunctionCall { function: String, arguments: Vec<Operation> },
    MethodCall { object: Box<Operation>, method: String, arguments: Vec<Operation> },
    CreateArray(Vec<Operation>),
    CreateObject(Vec<(String, Operation)>),
}

/// Runtime values
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    Null,
}

/// Binary operations
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add, Subtract, Multiply, Divide, Modulo, Power,
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    And, Or, Matches, Contains,
}

/// Unary operations
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Not, Minus, Plus,
}

/// Variable information
#[derive(Debug, Clone)]
struct VariableInfo {
    name: String,
    var_type: Option<String>,
    is_mutable: bool,
}

/// Function information
#[derive(Debug, Clone)]
struct FunctionInfo {
    name: String,
    parameter_types: Vec<Option<String>>,
    return_type: Option<String>,
    is_builtin: bool,
}

/// Proposition information
#[derive(Debug, Clone)]
struct PropositionInfo {
    name: String,
    motion_names: Vec<String>,
}

/// Goal information
#[derive(Debug, Clone)]
struct GoalInfo {
    name: String,
    description: String,
}

/// Compiler errors
#[derive(Debug, Clone)]
pub enum CompilerError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeMismatch { expected: String, found: String },
    UnsupportedExpression(String),
    InvalidOperation(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::UndefinedVariable(name) => {
                write!(f, "Undefined variable: {}", name)
            }
            CompilerError::UndefinedFunction(name) => {
                write!(f, "Undefined function: {}", name)
            }
            CompilerError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            CompilerError::UnsupportedExpression(expr) => {
                write!(f, "Unsupported expression: {}", expr)
            }
            CompilerError::InvalidOperation(op) => {
                write!(f, "Invalid operation: {}", op)
            }
        }
    }
}

impl std::error::Error for CompilerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::ast::{TurbulanceAst, AstNode, LiteralValue};

    #[test]
    fn test_compile_simple_variable() {
        let mut compiler = TurbulanceCompiler::new();
        
        let mut ast = TurbulanceAst::new();
        ast.add_statement(AstNode::VariableDeclaration {
            name: "x".to_string(),
            value: Some(Box::new(AstNode::Literal {
                value: LiteralValue::Integer(42),
            })),
            type_annotation: None,
        });
        
        let program = compiler.compile(ast).unwrap();
        assert_eq!(program.instructions.len(), 1);
        
        match &program.instructions[0] {
            Instruction::VariableDeclaration { name, .. } => {
                assert_eq!(name, "x");
            }
            _ => panic!("Expected variable declaration"),
        }
    }

    #[test]
    fn test_compile_binary_expression() {
        let mut compiler = TurbulanceCompiler::new();
        
        let expr = AstNode::BinaryOperation {
            left: Box::new(AstNode::Literal { value: LiteralValue::Integer(1) }),
            operator: BinaryOperator::Add,
            right: Box::new(AstNode::Literal { value: LiteralValue::Integer(2) }),
        };
        
        let operation = compiler.compile_expression(&expr).unwrap();
        
        match operation {
            Operation::BinaryOperation { operator, .. } => {
                match operator {
                    BinaryOp::Add => {},
                    _ => panic!("Expected Add operator"),
                }
            }
            _ => panic!("Expected binary operation"),
        }
    }
} 