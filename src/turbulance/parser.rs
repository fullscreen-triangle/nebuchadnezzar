//! # Turbulance Parser
//!
//! Parses Turbulance tokens into an Abstract Syntax Tree (AST).

use crate::turbulance::lexer::{Token, TokenType};
use crate::turbulance::ast::{
    TurbulanceAst, AstNode, Parameter, PatternMatch, MotionDeclaration, Requirements, Criteria,
    EvaluationBlock, EvaluationCondition, EvaluationAction, EvidenceSource, CollectionSpec,
    ProcessingStep, StorageSpec, Priority, PatternCategory, PatternDefinition, MatchingRules,
    PatternRelationship, TimeScope, TemporalPattern, TemporalPatternType, TemporalOperation,
    TrackingSpec, EvaluationSpec, AdaptationRule, BinaryOperator, UnaryOperator, LiteralValue,
    CatchBlock,
};
use std::collections::HashMap;
use std::fmt;

/// Turbulance parser
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    /// Create a new parser instance
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            current: 0,
        }
    }

    /// Parse tokens into an AST
    pub fn parse(&mut self, tokens: Vec<Token>) -> Result<TurbulanceAst, ParseError> {
        self.tokens = tokens;
        self.current = 0;
        
        let mut ast = TurbulanceAst::new();
        
        while !self.is_at_end() {
            let statement = self.parse_statement()?;
            ast.add_statement(statement);
        }
        
        Ok(ast)
    }
    
    /// Parse a top-level statement
    fn parse_statement(&mut self) -> Result<AstNode, ParseError> {
        match self.peek().token_type {
            TokenType::Item => self.parse_variable_declaration(),
            TokenType::Funxn => self.parse_function_declaration(),
            TokenType::Proposition => self.parse_proposition(),
            TokenType::Motion => self.parse_motion(),
            TokenType::Evidence => self.parse_evidence(),
            TokenType::Goal => self.parse_goal(),
            TokenType::PatternRegistry => self.parse_pattern_registry(),
            TokenType::Temporal => self.parse_temporal(),
            TokenType::Metacognitive => self.parse_metacognitive(),
            TokenType::Given => self.parse_given(),
            TokenType::Within => self.parse_within(),
            TokenType::Considering => self.parse_considering(),
            TokenType::For => self.parse_for_each(),
            TokenType::While => self.parse_while(),
            TokenType::Return => self.parse_return(),
            TokenType::Parallel => self.parse_parallel(),
            TokenType::Async => self.parse_async(),
            TokenType::Try => self.parse_try(),
            TokenType::Import => self.parse_import(),
            TokenType::Support => self.parse_support(),
            TokenType::Contradict => self.parse_contradict(),
            _ => self.parse_expression_statement(),
        }
    }
    
    /// Parse variable declaration: item name = value
    fn parse_variable_declaration(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Item, "Expected 'item'")?;
        
        let name = self.consume(TokenType::Identifier, "Expected variable name")?.value;
        
        let mut type_annotation = None;
        if self.match_token(&TokenType::Colon) {
            type_annotation = Some(self.consume(TokenType::Identifier, "Expected type")?.value);
        }
        
        let mut value = None;
        if self.match_token(&TokenType::Assign) {
            value = Some(Box::new(self.parse_expression()?));
        }
        
        Ok(AstNode::VariableDeclaration {
            name,
            value,
            type_annotation,
        })
    }
    
    /// Parse function declaration: funxn name(params) -> return_type: body
    fn parse_function_declaration(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Funxn, "Expected 'funxn'")?;
        
        let name = self.consume(TokenType::Identifier, "Expected function name")?.value;
        
        self.consume(TokenType::LeftParen, "Expected '('")?;
        
        let mut parameters = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                let param_name = self.consume(TokenType::Identifier, "Expected parameter name")?.value;
                let mut type_annotation = None;
                let mut default_value = None;
                
                if self.match_token(&TokenType::Colon) {
                    type_annotation = Some(self.consume(TokenType::Identifier, "Expected type")?.value);
                }
                
                if self.match_token(&TokenType::Assign) {
                    default_value = Some(self.parse_expression()?);
                }
                
                parameters.push(Parameter {
                    name: param_name,
                    type_annotation,
                    default_value,
                });
                
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(TokenType::RightParen, "Expected ')'")?;
        
        let mut return_type = None;
        if self.match_token(&TokenType::Arrow) {
            return_type = Some(self.consume(TokenType::Identifier, "Expected return type")?.value);
        }
        
        self.consume(TokenType::Colon, "Expected ':'")?;
        
        let body = self.parse_block()?;
        
        Ok(AstNode::FunctionDeclaration {
            name,
            parameters,
            return_type,
            body,
        })
    }
    
    /// Parse proposition: proposition Name: motions...
    fn parse_proposition(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Proposition, "Expected 'proposition'")?;
        
        let name = self.consume(TokenType::Identifier, "Expected proposition name")?.value;
        
        self.consume(TokenType::Colon, "Expected ':'")?;
        
        let mut context = None;
        let mut motions = Vec::new();
        let mut evaluation_blocks = Vec::new();
        
        // Parse the proposition body
        while !self.is_at_end() && self.current_indent_level() > 0 {
            match self.peek().token_type {
                TokenType::Motion => {
                    let motion = self.parse_motion_declaration()?;
                    motions.push(motion);
                }
                TokenType::Within => {
                    let eval_block = self.parse_evaluation_block()?;
                    evaluation_blocks.push(eval_block);
                }
                _ => break,
            }
        }
        
        Ok(AstNode::Proposition {
            name,
            context,
            motions,
            evaluation_blocks,
        })
    }
    
    /// Parse motion declaration within a proposition
    fn parse_motion_declaration(&mut self) -> Result<MotionDeclaration, ParseError> {
        self.consume(TokenType::Motion, "Expected 'motion'")?;
        
        let name = self.consume(TokenType::Identifier, "Expected motion name")?.value;
        
        self.consume(TokenType::LeftParen, "Expected '('")?;
        let description = self.consume(TokenType::String, "Expected description string")?.value;
        self.consume(TokenType::RightParen, "Expected ')'")?;
        
        let mut requirements = None;
        let mut criteria = None;
        let mut patterns = Vec::new();
        
        // Parse optional motion properties
        if self.match_token(&TokenType::Colon) {
            while !self.is_at_end() && self.current_indent_level() > 1 {
                match self.peek().token_type {
                    TokenType::Identifier => {
                        let key = self.advance().value;
                        match key.as_str() {
                            "requires" => {
                                requirements = Some(self.parse_requirements()?);
                            }
                            "criteria" => {
                                criteria = Some(self.parse_criteria()?);
                            }
                            "patterns" => {
                                patterns = self.parse_string_array()?;
                            }
                            _ => break,
                        }
                    }
                    _ => break,
                }
            }
        }
        
        Ok(MotionDeclaration {
            name,
            description,
            requirements,
            criteria,
            patterns,
        })
    }
    
    /// Parse evaluation block: within scope: conditions...
    fn parse_evaluation_block(&mut self) -> Result<EvaluationBlock, ParseError> {
        self.consume(TokenType::Within, "Expected 'within'")?;
        
        let scope = Some(self.parse_expression()?);
        
        self.consume(TokenType::Colon, "Expected ':'")?;
        
        let mut conditions = Vec::new();
        
        while !self.is_at_end() && self.current_indent_level() > 1 {
            if self.match_token(&TokenType::Given) {
                let condition = self.parse_expression()?;
                self.consume(TokenType::Colon, "Expected ':'")?;
                
                let action = self.parse_evaluation_action()?;
                
                conditions.push(EvaluationCondition { condition, action });
            } else {
                break;
            }
        }
        
        Ok(EvaluationBlock { scope, conditions })
    }
    
    /// Parse evaluation action (support/contradict/etc.)
    fn parse_evaluation_action(&mut self) -> Result<EvaluationAction, ParseError> {
        match self.peek().token_type {
            TokenType::Support => {
                self.advance();
                let motion = self.consume(TokenType::Identifier, "Expected motion name")?.value;
                let weight = if self.match_token(&TokenType::With) {
                    self.consume(TokenType::Identifier, "Expected weight")?; // "weight"
                    self.consume(TokenType::LeftParen, "Expected '('")?;
                    let weight_expr = self.parse_expression()?;
                    self.consume(TokenType::RightParen, "Expected ')'")?;
                    Some(weight_expr)
                } else {
                    None
                };
                Ok(EvaluationAction::Support { motion, weight })
            }
            TokenType::Contradict => {
                self.advance();
                let motion = self.consume(TokenType::Identifier, "Expected motion name")?.value;
                let weight = if self.match_token(&TokenType::With) {
                    self.consume(TokenType::Identifier, "Expected weight")?; // "weight"
                    self.consume(TokenType::LeftParen, "Expected '('")?;
                    let weight_expr = self.parse_expression()?;
                    self.consume(TokenType::RightParen, "Expected ')'")?;
                    Some(weight_expr)
                } else {
                    None
                };
                Ok(EvaluationAction::Contradict { motion, weight })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "support or contradict".to_string(),
                found: self.peek().clone(),
            }),
        }
    }
    
    /// Parse goal declaration
    fn parse_goal(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Goal, "Expected 'goal'")?;
        
        let name = self.consume(TokenType::Identifier, "Expected goal name")?.value;
        
        self.consume(TokenType::Assign, "Expected '='")?;
        
        // Parse Goal.new("description") { properties }
        self.consume(TokenType::Identifier, "Expected 'Goal'")?; // Goal
        self.consume(TokenType::Dot, "Expected '.'")?;
        self.consume(TokenType::Identifier, "Expected 'new'")?; // new
        self.consume(TokenType::LeftParen, "Expected '('")?;
        let description = self.consume(TokenType::String, "Expected description")?.value;
        self.consume(TokenType::RightParen, "Expected ')'")?;
        
        let mut success_threshold = None;
        let mut keywords = Vec::new();
        let mut domain = None;
        let mut audience = None;
        let mut priority = None;
        let mut deadline = None;
        let mut metrics = None;
        
        if self.match_token(&TokenType::LeftBrace) {
            while !self.check(&TokenType::RightBrace) {
                let key = self.consume(TokenType::Identifier, "Expected property name")?.value;
                self.consume(TokenType::Colon, "Expected ':'")?;
                
                match key.as_str() {
                    "success_threshold" => {
                        success_threshold = Some(self.parse_float_literal()?);
                    }
                    "keywords" => {
                        keywords = self.parse_string_array()?;
                    }
                    "domain" => {
                        domain = Some(self.consume(TokenType::String, "Expected domain string")?.value);
                    }
                    "audience" => {
                        audience = Some(self.consume(TokenType::String, "Expected audience string")?.value);
                    }
                    "priority" => {
                        priority = Some(self.parse_priority()?);
                    }
                    "deadline" => {
                        deadline = Some(self.consume(TokenType::String, "Expected deadline string")?.value);
                    }
                    "metrics" => {
                        metrics = Some(self.parse_metrics_object()?);
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "valid goal property".to_string(),
                            found: self.peek().clone(),
                        });
                    }
                }
                
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            
            self.consume(TokenType::RightBrace, "Expected '}'")?;
        }
        
        Ok(AstNode::Goal {
            name,
            description,
            success_threshold,
            keywords,
            domain,
            audience,
            priority,
            deadline,
            metrics,
        })
    }
    
    /// Parse support statement
    fn parse_support(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Support, "Expected 'support'")?;
        let motion = self.consume(TokenType::Identifier, "Expected motion name")?.value;
        
        let weight = if self.match_token(&TokenType::With) {
            self.consume(TokenType::Identifier, "Expected 'weight'")?;
            self.consume(TokenType::LeftParen, "Expected '('")?;
            let weight_expr = self.parse_expression()?;
            self.consume(TokenType::RightParen, "Expected ')'")?;
            Some(Box::new(weight_expr))
        } else {
            None
        };
        
        Ok(AstNode::Support { motion, weight })
    }
    
    /// Parse contradict statement
    fn parse_contradict(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Contradict, "Expected 'contradict'")?;
        let motion = self.consume(TokenType::Identifier, "Expected motion name")?.value;
        
        let weight = if self.match_token(&TokenType::With) {
            self.consume(TokenType::Identifier, "Expected 'weight'")?;
            self.consume(TokenType::LeftParen, "Expected '('")?;
            let weight_expr = self.parse_expression()?;
            self.consume(TokenType::RightParen, "Expected ')'")?;
            Some(Box::new(weight_expr))
        } else {
            None
        };
        
        Ok(AstNode::Contradict { motion, weight })
    }
    
    /// Parse expression
    fn parse_expression(&mut self) -> Result<AstNode, ParseError> {
        self.parse_logical_or()
    }
    
    /// Parse logical OR expression
    fn parse_logical_or(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_logical_and()?;
        
        while self.match_token(&TokenType::Or) {
            let right = self.parse_logical_and()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse logical AND expression
    fn parse_logical_and(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_equality()?;
        
        while self.match_token(&TokenType::And) {
            let right = self.parse_equality()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse equality expression
    fn parse_equality(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_comparison()?;
        
        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_comparison()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse comparison expression
    fn parse_comparison(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_term()?;
        
        while let Some(op) = self.match_comparison_operator() {
            let right = self.parse_term()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse term (addition/subtraction)
    fn parse_term(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_factor()?;
        
        while let Some(op) = self.match_term_operator() {
            let right = self.parse_factor()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse factor (multiplication/division)
    fn parse_factor(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_unary()?;
        
        while let Some(op) = self.match_factor_operator() {
            let right = self.parse_unary()?;
            expr = AstNode::BinaryOperation {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse unary expression
    fn parse_unary(&mut self) -> Result<AstNode, ParseError> {
        if let Some(op) = self.match_unary_operator() {
            let operand = self.parse_unary()?;
            Ok(AstNode::UnaryOperation {
                operator: op,
                operand: Box::new(operand),
            })
        } else {
            self.parse_call()
        }
    }
    
    /// Parse function/method call
    fn parse_call(&mut self) -> Result<AstNode, ParseError> {
        let mut expr = self.parse_primary()?;
        
        loop {
            if self.match_token(&TokenType::LeftParen) {
                expr = self.finish_call(expr)?;
            } else if self.match_token(&TokenType::Dot) {
                let method = self.consume(TokenType::Identifier, "Expected method name")?.value;
                if self.match_token(&TokenType::LeftParen) {
                    let arguments = self.parse_arguments()?;
                    expr = AstNode::MethodCall {
                        object: Box::new(expr),
                        method,
                        arguments,
                    };
                } else {
                    // Property access - treat as method call with no arguments
                    expr = AstNode::MethodCall {
                        object: Box::new(expr),
                        method,
                        arguments: Vec::new(),
                    };
                }
            } else {
                break;
            }
        }
        
        Ok(expr)
    }
    
    /// Parse primary expression
    fn parse_primary(&mut self) -> Result<AstNode, ParseError> {
        match self.peek().token_type {
            TokenType::True => {
                self.advance();
                Ok(AstNode::Literal {
                    value: LiteralValue::Boolean(true),
                })
            }
            TokenType::False => {
                self.advance();
                Ok(AstNode::Literal {
                    value: LiteralValue::Boolean(false),
                })
            }
            TokenType::Integer => {
                let value = self.advance().value;
                let int_val = value.parse::<i64>().map_err(|_| ParseError::InvalidNumber { value })?;
                Ok(AstNode::Literal {
                    value: LiteralValue::Integer(int_val),
                })
            }
            TokenType::Float => {
                let value = self.advance().value;
                let float_val = value.parse::<f64>().map_err(|_| ParseError::InvalidNumber { value })?;
                Ok(AstNode::Literal {
                    value: LiteralValue::Float(float_val),
                })
            }
            TokenType::String => {
                let value = self.advance().value;
                Ok(AstNode::Literal {
                    value: LiteralValue::String(value),
                })
            }
            TokenType::Char => {
                let value = self.advance().value;
                let char_val = value.chars().next().unwrap_or('\0');
                Ok(AstNode::Literal {
                    value: LiteralValue::Char(char_val),
                })
            }
            TokenType::Identifier => {
                let name = self.advance().value;
                Ok(AstNode::Identifier { name })
            }
            TokenType::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(TokenType::RightParen, "Expected ')'")?;
                Ok(expr)
            }
            TokenType::LeftBracket => {
                self.advance();
                let mut elements = Vec::new();
                
                if !self.check(&TokenType::RightBracket) {
                    loop {
                        elements.push(self.parse_expression()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(TokenType::RightBracket, "Expected ']'")?;
                Ok(AstNode::Array { elements })
            }
            TokenType::LeftBrace => {
                self.advance();
                let mut fields = Vec::new();
                
                if !self.check(&TokenType::RightBrace) {
                    loop {
                        let key = self.consume(TokenType::String, "Expected string key")?.value;
                        self.consume(TokenType::Colon, "Expected ':'")?;
                        let value = self.parse_expression()?;
                        fields.push((key, value));
                        
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.consume(TokenType::RightBrace, "Expected '}'")?;
                Ok(AstNode::Object { fields })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: self.peek().clone(),
            }),
        }
    }
    
    // Helper methods
    
    /// Stub implementations for methods that would be too complex for this example
    fn parse_motion(&mut self) -> Result<AstNode, ParseError> {
        // Simplified motion parsing - would need full implementation
        Ok(AstNode::Identifier { name: "motion_stub".to_string() })
    }
    
    fn parse_evidence(&mut self) -> Result<AstNode, ParseError> {
        // Simplified evidence parsing - would need full implementation
        Ok(AstNode::Identifier { name: "evidence_stub".to_string() })
    }
    
    fn parse_pattern_registry(&mut self) -> Result<AstNode, ParseError> {
        // Simplified pattern registry parsing - would need full implementation
        Ok(AstNode::Identifier { name: "pattern_registry_stub".to_string() })
    }
    
    fn parse_temporal(&mut self) -> Result<AstNode, ParseError> {
        // Simplified temporal parsing - would need full implementation
        Ok(AstNode::Identifier { name: "temporal_stub".to_string() })
    }
    
    fn parse_metacognitive(&mut self) -> Result<AstNode, ParseError> {
        // Simplified metacognitive parsing - would need full implementation
        Ok(AstNode::Identifier { name: "metacognitive_stub".to_string() })
    }
    
    fn parse_given(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Given, "Expected 'given'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenType::Colon, "Expected ':'")?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.match_token(&TokenType::Otherwise) {
            self.consume(TokenType::Colon, "Expected ':'")?;
            Some(self.parse_block()?)
        } else {
            None
        };
        
        Ok(AstNode::Given {
            condition: Box::new(condition),
            then_branch,
            else_branch,
        })
    }
    
    fn parse_within(&mut self) -> Result<AstNode, ParseError> {
        // Simplified within parsing - would need full implementation
        Ok(AstNode::Identifier { name: "within_stub".to_string() })
    }
    
    fn parse_considering(&mut self) -> Result<AstNode, ParseError> {
        // Simplified considering parsing - would need full implementation
        Ok(AstNode::Identifier { name: "considering_stub".to_string() })
    }
    
    fn parse_for_each(&mut self) -> Result<AstNode, ParseError> {
        // Simplified for each parsing - would need full implementation
        Ok(AstNode::Identifier { name: "for_each_stub".to_string() })
    }
    
    fn parse_while(&mut self) -> Result<AstNode, ParseError> {
        // Simplified while parsing - would need full implementation
        Ok(AstNode::Identifier { name: "while_stub".to_string() })
    }
    
    fn parse_return(&mut self) -> Result<AstNode, ParseError> {
        self.consume(TokenType::Return, "Expected 'return'")?;
        let value = if self.is_at_end() || self.peek().token_type == TokenType::Semicolon {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        
        Ok(AstNode::Return { value })
    }
    
    fn parse_parallel(&mut self) -> Result<AstNode, ParseError> {
        // Simplified parallel parsing - would need full implementation
        Ok(AstNode::Identifier { name: "parallel_stub".to_string() })
    }
    
    fn parse_async(&mut self) -> Result<AstNode, ParseError> {
        // Simplified async parsing - would need full implementation
        Ok(AstNode::Identifier { name: "async_stub".to_string() })
    }
    
    fn parse_try(&mut self) -> Result<AstNode, ParseError> {
        // Simplified try parsing - would need full implementation
        Ok(AstNode::Identifier { name: "try_stub".to_string() })
    }
    
    fn parse_import(&mut self) -> Result<AstNode, ParseError> {
        // Simplified import parsing - would need full implementation
        Ok(AstNode::Identifier { name: "import_stub".to_string() })
    }
    
    fn parse_expression_statement(&mut self) -> Result<AstNode, ParseError> {
        self.parse_expression()
    }
    
    fn parse_block(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut statements = Vec::new();
        // Simplified block parsing - would need proper indentation handling
        statements.push(self.parse_statement()?);
        Ok(statements)
    }
    
    fn parse_requirements(&mut self) -> Result<Requirements, ParseError> {
        // Simplified requirements parsing
        Ok(Requirements {
            data_types: Vec::new(),
            min_data_points: None,
            quality_threshold: None,
        })
    }
    
    fn parse_criteria(&mut self) -> Result<Criteria, ParseError> {
        // Simplified criteria parsing
        Ok(Criteria {
            conditions: Vec::new(),
            thresholds: HashMap::new(),
        })
    }
    
    fn parse_string_array(&mut self) -> Result<Vec<String>, ParseError> {
        self.consume(TokenType::LeftBracket, "Expected '['")?;
        let mut strings = Vec::new();
        
        if !self.check(&TokenType::RightBracket) {
            loop {
                strings.push(self.consume(TokenType::String, "Expected string")?.value);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(TokenType::RightBracket, "Expected ']'")?;
        Ok(strings)
    }
    
    fn parse_float_literal(&mut self) -> Result<f64, ParseError> {
        let token = self.advance();
        match token.token_type {
            TokenType::Float => {
                token.value.parse::<f64>().map_err(|_| ParseError::InvalidNumber { value: token.value })
            }
            TokenType::Integer => {
                token.value.parse::<f64>().map_err(|_| ParseError::InvalidNumber { value: token.value })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "number".to_string(),
                found: token,
            }),
        }
    }
    
    fn parse_priority(&mut self) -> Result<Priority, ParseError> {
        let name = self.consume(TokenType::Identifier, "Expected priority")?.value;
        match name.as_str() {
            "Low" => Ok(Priority::Low),
            "Medium" => Ok(Priority::Medium),
            "High" => Ok(Priority::High),
            "Critical" => Ok(Priority::Critical),
            _ => Err(ParseError::InvalidPriority { value: name }),
        }
    }
    
    fn parse_metrics_object(&mut self) -> Result<HashMap<String, f64>, ParseError> {
        self.consume(TokenType::LeftBrace, "Expected '{'")?;
        let mut metrics = HashMap::new();
        
        if !self.check(&TokenType::RightBrace) {
            loop {
                let key = self.consume(TokenType::Identifier, "Expected metric name")?.value;
                self.consume(TokenType::Colon, "Expected ':'")?;
                let value = self.parse_float_literal()?;
                metrics.insert(key, value);
                
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(TokenType::RightBrace, "Expected '}'")?;
        Ok(metrics)
    }
    
    fn finish_call(&mut self, callee: AstNode) -> Result<AstNode, ParseError> {
        let arguments = self.parse_arguments()?;
        
        match callee {
            AstNode::Identifier { name } => Ok(AstNode::FunctionCall {
                function: name,
                arguments,
            }),
            _ => Err(ParseError::InvalidCallTarget),
        }
    }
    
    fn parse_arguments(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut arguments = Vec::new();
        
        if !self.check(&TokenType::RightParen) {
            loop {
                arguments.push(self.parse_expression()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        
        self.consume(TokenType::RightParen, "Expected ')'")?;
        Ok(arguments)
    }
    
    fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Equals => {
                self.advance();
                Some(BinaryOperator::Equal)
            }
            TokenType::NotEquals => {
                self.advance();
                Some(BinaryOperator::NotEqual)
            }
            _ => None,
        }
    }
    
    fn match_comparison_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Greater => {
                self.advance();
                Some(BinaryOperator::Greater)
            }
            TokenType::GreaterEquals => {
                self.advance();
                Some(BinaryOperator::GreaterEqual)
            }
            TokenType::Less => {
                self.advance();
                Some(BinaryOperator::Less)
            }
            TokenType::LessEquals => {
                self.advance();
                Some(BinaryOperator::LessEqual)
            }
            _ => None,
        }
    }
    
    fn match_term_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Minus => {
                self.advance();
                Some(BinaryOperator::Subtract)
            }
            TokenType::Plus => {
                self.advance();
                Some(BinaryOperator::Add)
            }
            _ => None,
        }
    }
    
    fn match_factor_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Slash => {
                self.advance();
                Some(BinaryOperator::Divide)
            }
            TokenType::Star => {
                self.advance();
                Some(BinaryOperator::Multiply)
            }
            TokenType::Percent => {
                self.advance();
                Some(BinaryOperator::Modulo)
            }
            TokenType::Power => {
                self.advance();
                Some(BinaryOperator::Power)
            }
            _ => None,
        }
    }
    
    fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        match self.peek().token_type {
            TokenType::Bang => {
                self.advance();
                Some(UnaryOperator::Not)
            }
            TokenType::Minus => {
                self.advance();
                Some(UnaryOperator::Minus)
            }
            TokenType::Plus => {
                self.advance();
                Some(UnaryOperator::Plus)
            }
            _ => None,
        }
    }
    
    // Utility methods
    
    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }
    
    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.tokens[self.current - 1].clone()
    }
    
    fn is_at_end(&self) -> bool {
        self.peek().token_type == TokenType::Eof
    }
    
    fn check(&self, token_type: &TokenType) -> bool {
        if self.is_at_end() {
            false
        } else {
            &self.peek().token_type == token_type
        }
    }
    
    fn match_token(&mut self, token_type: &TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    fn consume(&mut self, token_type: TokenType, message: &str) -> Result<Token, ParseError> {
        if self.check(&token_type) {
            Ok(self.advance())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: message.to_string(),
                found: self.peek().clone(),
            })
        }
    }
    
    fn current_indent_level(&self) -> usize {
        // This is a simplified indentation check
        // In a real implementation, you'd track indentation properly
        1
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse errors
#[derive(Debug, Clone)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: Token,
    },
    UnexpectedEndOfInput,
    InvalidNumber {
        value: String,
    },
    InvalidPriority {
        value: String,
    },
    InvalidCallTarget,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken { expected, found } => {
                write!(f, "Expected {}, found {} at line {}, column {}", 
                       expected, found.value, found.line, found.column)
            }
            ParseError::UnexpectedEndOfInput => {
                write!(f, "Unexpected end of input")
            }
            ParseError::InvalidNumber { value } => {
                write!(f, "Invalid number: {}", value)
            }
            ParseError::InvalidPriority { value } => {
                write!(f, "Invalid priority: {}", value)
            }
            ParseError::InvalidCallTarget => {
                write!(f, "Invalid function call target")
            }
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::lexer::Lexer;

    #[test]
    fn test_parse_simple_variable() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("item x = 42").unwrap();
        
        let mut parser = Parser::new();
        let ast = parser.parse(tokens).unwrap();
        
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            AstNode::VariableDeclaration { name, .. } => {
                assert_eq!(name, "x");
            }
            _ => panic!("Expected variable declaration"),
        }
    }

    #[test]
    fn test_parse_simple_function() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("funxn test(): return 1").unwrap();
        
        let mut parser = Parser::new();
        let ast = parser.parse(tokens).unwrap();
        
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            AstNode::FunctionDeclaration { name, .. } => {
                assert_eq!(name, "test");
            }
            _ => panic!("Expected function declaration"),
        }
    }

    #[test]
    fn test_parse_proposition() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("proposition TestProp: motion Test(\"description\")").unwrap();
        
        let mut parser = Parser::new();
        let ast = parser.parse(tokens).unwrap();
        
        assert_eq!(ast.statements.len(), 1);
        match &ast.statements[0] {
            AstNode::Proposition { name, .. } => {
                assert_eq!(name, "TestProp");
            }
            _ => panic!("Expected proposition"),
        }
    }
} 