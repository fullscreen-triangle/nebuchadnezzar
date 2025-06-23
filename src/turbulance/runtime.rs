//! # Turbulance Runtime
//!
//! Executes compiled Turbulance programs within the Nebuchadnezzar environment.

use crate::turbulance::compiler::{
    TurbulanceProgram, Instruction, Operation, Value, BinaryOp, UnaryOp, EvaluationActionInstruction,
};
use crate::turbulance::integration::NebuIntegration;
use crate::turbulance::{
    TurbulanceResult, TurbulanceValue, PropositionResult, MotionResult, MotionStatus,
    GoalResult, GoalStatus, EvidenceResult, ExecutionMetrics,
};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Turbulance runtime system
pub struct TurbulanceRuntime {
    /// Variable stack for execution
    variables: HashMap<String, TurbulanceValue>,
    
    /// Function definitions
    functions: HashMap<String, FunctionDefinition>,
    
    /// Proposition definitions
    propositions: HashMap<String, PropositionDefinition>,
    
    /// Goal tracking
    goals: HashMap<String, GoalDefinition>,
    
    /// Integration with Nebuchadnezzar
    integration: NebuIntegration,
    
    /// Call stack for function execution
    call_stack: Vec<CallFrame>,
    
    /// Current execution metrics
    metrics: ExecutionMetrics,
}

impl TurbulanceRuntime {
    /// Create a new runtime instance
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            propositions: HashMap::new(),
            goals: HashMap::new(),
            integration: NebuIntegration::new(),
            call_stack: Vec::new(),
            metrics: ExecutionMetrics::default(),
        }
    }

    /// Execute a compiled program
    pub fn execute(&mut self, program: TurbulanceProgram) -> Result<TurbulanceResult, RuntimeError> {
        let start_time = Instant::now();
        
        // Initialize builtin functions
        self.initialize_builtins();
        
        // Execute instructions
        for instruction in &program.instructions {
            self.execute_instruction(instruction)?;
        }
        
        // Calculate execution time
        self.metrics.execute_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Collect results
        let result = TurbulanceResult {
            propositions: self.collect_proposition_results(),
            goals: self.collect_goal_results(),
            evidence: self.collect_evidence_results(),
            variables: self.collect_variable_results(),
            metrics: self.metrics.clone(),
        };
        
        Ok(result)
    }
    
    /// Execute a single instruction
    fn execute_instruction(&mut self, instruction: &Instruction) -> Result<TurbulanceValue, RuntimeError> {
        self.metrics.operations_count += 1;
        
        match instruction {
            Instruction::VariableDeclaration { name, value } => {
                let val = self.execute_operation(value)?;
                self.variables.insert(name.clone(), val.clone());
                self.integration.set_variable(name, val.clone());
                Ok(val)
            }
            
            Instruction::Assignment { target, value } => {
                let val = self.execute_operation(value)?;
                self.variables.insert(target.clone(), val.clone());
                self.integration.set_variable(target, val.clone());
                Ok(val)
            }
            
            Instruction::FunctionDeclaration { name, parameter_names, body } => {
                let func_def = FunctionDefinition {
                    name: name.clone(),
                    parameters: parameter_names.clone(),
                    body: body.clone(),
                    is_builtin: false,
                };
                self.functions.insert(name.clone(), func_def);
                Ok(TurbulanceValue::String(format!("Function {} defined", name)))
            }
            
            Instruction::PropositionDeclaration { name, motions, evaluations } => {
                let mut motion_names = Vec::new();
                for motion_instr in motions {
                    if let Instruction::MotionDeclaration { name: motion_name, .. } = motion_instr {
                        motion_names.push(motion_name.clone());
                    }
                }
                
                let prop_def = PropositionDefinition {
                    name: name.clone(),
                    motions: motion_names.clone(),
                    evaluations: evaluations.clone(),
                };
                self.propositions.insert(name.clone(), prop_def);
                
                // Initialize experiment in Nebuchadnezzar
                let experiment_id = self.integration.initialize_experiment(name, &format!("Proposition: {}", name))
                    .map_err(|e| RuntimeError::IntegrationError(e.to_string()))?;
                
                Ok(TurbulanceValue::String(format!("Proposition {} defined with experiment {}", name, experiment_id)))
            }
            
            Instruction::GoalDeclaration { name, description, success_threshold, keywords, domain, audience, priority, deadline, metrics } => {
                let goal_def = GoalDefinition {
                    name: name.clone(),
                    description: description.clone(),
                    success_threshold: success_threshold.unwrap_or(0.8),
                    keywords: keywords.clone(),
                    domain: domain.clone(),
                    audience: audience.clone(),
                    priority: priority.clone(),
                    deadline: deadline.clone(),
                    metrics: metrics.clone(),
                    current_progress: 0.0,
                    status: GoalStatus::Active,
                };
                self.goals.insert(name.clone(), goal_def);
                
                Ok(TurbulanceValue::String(format!("Goal {} defined", name)))
            }
            
            Instruction::Support { motion, weight } => {
                let weight_val = if let Some(w) = weight {
                    match self.execute_operation(w)? {
                        TurbulanceValue::Float(f) => f,
                        TurbulanceValue::Integer(i) => i as f64,
                        _ => 1.0,
                    }
                } else {
                    1.0
                };
                
                // Update motion support in current proposition
                self.update_motion_support(motion, weight_val, true)?;
                
                Ok(TurbulanceValue::String(format!("Motion {} supported with weight {}", motion, weight_val)))
            }
            
            Instruction::Contradict { motion, weight } => {
                let weight_val = if let Some(w) = weight {
                    match self.execute_operation(w)? {
                        TurbulanceValue::Float(f) => f,
                        TurbulanceValue::Integer(i) => i as f64,
                        _ => 1.0,
                    }
                } else {
                    1.0
                };
                
                // Update motion support in current proposition
                self.update_motion_support(motion, -weight_val, false)?;
                
                Ok(TurbulanceValue::String(format!("Motion {} contradicted with weight {}", motion, weight_val)))
            }
            
            Instruction::Conditional { condition, then_branch, else_branch } => {
                let condition_result = self.execute_operation(condition)?;
                let is_true = self.value_to_bool(&condition_result);
                
                if is_true {
                    let mut last_result = TurbulanceValue::Boolean(true);
                    for instr in then_branch {
                        last_result = self.execute_instruction(instr)?;
                    }
                    Ok(last_result)
                } else if let Some(else_instrs) = else_branch {
                    let mut last_result = TurbulanceValue::Boolean(false);
                    for instr in else_instrs {
                        last_result = self.execute_instruction(instr)?;
                    }
                    Ok(last_result)
                } else {
                    Ok(TurbulanceValue::Boolean(false))
                }
            }
            
            Instruction::EvaluationCondition { condition, action } => {
                let condition_result = self.execute_operation(condition)?;
                let is_true = self.value_to_bool(&condition_result);
                
                if is_true {
                    self.execute_evaluation_action(action)?;
                }
                
                Ok(TurbulanceValue::Boolean(is_true))
            }
            
            Instruction::Return { value } => {
                if let Some(val_op) = value {
                    let val = self.execute_operation(val_op)?;
                    Ok(val)
                } else {
                    Ok(TurbulanceValue::String("void".to_string()))
                }
            }
            
            Instruction::Expression(operation) => {
                self.execute_operation(operation)
            }
            
            _ => Ok(TurbulanceValue::String("Instruction executed".to_string())),
        }
    }
    
    /// Execute an operation
    fn execute_operation(&mut self, operation: &Operation) -> Result<TurbulanceValue, RuntimeError> {
        match operation {
            Operation::LoadConstant(value) => {
                Ok(self.value_to_turbulance_value(value))
            }
            
            Operation::LoadVariable(name) => {
                self.variables.get(name)
                    .cloned()
                    .or_else(|| self.integration.get_variable(name).cloned())
                    .ok_or_else(|| RuntimeError::UndefinedVariable(name.clone()))
            }
            
            Operation::LoadFunction(name) => {
                if self.functions.contains_key(name) {
                    Ok(TurbulanceValue::String(format!("Function: {}", name)))
                } else {
                    Err(RuntimeError::UndefinedFunction(name.clone()))
                }
            }
            
            Operation::BinaryOperation { left, operator, right } => {
                let left_val = self.execute_operation(left)?;
                let right_val = self.execute_operation(right)?;
                self.execute_binary_operation(&left_val, operator, &right_val)
            }
            
            Operation::UnaryOperation { operator, operand } => {
                let operand_val = self.execute_operation(operand)?;
                self.execute_unary_operation(operator, &operand_val)
            }
            
            Operation::FunctionCall { function, arguments } => {
                let mut arg_values = Vec::new();
                for arg in arguments {
                    arg_values.push(self.execute_operation(arg)?);
                }
                
                self.call_function(function, &arg_values)
            }
            
            Operation::MethodCall { object, method, arguments } => {
                let object_val = self.execute_operation(object)?;
                let mut arg_values = Vec::new();
                for arg in arguments {
                    arg_values.push(self.execute_operation(arg)?);
                }
                
                self.call_method(&object_val, method, &arg_values)
            }
            
            Operation::CreateArray(elements) => {
                let mut array_values = Vec::new();
                for element in elements {
                    array_values.push(self.execute_operation(element)?);
                }
                Ok(TurbulanceValue::Array(array_values))
            }
            
            Operation::CreateObject(fields) => {
                let mut object_map = HashMap::new();
                for (key, value_op) in fields {
                    let value = self.execute_operation(value_op)?;
                    object_map.insert(key.clone(), value);
                }
                Ok(TurbulanceValue::Object(object_map))
            }
        }
    }
    
    /// Execute evaluation action
    fn execute_evaluation_action(&mut self, action: &EvaluationActionInstruction) -> Result<(), RuntimeError> {
        match action {
            EvaluationActionInstruction::Support { motion, weight } => {
                let weight_val = if let Some(w) = weight {
                    match self.execute_operation(w)? {
                        TurbulanceValue::Float(f) => f,
                        TurbulanceValue::Integer(i) => i as f64,
                        _ => 1.0,
                    }
                } else {
                    1.0
                };
                
                self.update_motion_support(motion, weight_val, true)?;
            }
            
            EvaluationActionInstruction::Contradict { motion, weight } => {
                let weight_val = if let Some(w) = weight {
                    match self.execute_operation(w)? {
                        TurbulanceValue::Float(f) => f,
                        TurbulanceValue::Integer(i) => i as f64,
                        _ => 1.0,
                    }
                } else {
                    1.0
                };
                
                self.update_motion_support(motion, -weight_val, false)?;
            }
            
            EvaluationActionInstruction::Collect { evidence_type } => {
                // Collect evidence from biological systems
                let evidence = self.integration.collect_evidence("auto", evidence_type)
                    .map_err(|e| RuntimeError::IntegrationError(e.to_string()))?;
                
                // Store evidence for later retrieval
                self.variables.insert(
                    format!("evidence_{}", evidence_type),
                    TurbulanceValue::String(format!("Evidence collected: {:?}", evidence))
                );
            }
            
            EvaluationActionInstruction::Flag { uncertainty } => {
                // Flag uncertainty in the system
                self.variables.insert(
                    "uncertainty_flag".to_string(),
                    TurbulanceValue::String(uncertainty.clone())
                );
            }
        }
        
        Ok(())
    }
    
    /// Execute binary operation
    fn execute_binary_operation(&self, left: &TurbulanceValue, op: &BinaryOp, right: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match op {
            BinaryOp::Add => self.add_values(left, right),
            BinaryOp::Subtract => self.subtract_values(left, right),
            BinaryOp::Multiply => self.multiply_values(left, right),
            BinaryOp::Divide => self.divide_values(left, right),
            BinaryOp::Equal => Ok(TurbulanceValue::Boolean(self.values_equal(left, right))),
            BinaryOp::NotEqual => Ok(TurbulanceValue::Boolean(!self.values_equal(left, right))),
            BinaryOp::Less => self.compare_values(left, right, |a, b| a < b),
            BinaryOp::Greater => self.compare_values(left, right, |a, b| a > b),
            BinaryOp::LessEqual => self.compare_values(left, right, |a, b| a <= b),
            BinaryOp::GreaterEqual => self.compare_values(left, right, |a, b| a >= b),
            BinaryOp::And => Ok(TurbulanceValue::Boolean(self.value_to_bool(left) && self.value_to_bool(right))),
            BinaryOp::Or => Ok(TurbulanceValue::Boolean(self.value_to_bool(left) || self.value_to_bool(right))),
            BinaryOp::Matches => self.match_pattern(left, right),
            BinaryOp::Contains => self.contains_value(left, right),
            _ => Err(RuntimeError::UnsupportedOperation(format!("Binary operation: {:?}", op))),
        }
    }
    
    /// Execute unary operation
    fn execute_unary_operation(&self, op: &UnaryOp, operand: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match op {
            UnaryOp::Not => Ok(TurbulanceValue::Boolean(!self.value_to_bool(operand))),
            UnaryOp::Minus => self.negate_value(operand),
            UnaryOp::Plus => Ok(operand.clone()), // Unary plus is a no-op
        }
    }
    
    /// Call a function
    fn call_function(&mut self, name: &str, args: &[TurbulanceValue]) -> Result<TurbulanceValue, RuntimeError> {
        if let Some(func_def) = self.functions.get(name).cloned() {
            if func_def.is_builtin {
                self.call_builtin_function(name, args)
            } else {
                self.call_user_function(&func_def, args)
            }
        } else {
            // Try calling biological function through integration
            self.integration.call_biological_function(name, args)
                .map_err(|e| RuntimeError::IntegrationError(e.to_string()))
        }
    }
    
    /// Call a method on an object
    fn call_method(&mut self, object: &TurbulanceValue, method: &str, args: &[TurbulanceValue]) -> Result<TurbulanceValue, RuntimeError> {
        match object {
            TurbulanceValue::BiologicalData(bio_data) => {
                // Call methods on biological data through integration
                match method {
                    "analyze" => Ok(TurbulanceValue::String(format!("Analyzed biological data: {:?}", bio_data))),
                    "measure" => Ok(TurbulanceValue::Float(42.0)), // Placeholder measurement
                    "evolve" => Ok(TurbulanceValue::String("Evolution step completed".to_string())),
                    _ => Err(RuntimeError::UndefinedMethod(method.to_string())),
                }
            }
            TurbulanceValue::Array(arr) => {
                match method {
                    "length" => Ok(TurbulanceValue::Integer(arr.len() as i64)),
                    "push" => {
                        if let Some(value) = args.get(0) {
                            Ok(TurbulanceValue::String(format!("Pushed {:?} to array", value)))
                        } else {
                            Err(RuntimeError::InvalidArguments("push requires one argument".to_string()))
                        }
                    }
                    _ => Err(RuntimeError::UndefinedMethod(method.to_string())),
                }
            }
            _ => Err(RuntimeError::InvalidMethodCall(format!("Cannot call method {} on {:?}", method, object))),
        }
    }
    
    /// Initialize builtin functions
    fn initialize_builtins(&mut self) {
        let builtins = vec![
            "print", "len", "type", "str", "int", "float", "bool",
            "analyze_atp_dynamics", "simulate_oscillation", "quantum_membrane_transport",
            "run_maxwell_demon", "calculate_entropy", "optimize_circuit", "measure_coherence",
            "track_pattern",
        ];
        
        for builtin in builtins {
            let func_def = FunctionDefinition {
                name: builtin.to_string(),
                parameters: Vec::new(),
                body: Vec::new(),
                is_builtin: true,
            };
            self.functions.insert(builtin.to_string(), func_def);
        }
    }
    
    /// Call builtin function
    fn call_builtin_function(&mut self, name: &str, args: &[TurbulanceValue]) -> Result<TurbulanceValue, RuntimeError> {
        match name {
            "print" => {
                for arg in args {
                    println!("{}", self.turbulance_value_to_string(arg));
                }
                Ok(TurbulanceValue::String("printed".to_string()))
            }
            "len" => {
                if let Some(arg) = args.get(0) {
                    match arg {
                        TurbulanceValue::Array(arr) => Ok(TurbulanceValue::Integer(arr.len() as i64)),
                        TurbulanceValue::String(s) => Ok(TurbulanceValue::Integer(s.len() as i64)),
                        _ => Err(RuntimeError::InvalidArguments("len() requires array or string".to_string())),
                    }
                } else {
                    Err(RuntimeError::InvalidArguments("len() requires one argument".to_string()))
                }
            }
            "type" => {
                if let Some(arg) = args.get(0) {
                    let type_name = match arg {
                        TurbulanceValue::Integer(_) => "integer",
                        TurbulanceValue::Float(_) => "float",
                        TurbulanceValue::String(_) => "string",
                        TurbulanceValue::Boolean(_) => "boolean",
                        TurbulanceValue::Array(_) => "array",
                        TurbulanceValue::Object(_) => "object",
                        TurbulanceValue::BiologicalData(_) => "biological_data",
                    };
                    Ok(TurbulanceValue::String(type_name.to_string()))
                } else {
                    Err(RuntimeError::InvalidArguments("type() requires one argument".to_string()))
                }
            }
            _ => {
                // Try biological function
                self.integration.call_biological_function(name, args)
                    .map_err(|e| RuntimeError::IntegrationError(e.to_string()))
            }
        }
    }
    
    /// Call user-defined function
    fn call_user_function(&mut self, func_def: &FunctionDefinition, args: &[TurbulanceValue]) -> Result<TurbulanceValue, RuntimeError> {
        // Create new call frame
        let call_frame = CallFrame {
            function_name: func_def.name.clone(),
            variables: HashMap::new(),
        };
        
        self.call_stack.push(call_frame);
        
        // Bind parameters
        for (i, param_name) in func_def.parameters.iter().enumerate() {
            if let Some(arg_value) = args.get(i) {
                self.call_stack.last_mut().unwrap().variables.insert(param_name.clone(), arg_value.clone());
            }
        }
        
        // Execute function body
        let mut result = TurbulanceValue::String("void".to_string());
        for instruction in &func_def.body {
            result = self.execute_instruction(instruction)?;
            // Check for early return
            if matches!(instruction, Instruction::Return { .. }) {
                break;
            }
        }
        
        // Clean up call frame
        self.call_stack.pop();
        
        Ok(result)
    }
    
    // Helper methods for value operations
    
    fn value_to_turbulance_value(&self, value: &Value) -> TurbulanceValue {
        match value {
            Value::Integer(i) => TurbulanceValue::Integer(*i),
            Value::Float(f) => TurbulanceValue::Float(*f),
            Value::String(s) => TurbulanceValue::String(s.clone()),
            Value::Boolean(b) => TurbulanceValue::Boolean(*b),
            Value::Char(c) => TurbulanceValue::String(c.to_string()),
            Value::Array(arr) => {
                let mut turb_arr = Vec::new();
                for val in arr {
                    turb_arr.push(self.value_to_turbulance_value(val));
                }
                TurbulanceValue::Array(turb_arr)
            }
            Value::Object(obj) => {
                let mut turb_obj = HashMap::new();
                for (key, val) in obj {
                    turb_obj.insert(key.clone(), self.value_to_turbulance_value(val));
                }
                TurbulanceValue::Object(turb_obj)
            }
            Value::Null => TurbulanceValue::String("null".to_string()),
        }
    }
    
    fn value_to_bool(&self, value: &TurbulanceValue) -> bool {
        match value {
            TurbulanceValue::Boolean(b) => *b,
            TurbulanceValue::Integer(i) => *i != 0,
            TurbulanceValue::Float(f) => *f != 0.0,
            TurbulanceValue::String(s) => !s.is_empty(),
            TurbulanceValue::Array(arr) => !arr.is_empty(),
            TurbulanceValue::Object(obj) => !obj.is_empty(),
            TurbulanceValue::BiologicalData(_) => true,
        }
    }
    
    fn turbulance_value_to_string(&self, value: &TurbulanceValue) -> String {
        match value {
            TurbulanceValue::Integer(i) => i.to_string(),
            TurbulanceValue::Float(f) => f.to_string(),
            TurbulanceValue::String(s) => s.clone(),
            TurbulanceValue::Boolean(b) => b.to_string(),
            TurbulanceValue::Array(arr) => {
                let elements: Vec<String> = arr.iter().map(|v| self.turbulance_value_to_string(v)).collect();
                format!("[{}]", elements.join(", "))
            }
            TurbulanceValue::Object(obj) => {
                let pairs: Vec<String> = obj.iter().map(|(k, v)| format!("{}: {}", k, self.turbulance_value_to_string(v))).collect();
                format!("{{{}}}", pairs.join(", "))
            }
            TurbulanceValue::BiologicalData(data) => format!("BiologicalData({:?})", data),
        }
    }
    
    fn add_values(&self, left: &TurbulanceValue, right: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (left, right) {
            (TurbulanceValue::Integer(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Integer(a + b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(a + b)),
            (TurbulanceValue::Integer(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(*a as f64 + b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Float(a + *b as f64)),
            (TurbulanceValue::String(a), TurbulanceValue::String(b)) => Ok(TurbulanceValue::String(format!("{}{}", a, b))),
            _ => Err(RuntimeError::TypeMismatch(format!("Cannot add {:?} and {:?}", left, right))),
        }
    }
    
    fn subtract_values(&self, left: &TurbulanceValue, right: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (left, right) {
            (TurbulanceValue::Integer(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Integer(a - b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(a - b)),
            (TurbulanceValue::Integer(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(*a as f64 - b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Float(a - *b as f64)),
            _ => Err(RuntimeError::TypeMismatch(format!("Cannot subtract {:?} and {:?}", left, right))),
        }
    }
    
    fn multiply_values(&self, left: &TurbulanceValue, right: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (left, right) {
            (TurbulanceValue::Integer(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Integer(a * b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(a * b)),
            (TurbulanceValue::Integer(a), TurbulanceValue::Float(b)) => Ok(TurbulanceValue::Float(*a as f64 * b)),
            (TurbulanceValue::Float(a), TurbulanceValue::Integer(b)) => Ok(TurbulanceValue::Float(a * *b as f64)),
            _ => Err(RuntimeError::TypeMismatch(format!("Cannot multiply {:?} and {:?}", left, right))),
        }
    }
    
    fn divide_values(&self, left: &TurbulanceValue, right: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (left, right) {
            (TurbulanceValue::Integer(a), TurbulanceValue::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(TurbulanceValue::Float(*a as f64 / *b as f64))
                }
            }
            (TurbulanceValue::Float(a), TurbulanceValue::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(TurbulanceValue::Float(a / b))
                }
            }
            (TurbulanceValue::Integer(a), TurbulanceValue::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(TurbulanceValue::Float(*a as f64 / b))
                }
            }
            (TurbulanceValue::Float(a), TurbulanceValue::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(TurbulanceValue::Float(a / *b as f64))
                }
            }
            _ => Err(RuntimeError::TypeMismatch(format!("Cannot divide {:?} and {:?}", left, right))),
        }
    }
    
    fn values_equal(&self, left: &TurbulanceValue, right: &TurbulanceValue) -> bool {
        match (left, right) {
            (TurbulanceValue::Integer(a), TurbulanceValue::Integer(b)) => a == b,
            (TurbulanceValue::Float(a), TurbulanceValue::Float(b)) => (a - b).abs() < f64::EPSILON,
            (TurbulanceValue::String(a), TurbulanceValue::String(b)) => a == b,
            (TurbulanceValue::Boolean(a), TurbulanceValue::Boolean(b)) => a == b,
            _ => false,
        }
    }
    
    fn compare_values<F>(&self, left: &TurbulanceValue, right: &TurbulanceValue, op: F) -> Result<TurbulanceValue, RuntimeError>
    where
        F: Fn(f64, f64) -> bool,
    {
        let left_num = self.extract_number(left)?;
        let right_num = self.extract_number(right)?;
        Ok(TurbulanceValue::Boolean(op(left_num, right_num)))
    }
    
    fn extract_number(&self, value: &TurbulanceValue) -> Result<f64, RuntimeError> {
        match value {
            TurbulanceValue::Integer(i) => Ok(*i as f64),
            TurbulanceValue::Float(f) => Ok(*f),
            _ => Err(RuntimeError::TypeMismatch(format!("Expected number, got {:?}", value))),
        }
    }
    
    fn negate_value(&self, value: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match value {
            TurbulanceValue::Integer(i) => Ok(TurbulanceValue::Integer(-i)),
            TurbulanceValue::Float(f) => Ok(TurbulanceValue::Float(-f)),
            _ => Err(RuntimeError::TypeMismatch(format!("Cannot negate {:?}", value))),
        }
    }
    
    fn match_pattern(&self, text: &TurbulanceValue, pattern: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (text, pattern) {
            (TurbulanceValue::String(t), TurbulanceValue::String(p)) => {
                let matches = t.contains(p); // Simple contains check for now
                Ok(TurbulanceValue::Boolean(matches))
            }
            _ => Err(RuntimeError::TypeMismatch("Pattern matching requires strings".to_string())),
        }
    }
    
    fn contains_value(&self, container: &TurbulanceValue, item: &TurbulanceValue) -> Result<TurbulanceValue, RuntimeError> {
        match (container, item) {
            (TurbulanceValue::Array(arr), item) => {
                let contains = arr.iter().any(|v| self.values_equal(v, item));
                Ok(TurbulanceValue::Boolean(contains))
            }
            (TurbulanceValue::String(s), TurbulanceValue::String(substr)) => {
                Ok(TurbulanceValue::Boolean(s.contains(substr)))
            }
            _ => Err(RuntimeError::TypeMismatch("Contains operation not supported for these types".to_string())),
        }
    }
    
    fn update_motion_support(&mut self, motion: &str, weight: f64, is_support: bool) -> Result<(), RuntimeError> {
        // Find the proposition containing this motion and update support
        for prop_def in self.propositions.values_mut() {
            if prop_def.motions.contains(&motion.to_string()) {
                // Update motion support - this would be more sophisticated in a real implementation
                let result = self.integration.evaluate_motion(motion)
                    .map_err(|e| RuntimeError::IntegrationError(e.to_string()))?;
                
                // Store the motion result for later collection
                let motion_key = format!("motion_result_{}", motion);
                self.variables.insert(motion_key, TurbulanceValue::String(format!("Motion {} evaluated", motion)));
                
                return Ok(());
            }
        }
        
        Err(RuntimeError::UndefinedMotion(motion.to_string()))
    }
    
    // Result collection methods
    
    fn collect_proposition_results(&mut self) -> Vec<PropositionResult> {
        let mut results = Vec::new();
        
        for (name, _prop_def) in &self.propositions {
            // Test the proposition using integration
            if let Ok(result) = self.integration.test_proposition(name, &_prop_def.motions) {
                results.push(result);
            }
        }
        
        results
    }
    
    fn collect_goal_results(&self) -> Vec<GoalResult> {
        let mut results = Vec::new();
        
        for (name, goal_def) in &self.goals {
            let result = GoalResult {
                name: name.clone(),
                progress: goal_def.current_progress,
                status: goal_def.status.clone(),
                metrics: goal_def.metrics.clone(),
                suggestions: vec!["Continue experiment".to_string()], // Placeholder
            };
            results.push(result);
        }
        
        results
    }
    
    fn collect_evidence_results(&mut self) -> Vec<EvidenceResult> {
        let mut results = Vec::new();
        
        // Collect evidence from various sources
        let sources = vec!["atp_pool", "oscillation_state", "quantum_membrane"];
        
        for source in sources {
            if let Ok(evidence) = self.integration.collect_evidence(source, "biological") {
                results.push(evidence);
            }
        }
        
        results
    }
    
    fn collect_variable_results(&self) -> HashMap<String, TurbulanceValue> {
        self.variables.clone()
    }
}

impl Default for TurbulanceRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Function definition
#[derive(Debug, Clone)]
struct FunctionDefinition {
    name: String,
    parameters: Vec<String>,
    body: Vec<Instruction>,
    is_builtin: bool,
}

/// Proposition definition
#[derive(Debug, Clone)]
struct PropositionDefinition {
    name: String,
    motions: Vec<String>,
    evaluations: Vec<EvaluationActionInstruction>,
}

/// Goal definition
#[derive(Debug, Clone)]
struct GoalDefinition {
    name: String,
    description: String,
    success_threshold: f64,
    keywords: Vec<String>,
    domain: Option<String>,
    audience: Option<String>,
    priority: Option<crate::turbulance::ast::Priority>,
    deadline: Option<String>,
    metrics: HashMap<String, f64>,
    current_progress: f64,
    status: GoalStatus,
}

/// Call frame for function execution
#[derive(Debug, Clone)]
struct CallFrame {
    function_name: String,
    variables: HashMap<String, TurbulanceValue>,
}

/// Runtime errors
#[derive(Debug, Clone)]
pub enum RuntimeError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    UndefinedMethod(String),
    UndefinedMotion(String),
    TypeMismatch(String),
    InvalidArguments(String),
    InvalidMethodCall(String),
    DivisionByZero,
    UnsupportedOperation(String),
    IntegrationError(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            RuntimeError::UndefinedFunction(name) => write!(f, "Undefined function: {}", name),
            RuntimeError::UndefinedMethod(name) => write!(f, "Undefined method: {}", name),
            RuntimeError::UndefinedMotion(name) => write!(f, "Undefined motion: {}", name),
            RuntimeError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            RuntimeError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            RuntimeError::InvalidMethodCall(msg) => write!(f, "Invalid method call: {}", msg),
            RuntimeError::DivisionByZero => write!(f, "Division by zero"),
            RuntimeError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            RuntimeError::IntegrationError(msg) => write!(f, "Integration error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::compiler::{TurbulanceProgram, Instruction, Operation, Value};

    #[test]
    fn test_runtime_creation() {
        let runtime = TurbulanceRuntime::new();
        assert!(runtime.variables.is_empty());
        assert!(runtime.functions.is_empty());
    }

    #[test]
    fn test_execute_simple_program() {
        let mut runtime = TurbulanceRuntime::new();
        
        let mut program = TurbulanceProgram::new();
        program.add_instruction(Instruction::VariableDeclaration {
            name: "x".to_string(),
            value: Box::new(Operation::LoadConstant(Value::Integer(42))),
        });
        
        let result = runtime.execute(program).unwrap();
        assert_eq!(result.variables.len(), 1);
        
        if let Some(TurbulanceValue::Integer(val)) = result.variables.get("x") {
            assert_eq!(*val, 42);
        } else {
            panic!("Expected integer variable x");
        }
    }

    #[test]
    fn test_binary_operations() {
        let mut runtime = TurbulanceRuntime::new();
        
        let left = TurbulanceValue::Integer(10);
        let right = TurbulanceValue::Integer(5);
        
        let result = runtime.execute_binary_operation(&left, &BinaryOp::Add, &right).unwrap();
        if let TurbulanceValue::Integer(val) = result {
            assert_eq!(val, 15);
        } else {
            panic!("Expected integer result");
        }
    }

    #[test]
    fn test_builtin_functions() {
        let mut runtime = TurbulanceRuntime::new();
        runtime.initialize_builtins();
        
        let args = vec![TurbulanceValue::String("Hello, World!".to_string())];
        let result = runtime.call_builtin_function("print", &args).unwrap();
        
        match result {
            TurbulanceValue::String(s) => assert_eq!(s, "printed"),
            _ => panic!("Expected string result"),
        }
    }
} 