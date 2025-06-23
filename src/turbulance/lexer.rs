//! # Turbulance Lexer
//!
//! Tokenizes Turbulance source code into a stream of tokens for parsing.

use std::collections::HashMap;
use std::fmt;

/// Turbulance lexer
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    keywords: HashMap<String, TokenType>,
}

impl Lexer {
    /// Create a new lexer instance
    pub fn new() -> Self {
        let mut keywords = HashMap::new();
        
        // Basic language keywords
        keywords.insert("item".to_string(), TokenType::Item);
        keywords.insert("funxn".to_string(), TokenType::Funxn);
        keywords.insert("given".to_string(), TokenType::Given);
        keywords.insert("within".to_string(), TokenType::Within);
        keywords.insert("considering".to_string(), TokenType::Considering);
        keywords.insert("otherwise".to_string(), TokenType::Otherwise);
        keywords.insert("return".to_string(), TokenType::Return);
        keywords.insert("for".to_string(), TokenType::For);
        keywords.insert("each".to_string(), TokenType::Each);
        keywords.insert("in".to_string(), TokenType::In);
        keywords.insert("while".to_string(), TokenType::While);
        keywords.insert("try".to_string(), TokenType::Try);
        keywords.insert("catch".to_string(), TokenType::Catch);
        keywords.insert("finally".to_string(), TokenType::Finally);
        keywords.insert("import".to_string(), TokenType::Import);
        keywords.insert("from".to_string(), TokenType::From);
        keywords.insert("as".to_string(), TokenType::As);
        keywords.insert("true".to_string(), TokenType::True);
        keywords.insert("false".to_string(), TokenType::False);
        keywords.insert("and".to_string(), TokenType::And);
        keywords.insert("or".to_string(), TokenType::Or);
        keywords.insert("not".to_string(), TokenType::Not);
        keywords.insert("matches".to_string(), TokenType::Matches);
        keywords.insert("contains".to_string(), TokenType::Contains);
        keywords.insert("where".to_string(), TokenType::Where);
        
        // Scientific reasoning keywords
        keywords.insert("proposition".to_string(), TokenType::Proposition);
        keywords.insert("motion".to_string(), TokenType::Motion);
        keywords.insert("evidence".to_string(), TokenType::Evidence);
        keywords.insert("support".to_string(), TokenType::Support);
        keywords.insert("contradict".to_string(), TokenType::Contradict);
        keywords.insert("metacognitive".to_string(), TokenType::Metacognitive);
        keywords.insert("temporal".to_string(), TokenType::Temporal);
        keywords.insert("pattern_registry".to_string(), TokenType::PatternRegistry);
        keywords.insert("cross_domain_analysis".to_string(), TokenType::CrossDomainAnalysis);
        keywords.insert("evidence_integrator".to_string(), TokenType::EvidenceIntegrator);
        keywords.insert("orchestration".to_string(), TokenType::Orchestration);
        keywords.insert("compose_pattern".to_string(), TokenType::ComposePattern);
        keywords.insert("evidence_chain".to_string(), TokenType::EvidenceChain);
        
        // Goal system keywords
        keywords.insert("goal".to_string(), TokenType::Goal);
        keywords.insert("success_threshold".to_string(), TokenType::SuccessThreshold);
        keywords.insert("keywords".to_string(), TokenType::Keywords);
        keywords.insert("domain".to_string(), TokenType::Domain);
        keywords.insert("audience".to_string(), TokenType::Audience);
        keywords.insert("priority".to_string(), TokenType::Priority);
        keywords.insert("deadline".to_string(), TokenType::Deadline);
        keywords.insert("metrics".to_string(), TokenType::Metrics);
        
        // Control flow modifiers
        keywords.insert("parallel".to_string(), TokenType::Parallel);
        keywords.insert("async".to_string(), TokenType::Async);
        keywords.insert("await".to_string(), TokenType::Await);
        keywords.insert("concurrent".to_string(), TokenType::Concurrent);
        keywords.insert("stream".to_string(), TokenType::Stream);
        keywords.insert("lazy_evaluation".to_string(), TokenType::LazyEvaluation);
        
        // Data structure keywords
        keywords.insert("sources".to_string(), TokenType::Sources);
        keywords.insert("collection".to_string(), TokenType::Collection);
        keywords.insert("processing".to_string(), TokenType::Processing);
        keywords.insert("storage".to_string(), TokenType::Storage);
        keywords.insert("validation".to_string(), TokenType::Validation);
        keywords.insert("category".to_string(), TokenType::Category);
        keywords.insert("matching".to_string(), TokenType::Matching);
        keywords.insert("relationships".to_string(), TokenType::Relationships);
        keywords.insert("track".to_string(), TokenType::Track);
        keywords.insert("evaluate".to_string(), TokenType::Evaluate);
        keywords.insert("adapt".to_string(), TokenType::Adapt);
        keywords.insert("scope".to_string(), TokenType::Scope);
        keywords.insert("patterns".to_string(), TokenType::Patterns);
        keywords.insert("operations".to_string(), TokenType::Operations);
        
        Self {
            input: Vec::new(),
            position: 0,
            line: 1,
            column: 1,
            keywords,
        }
    }

    /// Tokenize source code into a vector of tokens
    pub fn tokenize(&mut self, input: &str) -> Result<Vec<Token>, LexError> {
        self.input = input.chars().collect();
        self.position = 0;
        self.line = 1;
        self.column = 1;
        
        let mut tokens = Vec::new();
        
        while self.position < self.input.len() {
            self.skip_whitespace();
            
            if self.position >= self.input.len() {
                break;
            }
            
            let token = self.next_token()?;
            tokens.push(token);
        }
        
        tokens.push(Token {
            token_type: TokenType::Eof,
            value: String::new(),
            line: self.line,
            column: self.column,
        });
        
        Ok(tokens)
    }
    
    /// Get the next token from the input
    fn next_token(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_column = self.column;
        
        let ch = self.current_char();
        
        match ch {
            // Single character tokens
            '(' => {
                self.advance();
                Ok(Token::new(TokenType::LeftParen, "(", start_line, start_column))
            }
            ')' => {
                self.advance();
                Ok(Token::new(TokenType::RightParen, ")", start_line, start_column))
            }
            '[' => {
                self.advance();
                Ok(Token::new(TokenType::LeftBracket, "[", start_line, start_column))
            }
            ']' => {
                self.advance();
                Ok(Token::new(TokenType::RightBracket, "]", start_line, start_column))
            }
            '{' => {
                self.advance();
                Ok(Token::new(TokenType::LeftBrace, "{", start_line, start_column))
            }
            '}' => {
                self.advance();
                Ok(Token::new(TokenType::RightBrace, "}", start_line, start_column))
            }
            ',' => {
                self.advance();
                Ok(Token::new(TokenType::Comma, ",", start_line, start_column))
            }
            ';' => {
                self.advance();
                Ok(Token::new(TokenType::Semicolon, ";", start_line, start_column))
            }
            ':' => {
                self.advance();
                Ok(Token::new(TokenType::Colon, ":", start_line, start_column))
            }
            '.' => {
                self.advance();
                Ok(Token::new(TokenType::Dot, ".", start_line, start_column))
            }
            '+' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::PlusEquals, "+=", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Plus, "+", start_line, start_column))
                }
            }
            '-' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::MinusEquals, "-=", start_line, start_column))
                } else if self.current_char() == '>' {
                    self.advance();
                    Ok(Token::new(TokenType::Arrow, "->", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Minus, "-", start_line, start_column))
                }
            }
            '*' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::StarEquals, "*=", start_line, start_column))
                } else if self.current_char() == '*' {
                    self.advance();
                    Ok(Token::new(TokenType::Power, "**", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Star, "*", start_line, start_column))
                }
            }
            '/' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::SlashEquals, "/=", start_line, start_column))
                } else if self.current_char() == '/' {
                    // Single line comment
                    self.skip_line_comment();
                    self.next_token()
                } else if self.current_char() == '*' {
                    // Multi-line comment
                    self.skip_block_comment()?;
                    self.next_token()
                } else {
                    Ok(Token::new(TokenType::Slash, "/", start_line, start_column))
                }
            }
            '%' => {
                self.advance();
                Ok(Token::new(TokenType::Percent, "%", start_line, start_column))
            }
            '=' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::Equals, "==", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Assign, "=", start_line, start_column))
                }
            }
            '!' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::NotEquals, "!=", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Bang, "!", start_line, start_column))
                }
            }
            '<' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::LessEquals, "<=", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Less, "<", start_line, start_column))
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == '=' {
                    self.advance();
                    Ok(Token::new(TokenType::GreaterEquals, ">=", start_line, start_column))
                } else {
                    Ok(Token::new(TokenType::Greater, ">", start_line, start_column))
                }
            }
            '&' => {
                self.advance();
                Ok(Token::new(TokenType::Ampersand, "&", start_line, start_column))
            }
            '|' => {
                self.advance();
                Ok(Token::new(TokenType::Pipe, "|", start_line, start_column))
            }
            '?' => {
                self.advance();
                Ok(Token::new(TokenType::Question, "?", start_line, start_column))
            }
            '"' => self.read_string(),
            '\'' => self.read_char(),
            _ if ch.is_ascii_digit() => self.read_number(),
            _ if ch.is_ascii_alphabetic() || ch == '_' => self.read_identifier(),
            _ => Err(LexError::UnexpectedCharacter {
                character: ch,
                line: self.line,
                column: self.column,
            }),
        }
    }
    
    /// Get the current character
    fn current_char(&self) -> char {
        if self.position >= self.input.len() {
            '\0'
        } else {
            self.input[self.position]
        }
    }
    
    /// Advance to the next character
    fn advance(&mut self) {
        if self.position < self.input.len() && self.input[self.position] == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        self.position += 1;
    }
    
    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.current_char().is_whitespace() {
            self.advance();
        }
    }
    
    /// Skip single line comment
    fn skip_line_comment(&mut self) {
        while self.position < self.input.len() && self.current_char() != '\n' {
            self.advance();
        }
    }
    
    /// Skip multi-line comment
    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        self.advance(); // Skip the '*'
        
        while self.position < self.input.len() - 1 {
            if self.current_char() == '*' && self.input[self.position + 1] == '/' {
                self.advance(); // Skip '*'
                self.advance(); // Skip '/'
                return Ok(());
            }
            self.advance();
        }
        
        Err(LexError::UnterminatedComment {
            line: self.line,
            column: self.column,
        })
    }
    
    /// Read a string literal
    fn read_string(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_column = self.column;
        
        self.advance(); // Skip opening quote
        let mut value = String::new();
        
        while self.position < self.input.len() && self.current_char() != '"' {
            if self.current_char() == '\\' {
                self.advance();
                if self.position >= self.input.len() {
                    return Err(LexError::UnterminatedString {
                        line: start_line,
                        column: start_column,
                    });
                }
                
                match self.current_char() {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    ch => value.push(ch),
                }
            } else {
                value.push(self.current_char());
            }
            self.advance();
        }
        
        if self.position >= self.input.len() {
            return Err(LexError::UnterminatedString {
                line: start_line,
                column: start_column,
            });
        }
        
        self.advance(); // Skip closing quote
        
        Ok(Token::new(TokenType::String, &value, start_line, start_column))
    }
    
    /// Read a character literal
    fn read_char(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_column = self.column;
        
        self.advance(); // Skip opening quote
        
        if self.position >= self.input.len() {
            return Err(LexError::UnterminatedChar {
                line: start_line,
                column: start_column,
            });
        }
        
        let ch = if self.current_char() == '\\' {
            self.advance();
            if self.position >= self.input.len() {
                return Err(LexError::UnterminatedChar {
                    line: start_line,
                    column: start_column,
                });
            }
            
            match self.current_char() {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                ch => ch,
            }
        } else {
            self.current_char()
        };
        
        self.advance();
        
        if self.position >= self.input.len() || self.current_char() != '\'' {
            return Err(LexError::UnterminatedChar {
                line: start_line,
                column: start_column,
            });
        }
        
        self.advance(); // Skip closing quote
        
        Ok(Token::new(TokenType::Char, &ch.to_string(), start_line, start_column))
    }
    
    /// Read a number literal
    fn read_number(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_column = self.column;
        let mut value = String::new();
        let mut is_float = false;
        
        // Read integer part
        while self.position < self.input.len() && self.current_char().is_ascii_digit() {
            value.push(self.current_char());
            self.advance();
        }
        
        // Check for decimal point
        if self.position < self.input.len() && self.current_char() == '.' {
            // Look ahead to see if this is a number or a method call
            if self.position + 1 < self.input.len() && self.input[self.position + 1].is_ascii_digit() {
                is_float = true;
                value.push(self.current_char());
                self.advance();
                
                // Read fractional part
                while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                    value.push(self.current_char());
                    self.advance();
                }
            }
        }
        
        // Check for scientific notation
        if self.position < self.input.len() && (self.current_char() == 'e' || self.current_char() == 'E') {
            is_float = true;
            value.push(self.current_char());
            self.advance();
            
            if self.position < self.input.len() && (self.current_char() == '+' || self.current_char() == '-') {
                value.push(self.current_char());
                self.advance();
            }
            
            while self.position < self.input.len() && self.current_char().is_ascii_digit() {
                value.push(self.current_char());
                self.advance();
            }
        }
        
        let token_type = if is_float {
            TokenType::Float
        } else {
            TokenType::Integer
        };
        
        Ok(Token::new(token_type, &value, start_line, start_column))
    }
    
    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> Result<Token, LexError> {
        let start_line = self.line;
        let start_column = self.column;
        let mut value = String::new();
        
        while self.position < self.input.len() && 
              (self.current_char().is_ascii_alphanumeric() || self.current_char() == '_') {
            value.push(self.current_char());
            self.advance();
        }
        
        let token_type = self.keywords.get(&value)
            .cloned()
            .unwrap_or(TokenType::Identifier);
        
        Ok(Token::new(token_type, &value, start_line, start_column))
    }
}

impl Default for Lexer {
    fn default() -> Self {
        Self::new()
    }
}

/// A token in the Turbulance language
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, value: &str, line: usize, column: usize) -> Self {
        Self {
            token_type,
            value: value.to_string(),
            line,
            column,
        }
    }
}

/// Token types in the Turbulance language
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    Integer,
    Float,
    String,
    Char,
    True,
    False,
    
    // Identifiers
    Identifier,
    
    // Keywords
    Item,
    Funxn,
    Given,
    Within,
    Considering,
    Otherwise,
    Return,
    For,
    Each,
    In,
    While,
    Try,
    Catch,
    Finally,
    Import,
    From,
    As,
    And,
    Or,
    Not,
    Matches,
    Contains,
    Where,
    
    // Scientific reasoning
    Proposition,
    Motion,
    Evidence,
    Support,
    Contradict,
    Metacognitive,
    Temporal,
    PatternRegistry,
    CrossDomainAnalysis,
    EvidenceIntegrator,
    Orchestration,
    ComposePattern,
    EvidenceChain,
    
    // Goal system
    Goal,
    SuccessThreshold,
    Keywords,
    Domain,
    Audience,
    Priority,
    Deadline,
    Metrics,
    
    // Control flow
    Parallel,
    Async,
    Await,
    Concurrent,
    Stream,
    LazyEvaluation,
    
    // Data structures
    Sources,
    Collection,
    Processing,
    Storage,
    Validation,
    Category,
    Matching,
    Relationships,
    Track,
    Evaluate,
    Adapt,
    Scope,
    Patterns,
    Operations,
    
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    Assign,
    PlusEquals,
    MinusEquals,
    StarEquals,
    SlashEquals,
    Equals,
    NotEquals,
    Less,
    Greater,
    LessEquals,
    GreaterEquals,
    Bang,
    Ampersand,
    Pipe,
    Question,
    Arrow,
    
    // Punctuation
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Semicolon,
    Colon,
    Dot,
    
    // Special
    Eof,
}

/// Lexer errors
#[derive(Debug, Clone)]
pub enum LexError {
    UnexpectedCharacter {
        character: char,
        line: usize,
        column: usize,
    },
    UnterminatedString {
        line: usize,
        column: usize,
    },
    UnterminatedChar {
        line: usize,
        column: usize,
    },
    UnterminatedComment {
        line: usize,
        column: usize,
    },
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexError::UnexpectedCharacter { character, line, column } => {
                write!(f, "Unexpected character '{}' at line {}, column {}", character, line, column)
            }
            LexError::UnterminatedString { line, column } => {
                write!(f, "Unterminated string literal at line {}, column {}", line, column)
            }
            LexError::UnterminatedChar { line, column } => {
                write!(f, "Unterminated character literal at line {}, column {}", line, column)
            }
            LexError::UnterminatedComment { line, column } => {
                write!(f, "Unterminated comment at line {}, column {}", line, column)
            }
        }
    }
}

impl std::error::Error for LexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("item x = 42").unwrap();
        
        assert_eq!(tokens.len(), 5); // item, x, =, 42, EOF
        assert_eq!(tokens[0].token_type, TokenType::Item);
        assert_eq!(tokens[1].token_type, TokenType::Identifier);
        assert_eq!(tokens[2].token_type, TokenType::Assign);
        assert_eq!(tokens[3].token_type, TokenType::Integer);
        assert_eq!(tokens[4].token_type, TokenType::Eof);
    }

    #[test]
    fn test_proposition_syntax() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("proposition TestHypothesis:").unwrap();
        
        assert_eq!(tokens[0].token_type, TokenType::Proposition);
        assert_eq!(tokens[1].token_type, TokenType::Identifier);
        assert_eq!(tokens[2].token_type, TokenType::Colon);
    }

    #[test]
    fn test_function_syntax() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("funxn test_function():").unwrap();
        
        assert_eq!(tokens[0].token_type, TokenType::Funxn);
        assert_eq!(tokens[1].token_type, TokenType::Identifier);
        assert_eq!(tokens[2].token_type, TokenType::LeftParen);
        assert_eq!(tokens[3].token_type, TokenType::RightParen);
        assert_eq!(tokens[4].token_type, TokenType::Colon);
    }

    #[test]
    fn test_string_literals() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize(r#""Hello, world!""#).unwrap();
        
        assert_eq!(tokens[0].token_type, TokenType::String);
        assert_eq!(tokens[0].value, "Hello, world!");
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new();
        let tokens = lexer.tokenize("// This is a comment\nitem x = 1").unwrap();
        
        assert_eq!(tokens[0].token_type, TokenType::Item);
        assert_eq!(tokens[1].token_type, TokenType::Identifier);
    }
} 