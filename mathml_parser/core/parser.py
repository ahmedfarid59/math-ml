"""
Main Mathematical Expression Parser
=================================

This module provides the main parser interface that combines all
functionality including grammar, validation, transformation, and
robust error handling.
"""

import re
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from lark import Lark, ParseError, UnexpectedInput, UnexpectedToken, UnexpectedCharacters
from lark.exceptions import LarkError

from .grammar import MathematicalGrammar
from .transformer import EnhancedMathMLTransformer
from .validator import InputValidator
from .exceptions import MathParseError, MathParseResult


@dataclass
class ParserMetrics:
    """Performance and usage metrics for the parser."""
    parse_time: float
    validation_time: float
    transformation_time: float
    input_length: int
    output_length: int
    complexity_score: float
    features_used: List[str]


class MathMLParser:
    """
    Comprehensive mathematical expression parser with robust error handling.
    
    This parser provides:
    - Comprehensive mathematical notation support
    - Input validation and sanitization
    - Detailed error reporting with suggestions
    - Performance monitoring
    - Extensible architecture
    """
    
    def __init__(self, 
                 enable_validation: bool = True,
                 enable_metrics: bool = False,
                 strict_mode: bool = False,
                 cache_grammar: bool = True):
        """
        Initialize the mathematical expression parser.
        
        Args:
            enable_validation: Whether to enable input validation
            enable_metrics: Whether to collect performance metrics
            strict_mode: Whether to use strict parsing (less permissive)
            cache_grammar: Whether to cache the compiled grammar
        """
        self.enable_validation = enable_validation
        self.enable_metrics = enable_metrics
        self.strict_mode = strict_mode
        self.cache_grammar = cache_grammar
        
        # Initialize components
        self.grammar = MathematicalGrammar()
        self.transformer = EnhancedMathMLTransformer()
        self.validator = InputValidator() if enable_validation else None
        
        # Grammar compilation cache
        self._parser_cache: Optional[Lark] = None
        self._last_grammar_hash: Optional[str] = None
        
        # Performance tracking
        self.metrics_history: List[ParserMetrics] = []
        
        # Initialize parser
        self._initialize_parser()
    
    def _initialize_parser(self) -> None:
        """Initialize the Lark parser with the mathematical grammar."""
        try:
            grammar_text = self.grammar.get_grammar()
            grammar_hash = str(hash(grammar_text))
            
            # Use cached parser if available and grammar hasn't changed
            if (self.cache_grammar and 
                self._parser_cache is not None and 
                self._last_grammar_hash == grammar_hash):
                return
            
            # Create new parser
            self._parser_cache = Lark(
                grammar_text,
                parser='lalr',
                transformer=self.transformer,
                debug=False,
                propagate_positions=True
            )
            self._last_grammar_hash = grammar_hash
            
        except Exception as e:
            raise MathParseError(
                f"Failed to initialize parser: {str(e)}",
                error_type="ParserInitializationError",
                position=0
            )
    
    def parse(self, expression: str, **kwargs) -> str:
        """
        Parse a mathematical expression and return MathML.
        
        Args:
            expression: The mathematical expression to parse
            **kwargs: Additional parsing options
        
        Returns:
            MathML representation of the expression
        
        Raises:
            MathParseError: If parsing fails
        """
        result = self.parse_safe(expression, **kwargs)
        if result.success:
            return result.mathml
        else:
            raise result.error
    
    def parse_safe(self, expression: str, **kwargs) -> MathParseResult:
        """
        Safely parse a mathematical expression with comprehensive error handling.
        
        Args:
            expression: The mathematical expression to parse
            **kwargs: Additional parsing options
        
        Returns:
            MathParseResult containing either success or error information
        """
        start_time = time.time()
        validation_time = 0.0
        transformation_time = 0.0
        
        try:
            # Input validation
            if self.enable_validation and self.validator:
                validation_start = time.time()
                validation_result = self.validator.validate_input(expression)
                validation_time = time.time() - validation_start
                
                if not validation_result.is_valid:
                    return MathParseResult(
                        success=False,
                        mathml="",
                        error=MathParseError(
                            f"Input validation failed: {validation_result.message}",
                            error_type="ValidationError",
                            position=0,
                            suggestions=validation_result.suggestions,
                            metadata={"validation_issues": validation_result.issues}
                        ),
                        metrics=None
                    )
                
                # Use corrected expression if available
                expression = validation_result.corrected_input or expression
            
            # Parse the expression
            transformation_start = time.time()
            
            if self._parser_cache is None:
                self._initialize_parser()
            
            mathml = self._parser_cache.parse(expression)
            transformation_time = time.time() - transformation_start
            
            # Collect metrics if enabled
            metrics = None
            if self.enable_metrics:
                parse_time = time.time() - start_time
                metrics = ParserMetrics(
                    parse_time=parse_time,
                    validation_time=validation_time,
                    transformation_time=transformation_time,
                    input_length=len(expression),
                    output_length=len(str(mathml)),
                    complexity_score=self._calculate_complexity(expression),
                    features_used=self._detect_features(expression)
                )
                self.metrics_history.append(metrics)
            
            return MathParseResult(
                success=True,
                mathml=str(mathml),
                error=None,
                metrics=metrics
            )
            
        except ParseError as e:
            return self._handle_parse_error(e, expression)
        except UnexpectedInput as e:
            return self._handle_unexpected_input(e, expression)
        except LarkError as e:
            return self._handle_lark_error(e, expression)
        except Exception as e:
            return self._handle_generic_error(e, expression)
    
    def _handle_parse_error(self, error: ParseError, expression: str) -> MathParseResult:
        """Handle Lark ParseError with detailed information."""
        position = getattr(error, 'pos_in_stream', 0)
        
        # Extract context around error position
        context = self._get_error_context(expression, position)
        
        # Generate suggestions
        suggestions = self._generate_parse_suggestions(error, expression, position)
        
        math_error = MathParseError(
            f"Parse error at position {position}: {str(error)}",
            error_type="ParseError",
            position=position,
            context=context,
            suggestions=suggestions,
            metadata={
                "error_class": error.__class__.__name__,
                "original_error": str(error)
            }
        )
        
        return MathParseResult(
            success=False,
            mathml="",
            error=math_error,
            metrics=None
        )
    
    def _handle_unexpected_input(self, error: UnexpectedInput, expression: str) -> MathParseResult:
        """Handle unexpected input errors."""
        position = getattr(error, 'pos_in_stream', 0)
        context = self._get_error_context(expression, position)
        
        suggestions = []
        if hasattr(error, 'expected'):
            expected = error.expected
            if expected:
                suggestions.append(f"Expected one of: {', '.join(str(e) for e in expected)}")
        
        # Add specific suggestions based on error type
        if isinstance(error, UnexpectedToken):
            token = error.token
            suggestions.extend(self._get_token_suggestions(token, expression, position))
        elif isinstance(error, UnexpectedCharacters):
            char = expression[position] if position < len(expression) else "EOF"
            suggestions.extend(self._get_character_suggestions(char, expression, position))
        
        math_error = MathParseError(
            f"Unexpected input at position {position}: {str(error)}",
            error_type="UnexpectedInput",
            position=position,
            context=context,
            suggestions=suggestions,
            metadata={
                "error_class": error.__class__.__name__,
                "original_error": str(error)
            }
        )
        
        return MathParseResult(
            success=False,
            mathml="",
            error=math_error,
            metrics=None
        )
    
    def _handle_lark_error(self, error: LarkError, expression: str) -> MathParseResult:
        """Handle generic Lark errors."""
        math_error = MathParseError(
            f"Parser error: {str(error)}",
            error_type="LarkError",
            position=0,
            suggestions=["Check expression syntax", "Verify all parentheses are balanced"],
            metadata={
                "error_class": error.__class__.__name__,
                "original_error": str(error)
            }
        )
        
        return MathParseResult(
            success=False,
            mathml="",
            error=math_error,
            metrics=None
        )
    
    def _handle_generic_error(self, error: Exception, expression: str) -> MathParseResult:
        """Handle unexpected generic errors."""
        math_error = MathParseError(
            f"Unexpected error: {str(error)}",
            error_type="GenericError",
            position=0,
            suggestions=["Please report this error with the input expression"],
            metadata={
                "error_class": error.__class__.__name__,
                "original_error": str(error)
            }
        )
        
        return MathParseResult(
            success=False,
            mathml="",
            error=math_error,
            metrics=None
        )
    
    def _get_error_context(self, expression: str, position: int, context_size: int = 10) -> str:
        """Get context around an error position."""
        start = max(0, position - context_size)
        end = min(len(expression), position + context_size)
        
        context = expression[start:end]
        relative_pos = position - start
        
        # Add position marker
        if 0 <= relative_pos <= len(context):
            context = context[:relative_pos] + "⚠" + context[relative_pos:]
        
        return f"...{context}..." if start > 0 or end < len(expression) else context
    
    def _generate_parse_suggestions(self, error: ParseError, expression: str, position: int) -> List[str]:
        """Generate helpful suggestions for parse errors."""
        suggestions = []
        
        # Common parsing issues
        if "RPAR" in str(error):
            suggestions.append("Check for missing closing parenthesis ')'")
        elif "LPAR" in str(error):
            suggestions.append("Check for missing opening parenthesis '('")
        elif "NUMBER" in str(error):
            suggestions.append("Check number format (e.g., use '3.14' not '3,14')")
        elif "NAME" in str(error):
            suggestions.append("Check variable or function name spelling")
        
        # Context-specific suggestions
        if position < len(expression):
            char = expression[position]
            if char in "()[]{}":
                suggestions.append("Check bracket matching")
            elif char in "+-*/^":
                suggestions.append("Check operator usage and operands")
        
        return suggestions
    
    def _get_token_suggestions(self, token, expression: str, position: int) -> List[str]:
        """Get suggestions for unexpected token errors."""
        suggestions = []
        
        if hasattr(token, 'type'):
            if token.type == "NUMBER":
                suggestions.append("Numbers should not start expressions unless in parentheses")
            elif token.type == "NAME":
                suggestions.append("Check if this should be a function call with parentheses")
            elif token.type in ["PLUS", "MINUS", "MUL", "DIV"]:
                suggestions.append("Operators need operands on both sides")
        
        return suggestions
    
    def _get_character_suggestions(self, char: str, expression: str, position: int) -> List[str]:
        """Get suggestions for unexpected character errors."""
        suggestions = []
        
        if char in "()[]{}":
            suggestions.append("Check bracket matching and nesting")
        elif char in "+-*/^%":
            suggestions.append("Make sure operators have proper operands")
        elif char.isalpha():
            suggestions.append("Variables and functions should be properly formatted")
        elif char.isdigit():
            suggestions.append("Numbers should be properly formatted")
        else:
            suggestions.append(f"Character '{char}' may not be supported")
        
        return suggestions
    
    def _calculate_complexity(self, expression: str) -> float:
        """Calculate complexity score for an expression."""
        complexity = 0.0
        
        # Base complexity from length
        complexity += len(expression) * 0.1
        
        # Add complexity for various features
        complexity += expression.count('(') * 0.5  # Parentheses
        complexity += expression.count('[') * 1.0  # Matrices
        complexity += expression.count('{') * 1.0  # Sets
        complexity += len(re.findall(r'[a-zA-Z]+', expression)) * 0.3  # Variables/functions
        complexity += len(re.findall(r'\d+', expression)) * 0.2  # Numbers
        complexity += expression.count('^') * 0.4  # Exponents
        complexity += expression.count('_') * 0.3  # Subscripts
        
        return complexity
    
    def _detect_features(self, expression: str) -> List[str]:
        """Detect mathematical features used in the expression."""
        features = []
        
        # Basic operations
        if any(op in expression for op in ['+', '-', '*', '/', '^', '%']):
            features.append("arithmetic")
        
        # Functions
        if re.search(r'[a-zA-Z]+\s*\(', expression):
            features.append("functions")
        
        # Greek letters
        greek_pattern = r'\b(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b'
        if re.search(greek_pattern, expression, re.IGNORECASE):
            features.append("greek_letters")
        
        # Subscripts/superscripts
        if '_' in expression or '^' in expression:
            features.append("subscripts_superscripts")
        
        # Matrices
        if '[' in expression and ']' in expression:
            features.append("matrices")
        
        # Sets
        if '{' in expression and '}' in expression:
            features.append("sets")
        
        # Absolute values
        if '|' in expression:
            features.append("absolute_values")
        
        # Comparisons
        if any(op in expression for op in ['=', '!=', '<', '>', '<=', '>=']):
            features.append("comparisons")
        
        # Constants
        if any(const in expression for const in ['pi', 'e', 'infinity', 'inf']):
            features.append("constants")
        
        return features
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {"message": "No metrics collected"}
        
        parse_times = [m.parse_time for m in self.metrics_history]
        complexities = [m.complexity_score for m in self.metrics_history]
        input_lengths = [m.input_length for m in self.metrics_history]
        
        return {
            "total_parses": len(self.metrics_history),
            "avg_parse_time": sum(parse_times) / len(parse_times),
            "max_parse_time": max(parse_times),
            "min_parse_time": min(parse_times),
            "avg_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "max_input_length": max(input_lengths),
            "features_frequency": self._get_feature_frequency()
        }
    
    def _get_feature_frequency(self) -> Dict[str, int]:
        """Get frequency count of features used."""
        feature_counts = {}
        for metrics in self.metrics_history:
            for feature in metrics.features_used:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        return feature_counts
    
    def clear_metrics(self) -> None:
        """Clear collected metrics history."""
        self.metrics_history.clear()
    
    def reset_parser(self) -> None:
        """Reset the parser cache and reinitialize."""
        self._parser_cache = None
        self._last_grammar_hash = None
        self._initialize_parser()


# Create default parser instance for convenient access
default_parser = MathMLParser()

# Convenience functions for backward compatibility
def parse(expression: str) -> str:
    """Parse expression using default parser."""
    return default_parser.parse(expression)

def parse_safe(expression: str) -> MathParseResult:
    """Safely parse expression using default parser."""
    return default_parser.parse_safe(expression)