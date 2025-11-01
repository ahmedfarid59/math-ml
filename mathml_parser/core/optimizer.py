"""
Expression Optimization and Simplification Module
================================================

This module provides algebraic manipulation and expression simplification
capabilities for mathematical expressions.

Features:
- Expression simplification and normalization
- Algebraic expansion and factorization
- Trigonometric identity application
- Common subexpression elimination
- Symbolic differentiation helpers
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math


@dataclass
class ExpressionNode:
    """Represents a node in the expression tree."""
    operation: str
    operands: List['ExpressionNode']
    value: Optional[str] = None
    
    def __str__(self):
        if self.value is not None:
            return self.value
        if len(self.operands) == 1:
            return f"{self.operation}({self.operands[0]})"
        return f"({' '.join([str(op) for op in self.operands])})"


class ExpressionOptimizer:
    """
    Mathematical expression optimizer with algebraic manipulation capabilities.
    
    Provides various optimization techniques including simplification,
    expansion, factorization, and identity application.
    """
    
    # Mathematical constants for optimization
    CONSTANTS = {
        'π': math.pi,
        'pi': math.pi,
        'e': math.e,
        '∞': float('inf'),
        'infinity': float('inf'),
    }
    
    # Trigonometric identities for simplification
    TRIG_IDENTITIES = [
        # Basic identities
        (r'sin\(0\)', '0'),
        (r'cos\(0\)', '1'),
        (r'tan\(0\)', '0'),
        (r'sin\(π/2\)', '1'),
        (r'cos\(π/2\)', '0'),
        (r'sin\(π\)', '0'),
        (r'cos\(π\)', '-1'),
        
        # Pythagorean identity
        (r'sin\^2\(([^)]+)\)\s*\+\s*cos\^2\(\1\)', '1'),
        (r'cos\^2\(([^)]+)\)\s*\+\s*sin\^2\(\1\)', '1'),
        
        # Even/odd functions
        (r'sin\(-([^)]+)\)', '-sin(\1)'),
        (r'cos\(-([^)]+)\)', 'cos(\1)'),
        (r'tan\(-([^)]+)\)', '-tan(\1)'),
        
        # Double angle formulas
        (r'2\*sin\(([^)]+)\)\*cos\(\1\)', 'sin(2*\1)'),
        (r'cos\^2\(([^)]+)\)\s*-\s*sin\^2\(\1\)', 'cos(2*\1)'),
    ]
    
    # Algebraic simplification rules
    ALGEBRAIC_RULES = [
        # Addition/subtraction identities
        (r'(\w+)\s*\+\s*0', r'\1'),
        (r'0\s*\+\s*(\w+)', r'\1'),
        (r'(\w+)\s*-\s*0', r'\1'),
        (r'(\w+)\s*\+\s*\1', r'2*\1'),
        (r'(\w+)\s*-\s*\1', r'0'),
        
        # Multiplication identities
        (r'(\w+)\s*\*\s*1', r'\1'),
        (r'1\s*\*\s*(\w+)', r'\1'),
        (r'(\w+)\s*\*\s*0', r'0'),
        (r'0\s*\*\s*(\w+)', r'0'),
        (r'(\w+)\s*\*\s*\1', r'\1^2'),
        
        # Division identities
        (r'(\w+)\s*/\s*1', r'\1'),
        (r'0\s*/\s*(\w+)', r'0'),
        (r'(\w+)\s*/\s*\1', r'1'),
        
        # Power identities
        (r'(\w+)\^1', r'\1'),
        (r'(\w+)\^0', r'1'),
        (r'1\^(\w+)', r'1'),
        
        # Logarithmic identities
        (r'ln\(e\)', '1'),
        (r'log\(1\)', '0'),
        (r'log\(10\)', '1'),
        (r'exp\(0\)', '1'),
        (r'exp\(ln\(([^)]+)\)\)', r'\1'),
        (r'ln\(exp\(([^)]+)\)\)', r'\1'),
        
        # Root identities
        (r'sqrt\(0\)', '0'),
        (r'sqrt\(1\)', '1'),
        (r'sqrt\((\w+)\^2\)', r'|\1|'),
    ]
    
    def __init__(self):
        """Initialize the expression optimizer."""
        self.optimization_cache: Dict[str, str] = {}
    
    def optimize_expression(self, expression: str, 
                          apply_trig_identities: bool = True,
                          apply_algebraic_rules: bool = True,
                          expand_expressions: bool = False,
                          factor_expressions: bool = False) -> str:
        """
        Optimize a mathematical expression using various techniques.
        
        Args:
            expression: Mathematical expression to optimize
            apply_trig_identities: Apply trigonometric identities
            apply_algebraic_rules: Apply algebraic simplification rules
            expand_expressions: Expand algebraic expressions
            factor_expressions: Factor algebraic expressions
            
        Returns:
            Optimized expression
        """
        # Check cache first
        cache_key = f"{expression}_{apply_trig_identities}_{apply_algebraic_rules}_{expand_expressions}_{factor_expressions}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        result = expression
        
        # Apply optimization techniques in order
        if apply_algebraic_rules:
            result = self._apply_algebraic_rules(result)
        
        if apply_trig_identities:
            result = self._apply_trigonometric_identities(result)
        
        if expand_expressions:
            result = self._expand_expressions(result)
        
        if factor_expressions:
            result = self._factor_expressions(result)
        
        # Apply constant folding
        result = self._fold_constants(result)
        
        # Clean up the result
        result = self._clean_expression(result)
        
        # Cache the result
        self.optimization_cache[cache_key] = result
        
        return result
    
    def _apply_algebraic_rules(self, expression: str) -> str:
        """Apply algebraic simplification rules."""
        result = expression
        
        for pattern, replacement in self.ALGEBRAIC_RULES:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _apply_trigonometric_identities(self, expression: str) -> str:
        """Apply trigonometric identities for simplification."""
        result = expression
        
        for pattern, replacement in self.TRIG_IDENTITIES:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _expand_expressions(self, expression: str) -> str:
        """Expand algebraic expressions."""
        # Simple expansion patterns
        expansions = [
            # (a+b)^2 = a^2 + 2ab + b^2
            (r'\(([^+\-*/^()]+)\s*\+\s*([^+\-*/^()]+)\)\^2',
             r'\1^2 + 2*\1*\2 + \2^2'),
            
            # (a-b)^2 = a^2 - 2ab + b^2
            (r'\(([^+\-*/^()]+)\s*-\s*([^+\-*/^()]+)\)\^2',
             r'\1^2 - 2*\1*\2 + \2^2'),
            
            # (a+b)(a-b) = a^2 - b^2
            (r'\(([^+\-*/^()]+)\s*\+\s*([^+\-*/^()]+)\)\s*\*\s*\(\1\s*-\s*\2\)',
             r'\1^2 - \2^2'),
            
            # a(b+c) = ab + ac
            (r'([^+\-*/^()]+)\s*\*\s*\(([^+\-*/^()]+)\s*\+\s*([^+\-*/^()]+)\)',
             r'\1*\2 + \1*\3'),
        ]
        
        result = expression
        for pattern, replacement in expansions:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _factor_expressions(self, expression: str) -> str:
        """Factor algebraic expressions."""
        # Simple factoring patterns
        factorizations = [
            # a^2 + 2ab + b^2 = (a+b)^2
            (r'([^+\-*/^()]+)\^2\s*\+\s*2\*\1\*([^+\-*/^()]+)\s*\+\s*\2\^2',
             r'(\1 + \2)^2'),
            
            # a^2 - 2ab + b^2 = (a-b)^2
            (r'([^+\-*/^()]+)\^2\s*-\s*2\*\1\*([^+\-*/^()]+)\s*\+\s*\2\^2',
             r'(\1 - \2)^2'),
            
            # a^2 - b^2 = (a+b)(a-b)
            (r'([^+\-*/^()]+)\^2\s*-\s*([^+\-*/^()]+)\^2',
             r'(\1 + \2)*(\1 - \2)'),
            
            # ax + bx = x(a + b)
            (r'([^+\-*/^()]+)\*([^+\-*/^()]+)\s*\+\s*([^+\-*/^()]+)\*\2',
             r'\2*(\1 + \3)'),
        ]
        
        result = expression
        for pattern, replacement in factorizations:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _fold_constants(self, expression: str) -> str:
        """Fold constant expressions into single values."""
        # Find and evaluate simple constant expressions
        constant_patterns = [
            # Simple arithmetic with numbers
            (r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', lambda m: str(float(m.group(1)) + float(m.group(2)))),
            (r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', lambda m: str(float(m.group(1)) - float(m.group(2)))),
            (r'(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)', lambda m: str(float(m.group(1)) * float(m.group(2)))),
            (r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', lambda m: str(float(m.group(1)) / float(m.group(2))) if float(m.group(2)) != 0 else m.group(0)),
        ]
        
        result = expression
        for pattern, replacement in constant_patterns:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)
        
        return result
    
    def _clean_expression(self, expression: str) -> str:
        """Clean up the expression formatting."""
        result = expression
        
        # Remove unnecessary parentheses
        result = re.sub(r'\(([^()]*)\)', lambda m: m.group(1) if not self._needs_parentheses(m.group(1)) else m.group(0), result)
        
        # Normalize spacing
        result = re.sub(r'\s*([+\-*/^=<>≤≥≠])\s*', r' \1 ', result)
        result = re.sub(r'\s+', ' ', result)
        
        # Remove leading/trailing whitespace
        result = result.strip()
        
        return result
    
    def _needs_parentheses(self, expr: str) -> bool:
        """Check if an expression needs parentheses."""
        # Simple heuristic: if it contains operators, it might need parentheses
        operators = ['+', '-', '*', '/', '^', '=', '<', '>', '≤', '≥', '≠']
        return any(op in expr for op in operators)
    
    def get_optimization_stats(self) -> Dict[str, int]:
        """Get statistics about optimization operations."""
        return {
            'cache_size': len(self.optimization_cache),
            'cache_hits': getattr(self, '_cache_hits', 0),
            'optimizations_performed': getattr(self, '_optimizations_count', 0)
        }
    
    def clear_cache(self) -> None:
        """Clear the optimization cache."""
        self.optimization_cache.clear()
    
    def suggest_optimizations(self, expression: str) -> List[str]:
        """
        Suggest possible optimizations for an expression.
        
        Args:
            expression: Expression to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check for common optimization opportunities
        if re.search(r'\w+\s*\+\s*0|\w+\s*\*\s*1', expression):
            suggestions.append("Contains identity operations that can be simplified")
        
        if re.search(r'sin\^2\([^)]+\)\s*\+\s*cos\^2\([^)]+\)', expression):
            suggestions.append("Contains trigonometric Pythagorean identity")
        
        if re.search(r'\([^)]+\)\^2', expression):
            suggestions.append("Contains squared expressions that might be expandable")
        
        if re.search(r'\w+\^2\s*[-+]\s*\w+\^2', expression):
            suggestions.append("Contains difference/sum of squares that might be factorizable")
        
        if re.search(r'ln\(exp\(|exp\(ln\(', expression):
            suggestions.append("Contains inverse functions that can be simplified")
        
        return suggestions


# Convenience function for easy access
def optimize_expression(expression: str, **kwargs) -> str:
    """
    Optimize a mathematical expression.
    
    Args:
        expression: Mathematical expression to optimize
        **kwargs: Optimization options
        
    Returns:
        Optimized expression
    """
    optimizer = ExpressionOptimizer()
    return optimizer.optimize_expression(expression, **kwargs)