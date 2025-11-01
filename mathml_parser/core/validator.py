"""
Input Validation and Sanitization for MathML Parser
==================================================

This module provides comprehensive input validation, sanitization, and
preprocessing for mathematical expressions before parsing.
"""

import re
from typing import Tuple, Optional, List, Dict
from .exceptions import ValidationError


class InputValidator:
    """
    Comprehensive input validator for mathematical expressions.
    
    Provides validation, sanitization, and preprocessing capabilities
    to ensure robust parsing and helpful error reporting.
    """
    
    # Invalid patterns and their error messages
    INVALID_PATTERNS = [
        (r'\+\+', "Double plus (++) not allowed"),
        (r'--(?![0-9])', "Double minus (--) not allowed (use parentheses for negative numbers)"),
        (r'\*\*', "Double asterisk (**) not supported, use ^ for powers"),
        (r'//', "Double slash (//) not allowed"),
        (r'\^\^', "Double caret (^^) not allowed"),
        (r'[+\-*/^%]{3,}', "Too many consecutive operators"),
        (r'[.,]{2,}', "Multiple consecutive decimal points or commas"),
    ]
    
    # Common typos and corrections
    COMMON_TYPOS = {
        'sinn': 'sin',
        'coss': 'cos',
        'tann': 'tan',
        'llog': 'log',
        'sqrrt': 'sqrt',
        'roott': 'root',
        'exppp': 'exp',
        'abss': 'abs',
        'maxx': 'max',
        'minn': 'min',
        'flooor': 'floor',
        'ceill': 'ceil',
        'alphaa': 'alpha',
        'betaa': 'beta',
        'gammaa': 'gamma'
    }
    
    # Valid mathematical function names
    VALID_FUNCTIONS = {
        'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
        'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh',
        'log', 'ln', 'exp', 'sqrt', 'root',
        'abs', 'floor', 'ceil', 'round',
        'max', 'min', 'sum', 'prod',
        'int', 'lim'
    }
    
    # Valid Greek letters
    VALID_GREEK = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
        'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi',
        'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
    }
    
    # Valid mathematical constants
    VALID_CONSTANTS = {'e', 'pi', 'π', 'infinity', 'inf', '∞'}
    
    @staticmethod
    def validate_input(expression: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate and sanitize mathematical expression input.
        
        Args:
            expression (str): Mathematical expression to validate
            
        Returns:
            tuple: (is_valid, error_message, sanitized_expression)
        """
        # Type validation
        if not isinstance(expression, str):
            return False, "Input must be a string", None
            
        # Empty input validation
        if not expression.strip():
            return False, "Input expression cannot be empty", None
        
        # Initial sanitization
        sanitized = InputValidator._sanitize_basic(expression)
        
        # Bracket validation
        is_valid, error_msg = InputValidator._validate_brackets(sanitized)
        if not is_valid:
            return False, error_msg, None
        
        # Pattern validation
        is_valid, error_msg = InputValidator._validate_patterns(sanitized)
        if not is_valid:
            return False, error_msg, None
        
        # Operator position validation
        is_valid, error_msg = InputValidator._validate_operators(sanitized)
        if not is_valid:
            return False, error_msg, None
        
        # Function name validation and typo detection
        suggestions = InputValidator._check_typos(sanitized)
        
        # Advanced sanitization
        sanitized = InputValidator._sanitize_advanced(sanitized)
        
        return True, None, sanitized
    
    @staticmethod
    def _sanitize_basic(expression: str) -> str:
        """Basic sanitization: whitespace and normalization."""
        # Remove extra whitespace
        sanitized = re.sub(r'\s+', ' ', expression.strip())
        
        # Normalize some common Unicode characters
        replacements = {
            '×': '*',           # Multiplication sign
            '÷': '/',           # Division sign
            '−': '-',           # Minus sign
            '∗': '*',           # Asterisk operator
            '∘': '*',           # Ring operator
        }
        
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        
        return sanitized
    
    @staticmethod
    def _sanitize_advanced(expression: str) -> str:
        """Advanced sanitization after validation."""
        # Remove unnecessary spaces around operators
        sanitized = re.sub(r'\s*([+\-*/^%=<>!]+)\s*', r'\1', expression)
        
        # Ensure space after commas
        sanitized = re.sub(r',\s*', ', ', sanitized)
        
        # Clean up multiple spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    @staticmethod
    def _validate_brackets(expression: str) -> Tuple[bool, Optional[str]]:
        """Validate bracket matching."""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for i, char in enumerate(expression):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    bracket_name = {')': 'parenthesis', ']': 'bracket', '}': 'brace'}[char]
                    return False, f"Unmatched closing {bracket_name} at position {i}"
                
                open_char, open_pos = stack.pop()
                expected_close = brackets[open_char]
                if char != expected_close:
                    bracket_names = {'(': 'parenthesis', '[': 'bracket', '{': 'brace'}
                    return False, f"Mismatched bracket: opened {bracket_names[open_char]} at position {open_pos}, but found closing {bracket_names.get(char, char)} at position {i}"
        
        if stack:
            open_char, open_pos = stack[-1]
            bracket_names = {'(': 'parentheses', '[': 'brackets', '{': 'braces'}
            return False, f"Unclosed {bracket_names[open_char]} opened at position {open_pos}"
        
        return True, None
    
    @staticmethod
    def _validate_patterns(expression: str) -> Tuple[bool, Optional[str]]:
        """Validate against invalid patterns."""
        for pattern, message in InputValidator.INVALID_PATTERNS:
            if re.search(pattern, expression):
                return False, message
        
        return True, None
    
    @staticmethod
    def _validate_operators(expression: str) -> Tuple[bool, Optional[str]]:
        """Validate operator positions."""
        # Check for operators at start (except + and -)
        if re.match(r'^[*/^%=<>!]', expression):
            return False, "Expression cannot start with this operator"
        
        # Check for operators at end
        if re.search(r'[+\-*/^%=<>!]$', expression):
            return False, "Expression cannot end with operator"
        
        # Check for operators after opening brackets
        if re.search(r'[({\[][+\-*/^%=<>!]', expression):
            return False, "Invalid operator immediately after opening bracket"
        
        # Check for operators before closing brackets
        if re.search(r'[+\-*/^%=<>!][)}\]]', expression):
            return False, "Invalid operator immediately before closing bracket"
        
        return True, None
    
    @staticmethod
    def _check_typos(expression: str) -> List[str]:
        """Check for common typos and return suggestions."""
        suggestions = []
        
        # Extract potential function names
        potential_functions = re.findall(r'[a-zA-Z]+', expression)
        
        for func in potential_functions:
            if func in InputValidator.COMMON_TYPOS:
                correct = InputValidator.COMMON_TYPOS[func]
                suggestions.append(f"Did you mean '{correct}' instead of '{func}'?")
            elif (func not in InputValidator.VALID_FUNCTIONS and 
                  func not in InputValidator.VALID_GREEK and 
                  func not in InputValidator.VALID_CONSTANTS and
                  len(func) > 1):  # Don't suggest for single letter variables
                # Find closest valid function
                closest = InputValidator._find_closest_function(func)
                if closest:
                    suggestions.append(f"Unknown function '{func}', did you mean '{closest}'?")
        
        return suggestions
    
    @staticmethod
    def _find_closest_function(func: str) -> Optional[str]:
        """Find the closest valid function name using edit distance."""
        def edit_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        all_valid = (InputValidator.VALID_FUNCTIONS | 
                    InputValidator.VALID_GREEK | 
                    InputValidator.VALID_CONSTANTS)
        
        min_distance = float('inf')
        closest_func = None
        
        for valid_func in all_valid:
            distance = edit_distance(func.lower(), valid_func.lower())
            if distance < min_distance and distance <= 2:  # Only suggest if close enough
                min_distance = distance
                closest_func = valid_func
        
        return closest_func
    
    @classmethod
    def validate_and_sanitize(cls, expression: str) -> str:
        """
        Validate and sanitize expression, raising exception if invalid.
        
        Args:
            expression (str): Expression to validate
            
        Returns:
            str: Sanitized expression
            
        Raises:
            ValidationError: If validation fails
        """
        is_valid, error_msg, sanitized = cls.validate_input(expression)
        if not is_valid:
            raise ValidationError(error_msg, expression=expression)
        return sanitized
    
    @classmethod
    def check_expression_complexity(cls, expression: str) -> Dict[str, int]:
        """
        Analyze expression complexity for performance estimation.
        
        Args:
            expression (str): Expression to analyze
            
        Returns:
            dict: Complexity metrics
        """
        metrics = {
            'length': len(expression),
            'operators': len(re.findall(r'[+\-*/^%=<>!]', expression)),
            'functions': len(re.findall(r'[a-zA-Z]+\s*\(', expression)),
            'brackets': len(re.findall(r'[()[\]{}]', expression)),
            'variables': len(set(re.findall(r'[a-zA-Z]+', expression))),
            'numbers': len(re.findall(r'\d+(?:\.\d+)?', expression))
        }
        
        # Calculate complexity score
        metrics['complexity_score'] = (
            metrics['length'] * 0.1 +
            metrics['operators'] * 1.0 +
            metrics['functions'] * 2.0 +
            metrics['brackets'] * 0.5 +
            metrics['variables'] * 0.3
        )
        
        return metrics