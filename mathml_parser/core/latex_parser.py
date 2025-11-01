"""
LaTeX Input Parser for MathML Conversion
========================================

This module provides support for parsing LaTeX mathematical expressions
and converting them to our internal format for MathML generation.

Supports common LaTeX mathematical notation including:
- Fractions: \frac{a}{b}
- Square roots: \sqrt{x}, \sqrt[n]{x}
- Superscripts/subscripts: x^{2}, x_{i}
- Greek letters: \alpha, \beta, \gamma, etc.
- Functions: \sin, \cos, \tan, \log, \ln
- Integrals: \int, \iint, \iiint
- Summations: \sum, \prod
- Derivatives: \frac{d}{dx}, \frac{\partial}{\partial x}
"""

import re
from typing import Dict, List, Tuple, Optional


class LaTeXParser:
    """
    Parser for converting LaTeX mathematical expressions to standard format.
    
    This parser converts LaTeX notation to our standard mathematical notation
    which can then be processed by the main MathML parser.
    """
    
    # LaTeX command mappings to standard notation
    LATEX_COMMANDS = {
        # Fractions
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        
        # Square roots
        r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
        r'\\sqrt\[([^]]+)\]\{([^}]+)\}': r'root(\1, \2)',
        
        # Greek letters
        r'\\alpha\b': 'α',
        r'\\beta\b': 'β', 
        r'\\gamma\b': 'γ',
        r'\\delta\b': 'δ',
        r'\\epsilon\b': 'ε',
        r'\\zeta\b': 'ζ',
        r'\\eta\b': 'η',
        r'\\theta\b': 'θ',
        r'\\iota\b': 'ι',
        r'\\kappa\b': 'κ',
        r'\\lambda\b': 'λ',
        r'\\mu\b': 'μ',
        r'\\nu\b': 'ν',
        r'\\xi\b': 'ξ',
        r'\\omicron\b': 'ο',
        r'\\pi\b': 'π',
        r'\\rho\b': 'ρ',
        r'\\sigma\b': 'σ',
        r'\\tau\b': 'τ',
        r'\\upsilon\b': 'υ',
        r'\\phi\b': 'φ',
        r'\\chi\b': 'χ',
        r'\\psi\b': 'ψ',
        r'\\omega\b': 'ω',
        
        # Capital Greek letters
        r'\\Alpha\b': 'Α',
        r'\\Beta\b': 'Β',
        r'\\Gamma\b': 'Γ',
        r'\\Delta\b': 'Δ',
        r'\\Epsilon\b': 'Ε',
        r'\\Zeta\b': 'Ζ',
        r'\\Eta\b': 'Η',
        r'\\Theta\b': 'Θ',
        r'\\Iota\b': 'Ι',
        r'\\Kappa\b': 'Κ',
        r'\\Lambda\b': 'Λ',
        r'\\Mu\b': 'Μ',
        r'\\Nu\b': 'Ν',
        r'\\Xi\b': 'Ξ',
        r'\\Omicron\b': 'Ο',
        r'\\Pi\b': 'Π',
        r'\\Rho\b': 'Ρ',
        r'\\Sigma\b': 'Σ',
        r'\\Tau\b': 'Τ',
        r'\\Upsilon\b': 'Υ',
        r'\\Phi\b': 'Φ',
        r'\\Chi\b': 'Χ',
        r'\\Psi\b': 'Ψ',
        r'\\Omega\b': 'Ω',
        
        # Trigonometric functions
        r'\\sin\b': 'sin',
        r'\\cos\b': 'cos',
        r'\\tan\b': 'tan',
        r'\\sec\b': 'sec',
        r'\\csc\b': 'csc',
        r'\\cot\b': 'cot',
        r'\\arcsin\b': 'arcsin',
        r'\\arccos\b': 'arccos',
        r'\\arctan\b': 'arctan',
        
        # Hyperbolic functions
        r'\\sinh\b': 'sinh',
        r'\\cosh\b': 'cosh',
        r'\\tanh\b': 'tanh',
        
        # Logarithmic functions
        r'\\ln\b': 'ln',
        r'\\log\b': 'log',
        r'\\exp\b': 'exp',
        
        # Mathematical operators
        r'\\cdot\b': '*',
        r'\\times\b': '×',
        r'\\div\b': '÷',
        r'\\pm\b': '±',
        r'\\mp\b': '∓',
        
        # Comparison operators
        r'\\leq\b': '≤',
        r'\\geq\b': '≥',
        r'\\neq\b': '≠',
        r'\\approx\b': '≈',
        r'\\equiv\b': '≡',
        
        # Calculus notation
        r'\\int\b': '∫',
        r'\\iint\b': '∬',
        r'\\iiint\b': '∭',
        r'\\sum\b': '∑',
        r'\\prod\b': '∏',
        r'\\lim\b': 'lim',
        r'\\partial\b': '∂',
        
        # Set theory
        r'\\in\b': '∈',
        r'\\notin\b': '∉',
        r'\\subset\b': '⊂',
        r'\\supset\b': '⊃',
        r'\\subseteq\b': '⊆',
        r'\\supseteq\b': '⊇',
        r'\\cup\b': '∪',
        r'\\cap\b': '∩',
        r'\\emptyset\b': '∅',
        
        # Logic
        r'\\forall\b': '∀',
        r'\\exists\b': '∃',
        r'\\land\b': '∧',
        r'\\lor\b': '∨',
        r'\\neg\b': '¬',
        r'\\Rightarrow\b': '⇒',
        r'\\Leftrightarrow\b': '⇔',
        
        # Mathematical constants
        r'\\infty\b': '∞',
        r'\\e\b': 'e',
        
        # Vector notation
        r'\\vec\{([^}]+)\}': r'vec(\1)',
        r'\\nabla\b': '∇',
        
        # Absolute value and norms
        r'\\left\|([^|]+)\\right\|': r'||\1||',
        r'\\left\|([^|]+)\|': r'|\1|',
        r'\|([^|]+)\|': r'|\1|',
        
        # Special brackets
        r'\\left\(': '(',
        r'\\right\)': ')',
        r'\\left\[': '[',
        r'\\right\]': ']',
        r'\\left\{': '{',
        r'\\right\}': '}',
    }
    
    # Complex LaTeX patterns that need special handling
    COMPLEX_PATTERNS = [
        # Derivatives
        (r'\\frac\{d\}\{d([^}]+)\}', r'd/d\1'),
        (r'\\frac\{\\partial\}\{\\partial\s*([^}]+)\}', r'∂/∂\1'),
        (r'\\frac\{d^2\}\{d([^}]+)^2\}', r'd²/d\1²'),
        (r'\\frac\{\\partial^2\}\{\\partial\s*([^}]+)^2\}', r'∂²/∂\1²'),
        
        # Integrals with limits
        (r'\\int_\{([^}]+)\}\^\{([^}]+)\}', r'∫ from \1 to \2'),
        (r'\\sum_\{([^}]+)\}\^\{([^}]+)\}', r'∑(\1 to \2)'),
        (r'\\prod_\{([^}]+)\}\^\{([^}]+)\}', r'∏(\1 to \2)'),
        
        # Limits
        (r'\\lim_\{([^}]+)\\to\s*([^}]+)\}', r'lim(\1, \2)'),
        (r'\\lim_\{([^}]+)\\rightarrow\s*([^}]+)\}', r'lim(\1, \2)'),
        
        # Matrix notation
        (r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', r'[\1]'),
        (r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', r'[\1]'),
        (r'\\begin\{matrix\}(.*?)\\end\{matrix\}', r'[\1]'),
        
        # Binomial coefficients
        (r'\\binom\{([^}]+)\}\{([^}]+)\}', r'C(\1, \2)'),
        (r'\\choose', r','),
    ]
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.DOTALL), replacement)
            for pattern, replacement in self.COMPLEX_PATTERNS
        ]
    
    def parse_latex(self, latex_expr: str) -> str:
        """
        Convert LaTeX mathematical expression to standard notation.
        
        Args:
            latex_expr: LaTeX mathematical expression
            
        Returns:
            Standard mathematical notation string
        """
        # Clean up the input
        result = self._clean_latex_input(latex_expr)
        
        # Apply complex pattern transformations first
        result = self._apply_complex_patterns(result)
        
        # Handle superscripts and subscripts
        result = self._convert_scripts(result)
        
        # Apply simple command mappings
        result = self._apply_simple_mappings(result)
        
        # Clean up the result
        result = self._clean_output(result)
        
        return result
    
    def _clean_latex_input(self, latex_expr: str) -> str:
        """Clean and prepare LaTeX input for processing."""
        # Remove common LaTeX environments
        result = re.sub(r'\\begin\{equation\*?\}|\\end\{equation\*?\}', '', latex_expr)
        result = re.sub(r'\\begin\{align\*?\}|\\end\{align\*?\}', '', result)
        result = re.sub(r'\$+', '', result)  # Remove $ delimiters
        
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def _apply_complex_patterns(self, expr: str) -> str:
        """Apply complex pattern transformations."""
        result = expr
        
        for pattern, replacement in self.compiled_patterns:
            result = pattern.sub(replacement, result)
        
        return result
    
    def _convert_scripts(self, expr: str) -> str:
        """Convert LaTeX superscripts and subscripts."""
        # Handle superscripts x^{content}
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω)]+)\^\{([^}]+)\}', r'\1^(\2)', expr)
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω)]+)\^([a-zA-Z0-9α-ωΑ-Ω]+)', r'\1^\2', result)
        
        # Handle subscripts x_{content}
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω)]+)_\{([^}]+)\}', r'\1_\2', result)
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω)]+)_([a-zA-Z0-9α-ωΑ-Ω]+)', r'\1_\2', result)
        
        return result
    
    def _apply_simple_mappings(self, expr: str) -> str:
        """Apply simple command mappings."""
        result = expr
        
        for pattern, replacement in self.LATEX_COMMANDS.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _clean_output(self, expr: str) -> str:
        """Clean up the final output."""
        # Remove extra braces
        result = re.sub(r'\{([^{}]*)\}', r'\1', expr)
        
        # Fix spacing around operators
        result = re.sub(r'\s*([+\-*/=<>≤≥≠])\s*', r' \1 ', result)
        
        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def convert_matrix_latex(self, matrix_content: str) -> str:
        """Convert LaTeX matrix content to standard notation."""
        # Replace \\ with ; for row separation
        result = re.sub(r'\\\\', ';', matrix_content)
        
        # Replace & with , for column separation
        result = re.sub(r'&', ',', result)
        
        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def is_latex_expression(self, expr: str) -> bool:
        """
        Check if an expression contains LaTeX notation.
        
        Args:
            expr: Expression to check
            
        Returns:
            True if expression contains LaTeX commands
        """
        latex_indicators = [
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'\{.*\}',       # Braces
            r'\$',           # Dollar signs
            r'\\begin',      # Environment starts
            r'\\end',        # Environment ends
        ]
        
        return any(re.search(pattern, expr) for pattern in latex_indicators)


# Convenience function for easy access
def latex_to_standard(latex_expr: str) -> str:
    """
    Convert LaTeX expression to standard mathematical notation.
    
    Args:
        latex_expr: LaTeX mathematical expression
        
    Returns:
        Standard mathematical notation
    """
    parser = LaTeXParser()
    return parser.parse_latex(latex_expr)