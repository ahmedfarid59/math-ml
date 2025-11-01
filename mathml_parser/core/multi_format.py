"""
Multiple Output Format Support for Mathematical Expressions
==========================================================

This module provides support for converting mathematical expressions
to various output formats including LaTeX, ASCII Math, HTML with CSS,
SVG rendering, and plain text.

Supported formats:
- MathML (default)
- LaTeX
- ASCII Math
- HTML with CSS
- SVG (basic rendering)
- Plain text
"""

from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import re
import html


class OutputFormatter(ABC):
    """
    Abstract base class for mathematical expression output formatters.
    
    This class defines the interface that all output formatters must implement
    to convert mathematical expressions from standard notation to their
    target format.
    """
    
    @abstractmethod
    def format_expression(self, expression: str) -> str:
        """
        Format the mathematical expression to the target output format.
        
        Args:
            expression: The mathematical expression to format
            
        Returns:
            The formatted expression in the target format
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement format_expression method")
    
    @abstractmethod
    def get_format_name(self) -> str:
        """
        Get the name of this output format.
        
        Returns:
            String identifier for this format (e.g., 'latex', 'ascii', 'html')
        """
        raise NotImplementedError("Subclasses must implement get_format_name method")


class LaTeXFormatter(OutputFormatter):
    """
    Formatter for LaTeX mathematical output.
    
    This formatter converts standard mathematical notation into LaTeX syntax,
    which is widely used in academic publishing and mathematical typesetting.
    
    Features:
    - Comprehensive Greek letter support
    - Proper fraction formatting with \\frac{}{}
    - Superscript and subscript handling
    - Mathematical function formatting
    - Matrix and vector notation
    - Integral, summation, and limit notation
    - Mathematical symbol conversion
    """
    
    # Conversion mappings from standard notation to LaTeX
    LATEX_MAPPINGS = {
        # Greek letters
        'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
        'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
        'ι': r'\iota', 'κ': r'\kappa', 'λ': r'\lambda', 'μ': r'\mu',
        'ν': r'\nu', 'ξ': r'\xi', 'ο': r'\omicron', 'π': r'\pi',
        'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon',
        'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega',
        
        # Capital Greek letters
        'Α': r'\Alpha', 'Β': r'\Beta', 'Γ': r'\Gamma', 'Δ': r'\Delta',
        'Ε': r'\Epsilon', 'Ζ': r'\Zeta', 'Η': r'\Eta', 'Θ': r'\Theta',
        'Ι': r'\Iota', 'Κ': r'\Kappa', 'Λ': r'\Lambda', 'Μ': r'\Mu',
        'Ν': r'\Nu', 'Ξ': r'\Xi', 'Ο': r'\Omicron', 'Π': r'\Pi',
        'Ρ': r'\Rho', 'Σ': r'\Sigma', 'Τ': r'\Tau', 'Υ': r'\Upsilon',
        'Φ': r'\Phi', 'Χ': r'\Chi', 'Ψ': r'\Psi', 'Ω': r'\Omega',
        
        # Mathematical symbols
        '∞': r'\infty', '∫': r'\int', '∬': r'\iint', '∭': r'\iiint',
        '∑': r'\sum', '∏': r'\prod', '∂': r'\partial', '∇': r'\nabla',
        '±': r'\pm', '∓': r'\mp', '≤': r'\leq', '≥': r'\geq',
        '≠': r'\neq', '≈': r'\approx', '≡': r'\equiv', '∈': r'\in',
        '∉': r'\notin', '⊂': r'\subset', '⊃': r'\supset',
        '⊆': r'\subseteq', '⊇': r'\supseteq', '∪': r'\cup', '∩': r'\cap',
        '∅': r'\emptyset', '∀': r'\forall', '∃': r'\exists',
        '∧': r'\land', '∨': r'\lor', '¬': r'\neg', '⇒': r'\Rightarrow',
        '⇔': r'\Leftrightarrow', '×': r'\times', '÷': r'\div', '·': r'\cdot',
    }
    
    def format_expression(self, expression: str) -> str:
        """
        Convert mathematical expression to LaTeX format.
        
        Args:
            expression: Mathematical expression in standard notation
            
        Returns:
            LaTeX-formatted expression wrapped in math delimiters
        """
        if not expression.strip():
            return "${}$"
            
        result = expression.strip()
        
        # Apply transformations in proper order
        result = self._handle_complex_fractions(result)
        result = self._handle_roots(result)
        result = self._handle_superscripts_subscripts(result)
        result = self._handle_functions(result)
        result = self._handle_matrices(result)
        result = self._handle_integrals_summations(result)
        result = self._handle_absolute_values(result)
        result = self._apply_symbol_mappings(result)
        result = self._handle_operator_spacing(result)
        
        return f"${result}$"
    
    def _handle_complex_fractions(self, expression: str) -> str:
        """Handle nested fractions with proper grouping."""
        # Complex fractions with parentheses: (a+b)/(c+d) -> \frac{a+b}{c+d}
        result = re.sub(r'\(([^)]+)\)/\(([^)]+)\)', r'\\frac{\1}{\2}', expression)
        
        # Mixed fractions: (a+b)/c -> \frac{a+b}{c}
        result = re.sub(r'\(([^)]+)\)/([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)', r'\\frac{\1}{\2}', result)
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)/\(([^)]+)\)', r'\\frac{\1}{\2}', result)
        
        # Simple fractions: a/b -> \frac{a}{b}
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)/([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)', r'\\frac{\1}{\2}', result)
        
        return result
    
    def _handle_roots(self, expression: str) -> str:
        """Handle square roots and nth roots."""
        result = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', expression)
        result = re.sub(r'root\(([^,]+),\s*([^)]+)\)', r'\\sqrt[\1]{\2}', result)
        result = result.replace('√', '\\sqrt')
        return result
    
    def _handle_superscripts_subscripts(self, expression: str) -> str:
        """Handle superscripts and subscripts with proper grouping."""
        # Complex superscripts with parentheses
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)\^\(([^)]+)\)', r'\1^{\2}', expression)
        # Simple superscripts
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)\^([a-zA-Z0-9α-ωΑ-Ω]+)', r'\1^{\2}', result)
        # Complex subscripts with parentheses
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)_\(([^)]+)\)', r'\1_{\2}', result)
        # Simple subscripts
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω_{}\\]+)_([a-zA-Z0-9α-ωΑ-Ω]+)', r'\1_{\2}', result)
        return result
    
    def _handle_functions(self, expression: str) -> str:
        """Handle mathematical functions."""
        functions = {
            'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan',
            'sec': '\\sec', 'csc': '\\csc', 'cot': '\\cot',
            'arcsin': '\\arcsin', 'arccos': '\\arccos', 'arctan': '\\arctan',
            'sinh': '\\sinh', 'cosh': '\\cosh', 'tanh': '\\tanh',
            'ln': '\\ln', 'log': '\\log', 'exp': '\\exp',
            'max': '\\max', 'min': '\\min', 'lim': '\\lim'
        }
        
        result = expression
        for func_name, latex_func in functions.items():
            result = re.sub(f'\\b{func_name}\\s*\\(', f'{latex_func}\\left(', result)
        return result
    
    def _handle_matrices(self, expression: str) -> str:
        """Handle matrix and vector notation."""
        def matrix_replacer(match):
            content = match.group(1)
            rows = content.split(';')
            formatted_rows = []
            for row in rows:
                cols = row.split(',')
                formatted_row = ' & '.join(col.strip() for col in cols)
                formatted_rows.append(formatted_row)
            matrix_content = ' \\\\ '.join(formatted_rows)
            return f'\\begin{{pmatrix}}{matrix_content}\\end{{pmatrix}}'
        
        result = re.sub(r'\[([^\]]+)\]', matrix_replacer, expression)
        result = re.sub(r'vec\(([^)]+)\)', r'\\vec{\1}', result)
        return result
    
    def _handle_integrals_summations(self, expression: str) -> str:
        """Handle integrals and summations with limits."""
        result = re.sub(r'∫\(([^)]+)\s+to\s+([^)]+)\)', r'\\int_{\1}^{\2}', expression)
        result = result.replace('∫', '\\int')
        result = re.sub(r'∑\(([^)]+)\s+to\s+([^)]+)\)', r'\\sum_{\1}^{\2}', result)
        result = result.replace('∑', '\\sum')
        result = re.sub(r'∏\(([^)]+)\s+to\s+([^)]+)\)', r'\\prod_{\1}^{\2}', result)
        result = result.replace('∏', '\\prod')
        result = re.sub(r'lim\(([^)]+)→([^)]+)\)', r'\\lim_{\1 \\to \2}', result)
        return result
    
    def _handle_absolute_values(self, expression: str) -> str:
        """Handle absolute values and norms."""
        result = re.sub(r'\|([^|]+)\|', r'\\left|\1\\right|', expression)
        result = re.sub(r'\|\|([^|]+)\|\|', r'\\left\\|\1\\right\\|', result)
        return result
    
    def _apply_symbol_mappings(self, expression: str) -> str:
        """Apply Greek letters and mathematical symbol mappings."""
        result = expression
        for symbol, latex in self.LATEX_MAPPINGS.items():
            result = result.replace(symbol, latex)
        return result
    
    def _handle_operator_spacing(self, expression: str) -> str:
        """Add proper spacing around operators."""
        result = re.sub(r'([a-zA-Z0-9}])\s*([+\-=<>])\s*([a-zA-Z0-9{\\])', r'\1 \2 \3', expression)
        result = result.replace('*', '\\cdot')
        return result
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "latex"


class ASCIIMathFormatter(OutputFormatter):
    """
    Formatter for ASCII Math output.
    
    ASCII Math uses plain ASCII characters to represent mathematical
    expressions in a readable format suitable for plain text environments.
    """
    
    ASCII_MAPPINGS = {
        # Greek letters
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho',
        'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
        'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
        
        # Mathematical symbols
        '∞': 'infinity', '∫': 'int', '∑': 'sum', '∏': 'prod',
        '∂': 'del', '∇': 'grad', '±': '+/-', '∓': '-/+',
        '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~~', '≡': '===',
        '∈': 'in', '∉': 'notin', '⊂': 'subset', '⊃': 'supset',
        '⊆': 'subseteq', '⊇': 'supseteq', '∪': 'uu', '∩': 'nn',
        '∅': 'O/', '∀': 'AA', '∃': 'EE', '∧': '^^', '∨': 'vv',
        '¬': 'not', '⇒': '=>', '⇔': '<==>', '×': 'xx', '÷': '-:',
        '·': '*', '√': 'sqrt'
    }
    
    def format_expression(self, expression: str) -> str:
        """Convert mathematical expression to ASCII Math format."""
        if not expression.strip():
            return ""
            
        result = expression.strip()
        
        # Apply ASCII mappings
        for symbol, ascii_equiv in self.ASCII_MAPPINGS.items():
            result = result.replace(symbol, ascii_equiv)
        
        # Handle roots
        result = re.sub(r'root\(([^,]+),\s*([^)]+)\)', r'root \1 \2', result)
        result = result.replace('√', 'sqrt')
        
        # Handle functions
        result = result.replace('arcsin', 'sin^-1')
        result = result.replace('arccos', 'cos^-1') 
        result = result.replace('arctan', 'tan^-1')
        result = result.replace('ln', 'log')
        
        # Handle limits
        result = re.sub(r'lim\(([^)]+)→([^)]+)\)', r'lim_(\1->\2)', result)
        
        # Clean spacing
        result = re.sub(r'\s*([+\-=<>])\s*', r' \1 ', result)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "ascii"


class HTMLFormatter(OutputFormatter):
    """
    Formatter for HTML with CSS output.
    
    Generates HTML markup with CSS styling to display mathematical
    expressions in web browsers with proper formatting.
    """
    
    CSS_STYLES = """
    <style>
    .math-expression {
        font-family: 'Times New Roman', 'Computer Modern', serif;
        font-size: 1.2em;
        line-height: 1.6;
        display: inline-block;
        vertical-align: middle;
    }
    .math-fraction {
        display: inline-block;
        vertical-align: middle;
        text-align: center;
        margin: 0 2px;
    }
    .math-fraction .numerator {
        display: block;
        border-bottom: 1px solid #000;
        padding: 2px 4px;
        font-size: 0.9em;
    }
    .math-fraction .denominator {
        display: block;
        padding: 2px 4px;
        font-size: 0.9em;
    }
    .math-superscript {
        vertical-align: super;
        font-size: 0.75em;
        line-height: 0;
    }
    .math-subscript {
        vertical-align: sub;
        font-size: 0.75em;
        line-height: 0;
    }
    .math-function {
        font-style: normal;
        margin-right: 2px;
    }
    .math-operator {
        margin: 0 3px;
    }
    </style>
    """
    
    def format_expression(self, expression: str) -> str:
        """Convert mathematical expression to HTML with CSS styling."""
        if not expression.strip():
            return '<div class="math-expression"></div>'
            
        result = expression.strip()
        
        # Handle fractions
        def fraction_replacer(match):
            num, den = match.group(1).strip(), match.group(2).strip()
            return (f'<div class="math-fraction">'
                   f'<div class="numerator">{num}</div>'
                   f'<div class="denominator">{den}</div></div>')
        
        result = re.sub(r'([^/\s]+)/([^/\s]+)', fraction_replacer, result)
        
        # Handle superscripts and subscripts
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω]+)\^([a-zA-Z0-9α-ωΑ-Ω]+)',
                       r'\1<span class="math-superscript">\2</span>', result)
        result = re.sub(r'([a-zA-Z0-9α-ωΑ-Ω]+)_([a-zA-Z0-9α-ωΑ-Ω]+)',
                       r'\1<span class="math-subscript">\2</span>', result)
        
        # Handle mathematical symbols
        symbol_mappings = {
            'α': '&alpha;', 'β': '&beta;', 'γ': '&gamma;', 'π': '&pi;',
            '∞': '&infin;', '∫': '&int;', '∑': '&sum;', '±': '&plusmn;',
            '≤': '&le;', '≥': '&ge;', '≠': '&ne;'
        }
        
        for symbol, html_entity in symbol_mappings.items():
            result = result.replace(symbol, html_entity)
        
        return f'<div class="math-expression">{result}</div>'
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "html"
    
    def get_css_styles(self) -> str:
        """Return the CSS styles for mathematical formatting."""
        return self.CSS_STYLES


class SVGFormatter(OutputFormatter):
    """
    Formatter for SVG mathematical output.
    
    Generates SVG markup for mathematical expressions with proper
    positioning and typography.
    """
    
    def format_expression(self, expression: str) -> str:
        """Convert mathematical expression to SVG format."""
        if not expression.strip():
            return '<svg></svg>'
            
        # Basic SVG implementation
        result = expression.strip()
        
        # Escape HTML entities
        result = html.escape(result)
        
        # Create basic SVG structure
        svg_content = f'''
        <svg width="200" height="50" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30" font-family="Times New Roman" font-size="16" fill="black">
                {result}
            </text>
        </svg>
        '''
        
        return svg_content.strip()
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "svg"


class PlainTextFormatter(OutputFormatter):
    """
    Formatter for plain text output.
    
    Converts mathematical expressions to readable plain text format
    suitable for console output and basic text environments.
    """
    
    def format_expression(self, expression: str) -> str:
        """Convert mathematical expression to plain text format."""
        if not expression.strip():
            return ""
            
        result = expression.strip()
        
        # Convert Unicode symbols to text equivalents
        text_mappings = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'π': 'pi',
            '∞': 'infinity', '∫': 'integral', '∑': 'sum',
            '±': '+/-', '≤': '<=', '≥': '>=', '≠': '!='
        }
        
        for symbol, text in text_mappings.items():
            result = result.replace(symbol, text)
        
        # Handle functions
        result = result.replace('sqrt(', 'sqrt(')
        
        return result
    
    def get_format_name(self) -> str:
        """Return the format name."""
        return "text"


class MultiFormatRenderer:
    """
    Main class for rendering mathematical expressions in multiple formats.
    
    This class coordinates multiple output formatters to provide a unified
    interface for converting mathematical expressions to various formats.
    
    Features:
    - Support for multiple output formats
    - Extensible formatter registration
    - Batch processing capabilities
    - Format validation and error handling
    """
    
    def __init__(self):
        """Initialize the multi-format renderer with default formatters."""
        self.formatters: Dict[str, OutputFormatter] = {}
        self._register_default_formatters()
    
    def _register_default_formatters(self) -> None:
        """Register the default set of formatters."""
        self.register_formatter(LaTeXFormatter())
        self.register_formatter(ASCIIMathFormatter())
        self.register_formatter(HTMLFormatter())
        self.register_formatter(SVGFormatter())
        self.register_formatter(PlainTextFormatter())
    
    def register_formatter(self, formatter: OutputFormatter) -> None:
        """
        Register a new output formatter.
        
        Args:
            formatter: The formatter instance to register
        """
        format_name = formatter.get_format_name()
        self.formatters[format_name] = formatter
    
    def get_available_formats(self) -> List[str]:
        """
        Get list of available output formats.
        
        Returns:
            List of format names
        """
        return list(self.formatters.keys())
    
    def render_expression(self, expression: str, format_type: str) -> str:
        """
        Render expression in specified format.
        
        Args:
            expression: Mathematical expression to render
            format_type: Target output format
            
        Returns:
            Formatted expression
            
        Raises:
            ValueError: If format_type is not supported
        """
        if format_type not in self.formatters:
            available = ', '.join(self.get_available_formats())
            raise ValueError(f"Format '{format_type}' not supported. Available: {available}")
        
        formatter = self.formatters[format_type]
        return formatter.format_expression(expression)
    
    def render_all_formats(self, expression: str) -> Dict[str, str]:
        """
        Render expression in all available formats.
        
        Args:
            expression: Mathematical expression to render
            
        Returns:
            Dictionary mapping format names to rendered expressions
        """
        results = {}
        for format_name, formatter in self.formatters.items():
            try:
                results[format_name] = formatter.format_expression(expression)
            except Exception as e:
                results[format_name] = f"Error: {str(e)}"
        
        return results


# Convenience functions for common operations
def render_expression(expression: str, format_type: str) -> str:
    """
    Render mathematical expression in specified format.
    
    Args:
        expression: Mathematical expression to render
        format_type: Target output format
        
    Returns:
        Formatted expression
    """
    renderer = MultiFormatRenderer()
    return renderer.render_expression(expression, format_type)


def render_all_formats(expression: str) -> Dict[str, str]:
    """
    Render mathematical expression in all available formats.
    
    Args:
        expression: Mathematical expression to render
        
    Returns:
        Dictionary mapping format names to rendered expressions
    """
    renderer = MultiFormatRenderer()
    return renderer.render_all_formats(expression)


def get_available_formats() -> List[str]:
    """
    Get list of available output formats.
    
    Returns:
        List of format names
    """
    renderer = MultiFormatRenderer()
    return renderer.get_available_formats()


# Default renderer instance for convenience
default_renderer = MultiFormatRenderer()