"""
MathML Parser - Comprehensive Mathematical Expression to MathML Converter
========================================================================

A powerful mathematical expression parser with advanced features including:

🧮 **Comprehensive Mathematical Notation**
- Basic arithmetic, advanced functions, Greek letters
- Calculus notation (derivatives, integrals, limits)
- Vector operations and matrix notation
- Number theory and logical operations

🛡️ **Robust Processing**
- Input validation with typo detection
- LaTeX input support
- Expression optimization and simplification
- Multiple output formats (MathML, LaTeX, HTML, SVG, ASCII)

⚡ **Performance & Developer Tools**
- Performance metrics and analysis
- Comprehensive error handling
- CLI interface for batch processing
- Extensive test coverage

Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "MathML Parser Team"
__license__ = "MIT"

# Import core functionality
from .core import (
    # Core parsing
    MathMLParser, parse, parse_safe, MathParseError, MathParseResult,
    ParserMetrics, default_parser,
    
    # Components
    MathematicalGrammar, EnhancedMathMLTransformer, InputValidator,
    
    # LaTeX support
    LaTeXParser, latex_to_standard,
    
    # Expression optimization
    ExpressionOptimizer, optimize_expression,
    
    # Multi-format output
    MultiFormatRenderer, render_expression, render_all_formats,
    LaTeXFormatter, ASCIIMathFormatter, HTMLFormatter, 
    SVGFormatter, PlainTextFormatter,
    
    # Document processing
    MathDocumentProcessor, process_document, process_documents,
    ProcessingOptions, ProcessingResult, MathExtractor,
    MathExpression, DocumentSection, DocumentExportManager
)

# Convenience functions for common use cases
def parse_latex(latex_expr: str) -> str:
    """
    Parse LaTeX expression and convert to MathML.
    
    Args:
        latex_expr: LaTeX mathematical expression
        
    Returns:
        MathML representation
    """
    standard_expr = latex_to_standard(latex_expr)
    return parse(standard_expr)

def parse_and_optimize(expression: str, **optimization_options) -> str:
    """
    Parse expression with optimization.
    
    Args:
        expression: Mathematical expression
        **optimization_options: Options for expression optimization
        
    Returns:
        MathML representation of optimized expression
    """
    optimized = optimize_expression(expression, **optimization_options)
    return parse(optimized)

def parse_to_format(expression: str, format_type: str = 'mathml') -> str:
    """
    Parse expression and render in specified format.
    
    Args:
        expression: Mathematical expression
        format_type: Output format ('mathml', 'latex', 'html', 'ascii', 'svg', 'text')
        
    Returns:
        Expression in specified format
    """
    if format_type == 'mathml':
        return parse(expression)
    else:
        return render_expression(expression, format_type)

def get_all_formats(expression: str) -> dict:
    """
    Get expression in all available formats.
    
    Args:
        expression: Mathematical expression
        
    Returns:
        Dictionary mapping format names to rendered expressions
    """
    result = render_all_formats(expression)
    result['mathml'] = parse(expression)
    return result

# Main exports
__all__ = [
    # Version info
    '__version__', '__author__', '__license__',
    
    # Core parsing functions
    'parse', 'parse_safe', 'parse_latex', 'parse_and_optimize', 'parse_to_format',
    
    # Core classes
    'MathMLParser', 'MathParseError', 'MathParseResult', 'ParserMetrics',
    
    # Component classes
    'MathematicalGrammar', 'EnhancedMathMLTransformer', 'InputValidator',
    
    # LaTeX support
    'LaTeXParser', 'latex_to_standard',
    
    # Optimization
    'ExpressionOptimizer', 'optimize_expression',
    
    # Multi-format output
    'MultiFormatRenderer', 'render_expression', 'render_all_formats',
    'get_all_formats',
    
    # Formatters
    'LaTeXFormatter', 'ASCIIMathFormatter', 'HTMLFormatter',
    'SVGFormatter', 'PlainTextFormatter',
    
    # Document processing
    'MathDocumentProcessor', 'process_document', 'process_documents',
    'ProcessingOptions', 'ProcessingResult', 'MathExtractor',
    'MathExpression', 'DocumentSection', 'DocumentExportManager',
    
    # Default instances
    'default_parser'
]