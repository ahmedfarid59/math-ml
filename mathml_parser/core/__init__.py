"""
Core module for MathML Parser package.

Contains the main parsing engine, grammar definitions, transformers,
optimization tools, LaTeX support, and multi-format output capabilities.
"""

from .grammar import MathematicalGrammar
from .exceptions import MathParseError, MathParseResult
from .validator import InputValidator
from .transformer import EnhancedMathMLTransformer, MathMLTransformer
from .parser import MathMLParser, ParserMetrics, default_parser, parse, parse_safe
from .latex_parser import LaTeXParser, latex_to_standard
from .optimizer import ExpressionOptimizer, optimize_expression
from .multi_format import (
    MultiFormatRenderer, LaTeXFormatter, ASCIIMathFormatter, 
    HTMLFormatter, SVGFormatter, PlainTextFormatter,
    render_expression, render_all_formats
)

# New document processing features
from .text_processor import (
    TextDocumentProcessor, MathExtractor, MathExpression, DocumentSection
)
from .document_exporters import (
    DocumentExportManager, MarkdownExporter, HTMLExporter, PDFExporter
)
from .document_processor import (
    MathDocumentProcessor, ProcessingOptions, ProcessingResult,
    process_document, process_documents
)

__all__ = [
    # Core parsing
    'MathematicalGrammar',
    'MathParseError',
    'MathParseResult', 
    'InputValidator',
    'EnhancedMathMLTransformer',
    'MathMLTransformer',
    'MathMLParser',
    'ParserMetrics',
    'default_parser',
    'parse',
    'parse_safe',
    
    # LaTeX support
    'LaTeXParser',
    'latex_to_standard',
    
    # Expression optimization
    'ExpressionOptimizer',
    'optimize_expression',
    
    # Multi-format output
    'MultiFormatRenderer',
    'LaTeXFormatter',
    'ASCIIMathFormatter', 
    'HTMLFormatter',
    'SVGFormatter',
    'PlainTextFormatter',
    'render_expression',
    'render_all_formats',
    
    # Document processing
    'TextDocumentProcessor',
    'MathExtractor',
    'MathExpression',
    'DocumentSection',
    'DocumentExportManager',
    'MarkdownExporter',
    'HTMLExporter', 
    'PDFExporter',
    'MathDocumentProcessor',
    'ProcessingOptions',
    'ProcessingResult',
    'process_document',
    'process_documents'
]