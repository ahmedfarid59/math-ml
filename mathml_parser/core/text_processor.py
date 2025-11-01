"""
Text File Math Extraction and Document Processing
================================================

This module provides functionality to extract mathematical expressions from text files
and process them for various output formats including PDF, Markdown, and HTML.
"""

import re
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from ..core.parser import MathMLParser
from ..core.exceptions import MathParseError


@dataclass
class MathExpression:
    """Represents a mathematical expression found in text."""
    original_text: str
    start_position: int
    end_position: int
    line_number: int
    column_start: int
    column_end: int
    context_before: str = ""
    context_after: str = ""
    mathml: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DocumentSection:
    """Represents a section of the document (text or math)."""
    content: str
    is_math: bool
    math_expression: Optional[MathExpression] = None
    line_number: int = 0


class MathExtractor:
    """
    Extracts mathematical expressions from text using various detection methods.
    """
    
    # Patterns for different types of mathematical expressions
    MATH_PATTERNS = [
        # Equations with equals
        (r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=\n]+', 'equation'),
        
        # Functions with parentheses
        (r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs|max|min|gcd|lcm)\s*\([^)]+\)', 'function'),
        
        # Power expressions
        (r'\b[a-zA-Z0-9_]+\s*[\^**]\s*[a-zA-Z0-9_\(\)]+', 'power'),
        
        # Fractions (text-based)
        (r'\b\d+/\d+\b', 'fraction_simple'),
        (r'\([^)]+\)\s*/\s*\([^)]+\)', 'fraction_complex'),
        
        # Integrals and derivatives
        (r'∫[^∫]*d[a-zA-Z]', 'integral'),
        (r'd/d[a-zA-Z]\([^)]+\)', 'derivative'),
        (r'∂/∂[a-zA-Z]\([^)]+\)', 'partial_derivative'),
        
        # Greek letters and mathematical symbols
        (r'[αβγδεζηθικλμνξοπρστυφχψω∇∂∫∑∏√±≤≥≠≈∞]', 'symbol'),
        
        # Summations and products
        (r'∑\s*\([^)]+\)', 'summation'),
        (r'∏\s*\([^)]+\)', 'product'),
        
        # Matrix notation
        (r'\[[^\]]*;[^\]]*\]', 'matrix'),
        
        # Vector notation
        (r'vec\([^)]+\)', 'vector'),
        (r'<[^>]+>', 'vector_angle'),
        
        # Complex mathematical expressions with multiple operators
        (r'[a-zA-Z0-9_\s\+\-\*/\^\(\)]+[=<>≤≥≠][a-zA-Z0-9_\s\+\-\*/\^\(\)]+', 'complex_equation'),
        
        # Subscripts and superscripts (text representation)
        (r'[a-zA-Z][_0-9]+', 'subscript'),
        (r'[a-zA-Z]\^[a-zA-Z0-9\(\)]+', 'superscript'),
    ]
    
    # Mathematical keywords that often indicate math content
    MATH_KEYWORDS = [
        'equation', 'formula', 'theorem', 'proof', 'lemma', 'corollary',
        'solve', 'calculate', 'derivative', 'integral', 'limit', 'sum',
        'product', 'matrix', 'vector', 'function', 'variable', 'coefficient',
        'polynomial', 'trigonometric', 'logarithm', 'exponential'
    ]
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize the math extractor.
        
        Args:
            confidence_threshold: Minimum confidence score for math detection
        """
        self.confidence_threshold = confidence_threshold
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), category) 
                                 for pattern, category in self.MATH_PATTERNS]
    
    def extract_from_text(self, text: str) -> List[MathExpression]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Input text content
            
        Returns:
            List of detected mathematical expressions
        """
        expressions = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_expressions = self._extract_from_line(line, line_num, lines)
            expressions.extend(line_expressions)
        
        # Remove duplicates and overlapping expressions
        expressions = self._remove_overlaps(expressions)
        
        # Filter by confidence threshold
        expressions = [expr for expr in expressions if expr.confidence >= self.confidence_threshold]
        
        return expressions
    
    def _extract_from_line(self, line: str, line_num: int, all_lines: List[str]) -> List[MathExpression]:
        """Extract mathematical expressions from a single line."""
        expressions = []
        
        # Try each pattern
        for pattern, category in self.compiled_patterns:
            for match in pattern.finditer(line):
                start, end = match.span()
                expr_text = match.group()
                
                # Calculate confidence based on various factors
                confidence = self._calculate_confidence(expr_text, line, category, all_lines, line_num)
                
                # Get context
                context_before = line[:start][-20:] if start > 20 else line[:start]
                context_after = line[end:end+20] if end + 20 < len(line) else line[end:]
                
                expr = MathExpression(
                    original_text=expr_text,
                    start_position=start,
                    end_position=end,
                    line_number=line_num,
                    column_start=start,
                    column_end=end,
                    context_before=context_before,
                    context_after=context_after,
                    confidence=confidence
                )
                
                expressions.append(expr)
        
        return expressions
    
    def _calculate_confidence(self, expr_text: str, line: str, category: str, 
                            all_lines: List[str], line_num: int) -> float:
        """Calculate confidence score for a potential mathematical expression."""
        confidence = 0.0
        
        # Base confidence by category
        category_weights = {
            'equation': 0.8,
            'function': 0.7,
            'integral': 0.9,
            'derivative': 0.9,
            'partial_derivative': 0.9,
            'symbol': 0.6,
            'complex_equation': 0.8,
            'matrix': 0.7,
            'vector': 0.7,
            'power': 0.6,
            'fraction_complex': 0.7,
            'fraction_simple': 0.4,
            'summation': 0.8,
            'product': 0.8,
            'subscript': 0.3,
            'superscript': 0.5
        }
        
        confidence += category_weights.get(category, 0.5)
        
        # Boost confidence if math keywords are nearby
        surrounding_text = ""
        if line_num > 1:
            surrounding_text += all_lines[line_num - 2] + " "
        surrounding_text += all_lines[line_num - 1] + " "
        if line_num < len(all_lines):
            surrounding_text += all_lines[line_num]
        
        for keyword in self.MATH_KEYWORDS:
            if keyword.lower() in surrounding_text.lower():
                confidence += 0.1
                break
        
        # Boost for mathematical symbols
        math_symbols = ['=', '≠', '≤', '≥', '±', '∞', '∫', '∑', '∏', '∂', '∇']
        symbol_count = sum(1 for symbol in math_symbols if symbol in expr_text)
        confidence += min(symbol_count * 0.1, 0.3)
        
        # Penalize very short expressions unless they're symbols
        if len(expr_text.strip()) < 3 and category not in ['symbol']:
            confidence -= 0.2
        
        # Boost for standalone expressions (on their own line or well-separated)
        if line.strip() == expr_text.strip():
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _remove_overlaps(self, expressions: List[MathExpression]) -> List[MathExpression]:
        """Remove overlapping expressions, keeping the one with higher confidence."""
        if not expressions:
            return expressions
        
        # Sort by line number, then by start position
        expressions.sort(key=lambda x: (x.line_number, x.start_position))
        
        filtered = []
        for expr in expressions:
            # Check for overlaps with already accepted expressions
            overlaps = False
            for accepted in filtered:
                if (expr.line_number == accepted.line_number and
                    not (expr.end_position <= accepted.start_position or
                         expr.start_position >= accepted.end_position)):
                    # There's an overlap
                    overlaps = True
                    # If current expression has higher confidence, replace the accepted one
                    if expr.confidence > accepted.confidence:
                        filtered.remove(accepted)
                        overlaps = False
                    break
            
            if not overlaps:
                filtered.append(expr)
        
        return filtered


class TextDocumentProcessor:
    """
    Processes text documents to extract and convert mathematical expressions.
    """
    
    def __init__(self, parser: Optional[MathMLParser] = None):
        """
        Initialize the document processor.
        
        Args:
            parser: MathML parser instance (creates default if None)
        """
        self.parser = parser or MathMLParser()
        self.extractor = MathExtractor()
    
    def process_file(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> List[DocumentSection]:
        """
        Process a text file to extract and convert mathematical expressions.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding
            
        Returns:
            List of document sections (text and math)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return self.process_text(content)
    
    def process_text(self, text: str) -> List[DocumentSection]:
        """
        Process text content to extract and convert mathematical expressions.
        
        Args:
            text: Input text content
            
        Returns:
            List of document sections
        """
        # Extract mathematical expressions
        math_expressions = self.extractor.extract_from_text(text)
        
        # Convert expressions to MathML
        for expr in math_expressions:
            try:
                result = self.parser.parse_safe(expr.original_text)
                if result.success:
                    expr.mathml = result.mathml
                else:
                    print(f"Warning: Could not parse math expression: {expr.original_text}")
            except Exception as e:
                print(f"Error parsing expression '{expr.original_text}': {e}")
        
        # Create document sections
        sections = self._create_sections(text, math_expressions)
        
        return sections
    
    def _create_sections(self, text: str, math_expressions: List[MathExpression]) -> List[DocumentSection]:
        """Create document sections from text and math expressions."""
        sections = []
        lines = text.split('\n')
        
        # Group expressions by line
        expressions_by_line = {}
        for expr in math_expressions:
            if expr.line_number not in expressions_by_line:
                expressions_by_line[expr.line_number] = []
            expressions_by_line[expr.line_number].append(expr)
        
        for line_num, line in enumerate(lines, 1):
            if line_num in expressions_by_line:
                # Line contains math expressions
                line_expressions = sorted(expressions_by_line[line_num], 
                                        key=lambda x: x.start_position)
                
                current_pos = 0
                for expr in line_expressions:
                    # Add text before the math expression
                    if expr.start_position > current_pos:
                        text_content = line[current_pos:expr.start_position]
                        if text_content.strip():
                            sections.append(DocumentSection(
                                content=text_content,
                                is_math=False,
                                line_number=line_num
                            ))
                    
                    # Add the math expression
                    sections.append(DocumentSection(
                        content=expr.original_text,
                        is_math=True,
                        math_expression=expr,
                        line_number=line_num
                    ))
                    
                    current_pos = expr.end_position
                
                # Add remaining text on the line
                if current_pos < len(line):
                    remaining_text = line[current_pos:]
                    if remaining_text.strip():
                        sections.append(DocumentSection(
                            content=remaining_text,
                            is_math=False,
                            line_number=line_num
                        ))
            else:
                # Line contains only text
                if line.strip():  # Skip empty lines for now
                    sections.append(DocumentSection(
                        content=line,
                        is_math=False,
                        line_number=line_num
                    ))
            
            # Add line break (except for last line)
            if line_num < len(lines):
                sections.append(DocumentSection(
                    content='\n',
                    is_math=False,
                    line_number=line_num
                ))
        
        return sections
    
    def get_math_summary(self, sections: List[DocumentSection]) -> Dict[str, any]:
        """Get a summary of mathematical content in the document."""
        math_sections = [s for s in sections if s.is_math]
        
        return {
            'total_expressions': len(math_sections),
            'successful_conversions': len([s for s in math_sections 
                                         if s.math_expression and s.math_expression.mathml]),
            'failed_conversions': len([s for s in math_sections 
                                     if s.math_expression and not s.math_expression.mathml]),
            'average_confidence': (sum(s.math_expression.confidence for s in math_sections) / 
                                 len(math_sections)) if math_sections else 0.0,
            'expressions_by_line': {s.line_number: s.content for s in math_sections}
        }