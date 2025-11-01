"""
Document Export Functionality
============================

This module provides functionality to export processed documents with mathematical
content to various formats including PDF, Markdown, and HTML.
"""

import os
import re
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime

from .text_processor import DocumentSection, MathExpression


class BaseExporter:
    """Base class for document exporters."""
    
    def __init__(self, title: Optional[str] = None, author: Optional[str] = None):
        """
        Initialize the exporter.
        
        Args:
            title: Document title
            author: Document author
        """
        self.title = title or "Mathematical Document"
        self.author = author or "MathML Parser"
        self.creation_date = datetime.now()
    
    def export(self, sections: List[DocumentSection], output_path: Union[str, Path]) -> None:
        """
        Export document sections to the specified output path.
        
        Args:
            sections: List of document sections
            output_path: Path for the output file
        """
        raise NotImplementedError("Subclasses must implement export method")


class MarkdownExporter(BaseExporter):
    """Exports documents to Markdown format with MathML/LaTeX math blocks."""
    
    def __init__(self, math_format: str = 'mathml', **kwargs):
        """
        Initialize the Markdown exporter.
        
        Args:
            math_format: Format for math expressions ('mathml', 'latex', 'both')
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.math_format = math_format
    
    def export(self, sections: List[DocumentSection], output_path: Union[str, Path]) -> None:
        """Export to Markdown format."""
        output_path = Path(output_path)
        
        # Generate Markdown content
        markdown_content = self._generate_markdown(sections)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _generate_markdown(self, sections: List[DocumentSection]) -> str:
        """Generate Markdown content from document sections."""
        lines = []
        
        # Add document header
        lines.append(f"# {self.title}")
        lines.append("")
        if self.author:
            lines.append(f"**Author**: {self.author}")
        lines.append(f"**Generated**: {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Process sections
        current_line = ""
        
        for section in sections:
            if section.is_math and section.math_expression:
                # Handle mathematical expression
                if current_line.strip():
                    lines.append(current_line)
                    current_line = ""
                
                math_content = self._format_math_for_markdown(section.math_expression)
                lines.append(math_content)
                lines.append("")
                
            elif section.content == '\n':
                # Handle line breaks
                if current_line.strip():
                    lines.append(current_line)
                    current_line = ""
                lines.append("")
                
            else:
                # Handle regular text
                current_line += section.content
        
        # Add any remaining content
        if current_line.strip():
            lines.append(current_line)
        
        return '\n'.join(lines)
    
    def _format_math_for_markdown(self, math_expr: MathExpression) -> str:
        """Format a mathematical expression for Markdown."""
        if self.math_format == 'mathml' and math_expr.mathml:
            return f"```mathml\n{math_expr.mathml}\n```"
        elif self.math_format == 'latex':
            # Convert to LaTeX format (simplified)
            latex_expr = self._convert_to_latex(math_expr.original_text)
            return f"$$\n{latex_expr}\n$$"
        elif self.math_format == 'both':
            result = f"**Original**: `{math_expr.original_text}`\n\n"
            if math_expr.mathml:
                result += f"**MathML**:\n```mathml\n{math_expr.mathml}\n```\n\n"
            latex_expr = self._convert_to_latex(math_expr.original_text)
            result += f"**LaTeX**:\n$$\n{latex_expr}\n$$"
            return result
        else:
            return f"`{math_expr.original_text}`"
    
    def _convert_to_latex(self, expr: str) -> str:
        """Convert expression to LaTeX format (basic conversion)."""
        # Simple conversions for common cases
        latex_expr = expr
        
        # Replace common symbols
        replacements = {
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'π': r'\pi', 'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau',
            'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega',
            '∫': r'\int', '∑': r'\sum', '∏': r'\prod', '√': r'\sqrt',
            '∂': r'\partial', '∇': r'\nabla', '≤': r'\leq', '≥': r'\geq',
            '≠': r'\neq', '≈': r'\approx', '∞': r'\infty'
        }
        
        for symbol, latex in replacements.items():
            latex_expr = latex_expr.replace(symbol, latex)
        
        # Handle power notation
        latex_expr = re.sub(r'([a-zA-Z0-9)]+)\^([a-zA-Z0-9(]+)', r'\1^{\2}', latex_expr)
        
        # Handle subscripts
        latex_expr = re.sub(r'([a-zA-Z])_([a-zA-Z0-9]+)', r'\1_{\2}', latex_expr)
        
        return latex_expr


class HTMLExporter(BaseExporter):
    """Exports documents to HTML format with embedded MathML."""
    
    def __init__(self, include_css: bool = True, math_renderer: str = 'mathml', **kwargs):
        """
        Initialize the HTML exporter.
        
        Args:
            include_css: Whether to include CSS styling
            math_renderer: Math rendering method ('mathml', 'mathjax')
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.include_css = include_css
        self.math_renderer = math_renderer
    
    def export(self, sections: List[DocumentSection], output_path: Union[str, Path]) -> None:
        """Export to HTML format."""
        output_path = Path(output_path)
        
        # Generate HTML content
        html_content = self._generate_html(sections)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html(self, sections: List[DocumentSection]) -> str:
        """Generate HTML content from document sections."""
        # HTML document structure
        html_parts = []
        
        # Document head
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="en">')
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append(f'<title>{self._escape_html(self.title)}</title>')
        
        # Include MathJax if requested
        if self.math_renderer == 'mathjax':
            html_parts.append('<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>')
            html_parts.append('<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>')
            html_parts.append('<script>')
            html_parts.append('window.MathJax = {')
            html_parts.append('  tex: { inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]] },')
            html_parts.append('  svg: { fontCache: "global" }')
            html_parts.append('};')
            html_parts.append('</script>')
        
        # Include CSS
        if self.include_css:
            html_parts.append('<style>')
            html_parts.append(self._get_css())
            html_parts.append('</style>')
        
        html_parts.append('</head>')
        html_parts.append('<body>')
        
        # Document content
        html_parts.append('<div class="document">')
        
        # Header
        html_parts.append(f'<h1>{self._escape_html(self.title)}</h1>')
        if self.author:
            html_parts.append(f'<p class="author"><strong>Author:</strong> {self._escape_html(self.author)}</p>')
        html_parts.append(f'<p class="date"><strong>Generated:</strong> {self.creation_date.strftime("%Y-%m-%d %H:%M:%S")}</p>')
        html_parts.append('<hr>')
        
        # Content
        html_parts.append('<div class="content">')
        
        # Process sections
        current_paragraph = []
        
        for section in sections:
            if section.is_math and section.math_expression:
                # Close current paragraph if needed
                if current_paragraph:
                    para_content = ''.join(current_paragraph).strip()
                    if para_content:
                        html_parts.append(f'<p>{para_content}</p>')
                    current_paragraph = []
                
                # Add mathematical expression
                math_html = self._format_math_for_html(section.math_expression)
                html_parts.append(f'<div class="math-expression">{math_html}</div>')
                
            elif section.content == '\n':
                # Handle paragraph breaks
                if current_paragraph:
                    para_content = ''.join(current_paragraph).strip()
                    if para_content:
                        html_parts.append(f'<p>{para_content}</p>')
                    current_paragraph = []
                
            else:
                # Handle regular text
                current_paragraph.append(self._escape_html(section.content))
        
        # Close any remaining paragraph
        if current_paragraph:
            para_content = ''.join(current_paragraph).strip()
            if para_content:
                html_parts.append(f'<p>{para_content}</p>')
        
        html_parts.append('</div>')  # content
        html_parts.append('</div>')  # document
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        return '\n'.join(html_parts)
    
    def _format_math_for_html(self, math_expr: MathExpression) -> str:
        """Format a mathematical expression for HTML."""
        if self.math_renderer == 'mathml' and math_expr.mathml:
            return math_expr.mathml
        elif self.math_renderer == 'mathjax':
            # Use LaTeX format for MathJax
            latex_expr = self._convert_to_latex(math_expr.original_text)
            return f'$${latex_expr}$$'
        else:
            return f'<code>{self._escape_html(math_expr.original_text)}</code>'
    
    def _convert_to_latex(self, expr: str) -> str:
        """Convert expression to LaTeX format (reuse from MarkdownExporter)."""
        # This is the same as in MarkdownExporter
        latex_expr = expr
        
        replacements = {
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'π': r'\pi', 'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau',
            'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi', 'ω': r'\omega',
            '∫': r'\int', '∑': r'\sum', '∏': r'\prod', '√': r'\sqrt',
            '∂': r'\partial', '∇': r'\nabla', '≤': r'\leq', '≥': r'\geq',
            '≠': r'\neq', '≈': r'\approx', '∞': r'\infty'
        }
        
        for symbol, latex in replacements.items():
            latex_expr = latex_expr.replace(symbol, latex)
        
        latex_expr = re.sub(r'([a-zA-Z0-9)]+)\^([a-zA-Z0-9(]+)', r'\1^{\2}', latex_expr)
        latex_expr = re.sub(r'([a-zA-Z])_([a-zA-Z0-9]+)', r'\1_{\2}', latex_expr)
        
        return latex_expr
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def _get_css(self) -> str:
        """Get CSS styles for the HTML document."""
        return """
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #fff;
            color: #333;
        }
        
        .document {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            background-color: white;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .author, .date {
            color: #7f8c8d;
            font-style: italic;
        }
        
        .content {
            margin-top: 20px;
        }
        
        p {
            margin: 1em 0;
            text-align: justify;
        }
        
        .math-expression {
            margin: 1.5em 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        
        .math-expression math {
            display: block;
            text-align: center;
        }
        
        code {
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        hr {
            border: none;
            height: 1px;
            background-color: #bdc3c7;
            margin: 20px 0;
        }
        """


class PDFExporter(BaseExporter):
    """Exports documents to PDF format using reportlab or weasyprint."""
    
    def __init__(self, use_weasyprint: bool = False, **kwargs):
        """
        Initialize the PDF exporter.
        
        Args:
            use_weasyprint: Whether to use WeasyPrint instead of ReportLab
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.use_weasyprint = use_weasyprint
        
        # Check for required libraries
        if use_weasyprint:
            try:
                import weasyprint
                self.weasyprint = weasyprint
            except ImportError:
                raise ImportError("WeasyPrint is required for PDF export. Install with: pip install weasyprint")
        else:
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                self.reportlab_available = True
            except ImportError:
                raise ImportError("ReportLab is required for PDF export. Install with: pip install reportlab")
    
    def export(self, sections: List[DocumentSection], output_path: Union[str, Path]) -> None:
        """Export to PDF format."""
        output_path = Path(output_path)
        
        if self.use_weasyprint:
            self._export_with_weasyprint(sections, output_path)
        else:
            self._export_with_reportlab(sections, output_path)
    
    def _export_with_weasyprint(self, sections: List[DocumentSection], output_path: Path) -> None:
        """Export to PDF using WeasyPrint (better MathML support)."""
        # Generate HTML first
        html_exporter = HTMLExporter(
            title=self.title, 
            author=self.author, 
            include_css=True,
            math_renderer='mathml'
        )
        
        # Create temporary HTML
        temp_html = output_path.with_suffix('.tmp.html')
        html_exporter.export(sections, temp_html)
        
        try:
            # Convert HTML to PDF
            html_doc = self.weasyprint.HTML(filename=str(temp_html))
            html_doc.write_pdf(str(output_path))
        finally:
            # Clean up temporary file
            if temp_html.exists():
                temp_html.unlink()
    
    def _export_with_reportlab(self, sections: List[DocumentSection], output_path: Path) -> None:
        """Export to PDF using ReportLab (basic support)."""
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        
        # Create document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        math_style = ParagraphStyle(
            'MathStyle',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=10,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=12,
            spaceAfter=12,
            backgroundColor='#f0f0f0'
        )
        
        story = []
        
        # Title
        title_style = styles['Title']
        story.append(Paragraph(self.title, title_style))
        story.append(Spacer(1, 12))
        
        # Author and date
        if self.author:
            story.append(Paragraph(f"<b>Author:</b> {self.author}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Content
        current_paragraph = []
        
        for section in sections:
            if section.is_math and section.math_expression:
                # Add current paragraph if any
                if current_paragraph:
                    para_text = ''.join(current_paragraph).strip()
                    if para_text:
                        story.append(Paragraph(para_text, styles['Normal']))
                    current_paragraph = []
                
                # Add math expression
                math_text = section.math_expression.original_text
                if section.math_expression.mathml:
                    # For ReportLab, we'll show both original and note about MathML
                    math_content = f"Math Expression: {math_text}<br/>(MathML available but not rendered in PDF)"
                else:
                    math_content = f"Math Expression: {math_text}"
                
                story.append(Preformatted(math_content, math_style))
                story.append(Spacer(1, 6))
                
            elif section.content == '\n':
                # End current paragraph
                if current_paragraph:
                    para_text = ''.join(current_paragraph).strip()
                    if para_text:
                        story.append(Paragraph(para_text, styles['Normal']))
                        story.append(Spacer(1, 6))
                    current_paragraph = []
                
            else:
                # Add to current paragraph
                current_paragraph.append(section.content)
        
        # Add final paragraph
        if current_paragraph:
            para_text = ''.join(current_paragraph).strip()
            if para_text:
                story.append(Paragraph(para_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)


class DocumentExportManager:
    """Manages document export to multiple formats."""
    
    def __init__(self):
        """Initialize the export manager."""
        self.exporters = {
            'markdown': MarkdownExporter,
            'html': HTMLExporter,
            'pdf': PDFExporter
        }
    
    def export_document(self, sections: List[DocumentSection], 
                       output_path: Union[str, Path], 
                       format_type: str,
                       **kwargs) -> None:
        """
        Export document to specified format.
        
        Args:
            sections: Document sections to export
            output_path: Output file path
            format_type: Export format ('markdown', 'html', 'pdf')
            **kwargs: Additional arguments for the exporter
        """
        if format_type not in self.exporters:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {list(self.exporters.keys())}")
        
        exporter_class = self.exporters[format_type]
        exporter = exporter_class(**kwargs)
        exporter.export(sections, output_path)
    
    def export_to_multiple_formats(self, sections: List[DocumentSection],
                                  base_path: Union[str, Path],
                                  formats: List[str],
                                  **kwargs) -> Dict[str, Path]:
        """
        Export document to multiple formats.
        
        Args:
            sections: Document sections to export
            base_path: Base path for output files (without extension)
            formats: List of formats to export to
            **kwargs: Additional arguments for exporters
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        base_path = Path(base_path)
        output_paths = {}
        
        format_extensions = {
            'markdown': '.md',
            'html': '.html',
            'pdf': '.pdf'
        }
        
        for format_type in formats:
            if format_type in self.exporters:
                extension = format_extensions.get(format_type, f'.{format_type}')
                output_path = base_path.with_suffix(extension)
                
                try:
                    self.export_document(sections, output_path, format_type, **kwargs)
                    output_paths[format_type] = output_path
                except Exception as e:
                    print(f"Warning: Failed to export to {format_type}: {e}")
        
        return output_paths