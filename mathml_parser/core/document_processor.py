"""
Complete Document Processing System
==================================

This module provides a unified interface for processing text documents,
extracting mathematical content, and exporting to various formats.
"""

import os
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

from .text_processor import TextDocumentProcessor, DocumentSection, MathExpression
from .document_exporters import DocumentExportManager, MarkdownExporter, HTMLExporter, PDFExporter
from .parser import MathMLParser


@dataclass
class ProcessingOptions:
    """Options for document processing."""
    confidence_threshold: float = 0.3
    encoding: str = 'utf-8'
    math_format: str = 'mathml'  # 'mathml', 'latex', 'both'
    include_css: bool = True
    math_renderer: str = 'mathml'  # 'mathml', 'mathjax'
    use_weasyprint: bool = False
    title: Optional[str] = None
    author: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of document processing."""
    input_file: Path
    sections: List[DocumentSection]
    math_summary: Dict[str, any]
    export_paths: Dict[str, Path]
    success: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class MathDocumentProcessor:
    """
    Complete document processing system for extracting and converting mathematical content.
    
    This class coordinates the entire pipeline:
    1. Reading text files
    2. Extracting mathematical expressions
    3. Converting to MathML
    4. Exporting to various formats (PDF, Markdown, HTML)
    """
    
    def __init__(self, parser: Optional[MathMLParser] = None, 
                 options: Optional[ProcessingOptions] = None):
        """
        Initialize the document processor.
        
        Args:
            parser: MathML parser instance
            options: Processing options
        """
        self.parser = parser or MathMLParser()
        self.options = options or ProcessingOptions()
        self.text_processor = TextDocumentProcessor(self.parser)
        self.export_manager = DocumentExportManager()
        
        # Update extractor confidence threshold
        self.text_processor.extractor.confidence_threshold = self.options.confidence_threshold
    
    def process_file(self, input_path: Union[str, Path], 
                    output_formats: List[str],
                    output_dir: Optional[Union[str, Path]] = None) -> ProcessingResult:
        """
        Process a single text file and export to specified formats.
        
        Args:
            input_path: Path to input text file
            output_formats: List of output formats ('pdf', 'html', 'markdown')
            output_dir: Output directory (defaults to same as input file)
            
        Returns:
            ProcessingResult with details of the processing
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            return ProcessingResult(
                input_file=input_path,
                sections=[],
                math_summary={},
                export_paths={},
                success=False,
                errors=[f"Input file not found: {input_path}"]
            )
        
        # Set output directory
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        errors = []
        
        try:
            # Process the text file
            sections = self.text_processor.process_file(input_path, self.options.encoding)
            
            # Get math summary
            math_summary = self.text_processor.get_math_summary(sections)
            
            # Generate base output path
            base_output_path = output_dir / input_path.stem
            
            # Export to requested formats
            export_paths = {}
            
            for format_type in output_formats:
                try:
                    export_kwargs = self._get_export_kwargs(format_type)
                    
                    if format_type == 'markdown':
                        output_path = base_output_path.with_suffix('.md')
                        self.export_manager.export_document(
                            sections, output_path, 'markdown', **export_kwargs
                        )
                        export_paths['markdown'] = output_path
                        
                    elif format_type == 'html':
                        output_path = base_output_path.with_suffix('.html')
                        self.export_manager.export_document(
                            sections, output_path, 'html', **export_kwargs
                        )
                        export_paths['html'] = output_path
                        
                    elif format_type == 'pdf':
                        output_path = base_output_path.with_suffix('.pdf')
                        self.export_manager.export_document(
                            sections, output_path, 'pdf', **export_kwargs
                        )
                        export_paths['pdf'] = output_path
                        
                    else:
                        errors.append(f"Unsupported output format: {format_type}")
                        
                except Exception as e:
                    errors.append(f"Failed to export to {format_type}: {str(e)}")
            
            return ProcessingResult(
                input_file=input_path,
                sections=sections,
                math_summary=math_summary,
                export_paths=export_paths,
                success=len(export_paths) > 0,
                errors=errors
            )
            
        except Exception as e:
            return ProcessingResult(
                input_file=input_path,
                sections=[],
                math_summary={},
                export_paths={},
                success=False,
                errors=[f"Processing failed: {str(e)}"]
            )
    
    def process_multiple_files(self, input_paths: List[Union[str, Path]],
                             output_formats: List[str],
                             output_dir: Optional[Union[str, Path]] = None) -> List[ProcessingResult]:
        """
        Process multiple text files.
        
        Args:
            input_paths: List of input file paths
            output_formats: List of output formats
            output_dir: Output directory
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for input_path in input_paths:
            result = self.process_file(input_path, output_formats, output_dir)
            results.append(result)
        
        return results
    
    def process_directory(self, input_dir: Union[str, Path],
                         output_formats: List[str],
                         output_dir: Optional[Union[str, Path]] = None,
                         file_patterns: List[str] = None) -> List[ProcessingResult]:
        """
        Process all text files in a directory.
        
        Args:
            input_dir: Input directory path
            output_formats: List of output formats
            output_dir: Output directory
            file_patterns: File patterns to match (e.g., ['*.txt', '*.md'])
            
        Returns:
            List of ProcessingResult objects
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists() or not input_dir.is_dir():
            return [ProcessingResult(
                input_file=input_dir,
                sections=[],
                math_summary={},
                export_paths={},
                success=False,
                errors=[f"Input directory not found: {input_dir}"]
            )]
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ['*.txt', '*.md', '*.text']
        
        # Find matching files
        input_files = []
        for pattern in file_patterns:
            input_files.extend(input_dir.glob(pattern))
        
        if not input_files:
            return [ProcessingResult(
                input_file=input_dir,
                sections=[],
                math_summary={},
                export_paths={},
                success=False,
                errors=[f"No matching files found in {input_dir}"]
            )]
        
        return self.process_multiple_files(input_files, output_formats, output_dir)
    
    def _get_export_kwargs(self, format_type: str) -> Dict[str, any]:
        """Get export arguments based on format type and options."""
        base_kwargs = {
            'title': self.options.title,
            'author': self.options.author
        }
        
        if format_type == 'markdown':
            return {
                **base_kwargs,
                'math_format': self.options.math_format
            }
        elif format_type == 'html':
            return {
                **base_kwargs,
                'include_css': self.options.include_css,
                'math_renderer': self.options.math_renderer
            }
        elif format_type == 'pdf':
            return {
                **base_kwargs,
                'use_weasyprint': self.options.use_weasyprint
            }
        else:
            return base_kwargs
    
    def get_processing_summary(self, results: List[ProcessingResult]) -> Dict[str, any]:
        """Get a summary of processing results."""
        total_files = len(results)
        successful_files = len([r for r in results if r.success])
        failed_files = total_files - successful_files
        
        total_math_expressions = sum(r.math_summary.get('total_expressions', 0) for r in results)
        successful_conversions = sum(r.math_summary.get('successful_conversions', 0) for r in results)
        failed_conversions = sum(r.math_summary.get('failed_conversions', 0) for r in results)
        
        # Collect all export formats generated
        export_formats = set()
        for result in results:
            export_formats.update(result.export_paths.keys())
        
        # Collect all errors
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)
        
        return {
            'files_processed': total_files,
            'files_successful': successful_files,
            'files_failed': failed_files,
            'total_math_expressions': total_math_expressions,
            'successful_math_conversions': successful_conversions,
            'failed_math_conversions': failed_conversions,
            'conversion_success_rate': (successful_conversions / total_math_expressions * 100) 
                                     if total_math_expressions > 0 else 0.0,
            'export_formats_generated': list(export_formats),
            'errors': all_errors
        }
    
    def preview_math_extraction(self, input_path: Union[str, Path]) -> List[MathExpression]:
        """
        Preview mathematical expressions that would be extracted from a file.
        
        Args:
            input_path: Path to input text file
            
        Returns:
            List of detected mathematical expressions
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Read file content
        with open(input_path, 'r', encoding=self.options.encoding) as f:
            content = f.read()
        
        # Extract mathematical expressions
        return self.text_processor.extractor.extract_from_text(content)
    
    def validate_math_expressions(self, expressions: List[MathExpression]) -> Dict[str, any]:
        """
        Validate a list of mathematical expressions by trying to parse them.
        
        Args:
            expressions: List of mathematical expressions
            
        Returns:
            Validation summary
        """
        results = {
            'total': len(expressions),
            'valid': 0,
            'invalid': 0,
            'validation_details': []
        }
        
        for expr in expressions:
            try:
                parse_result = self.parser.parse_safe(expr.original_text)
                if parse_result.success:
                    results['valid'] += 1
                    status = 'valid'
                    error = None
                else:
                    results['invalid'] += 1
                    status = 'invalid'
                    error = str(parse_result.error) if parse_result.error else "Unknown error"
            except Exception as e:
                results['invalid'] += 1
                status = 'invalid'
                error = str(e)
            
            results['validation_details'].append({
                'expression': expr.original_text,
                'confidence': expr.confidence,
                'line_number': expr.line_number,
                'status': status,
                'error': error
            })
        
        results['success_rate'] = (results['valid'] / results['total'] * 100) if results['total'] > 0 else 0.0
        
        return results


def process_document(input_path: Union[str, Path], 
                    output_formats: List[str] = None,
                    output_dir: Optional[Union[str, Path]] = None,
                    **options) -> ProcessingResult:
    """
    Convenience function to process a single document.
    
    Args:
        input_path: Path to input text file
        output_formats: List of output formats (defaults to ['html'])
        output_dir: Output directory
        **options: Processing options
        
    Returns:
        ProcessingResult
    """
    if output_formats is None:
        output_formats = ['html']
    
    processing_options = ProcessingOptions(**options)
    processor = MathDocumentProcessor(options=processing_options)
    
    return processor.process_file(input_path, output_formats, output_dir)


def process_documents(input_paths: List[Union[str, Path]],
                     output_formats: List[str] = None,
                     output_dir: Optional[Union[str, Path]] = None,
                     **options) -> List[ProcessingResult]:
    """
    Convenience function to process multiple documents.
    
    Args:
        input_paths: List of input file paths
        output_formats: List of output formats (defaults to ['html'])
        output_dir: Output directory
        **options: Processing options
        
    Returns:
        List of ProcessingResult objects
    """
    if output_formats is None:
        output_formats = ['html']
    
    processing_options = ProcessingOptions(**options)
    processor = MathDocumentProcessor(options=processing_options)
    
    return processor.process_multiple_files(input_paths, output_formats, output_dir)