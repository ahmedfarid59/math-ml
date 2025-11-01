#!/usr/bin/env python3
"""
Tests for document processing functionality of the MathML parser.

This module tests mathematical expression extraction from text files,
document processing workflows, and export format generation.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from mathml_parser.core.text_processor import MathExtractor, MathExpression
from mathml_parser.core.document_processor import MathDocumentProcessor, ProcessingOptions, ProcessingResult
from mathml_parser.core.document_exporters import MarkdownExporter, HTMLExporter, PDFExporter


class TestMathExtractor(unittest.TestCase):
    """Test the MathExtractor class for detecting mathematical expressions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MathExtractor()
    
    def test_basic_math_detection(self):
        """Test detection of basic mathematical expressions."""
        text = """
        The quadratic formula is x = (-b ± √(b² - 4ac)) / (2a)
        For the equation y = mx + b, the slope is m.
        """
        
        expressions = self.extractor.extract_from_text(text)
        
        # Should find the quadratic formula
        self.assertGreater(len(expressions), 0)
        
        # Check that we found the quadratic formula
        quadratic_found = any("(-b ± √(b² - 4ac))" in expr.original_text 
                            for expr in expressions)
        self.assertTrue(quadratic_found, "Should detect quadratic formula")
    
    def test_calculus_expressions(self):
        """Test detection of calculus expressions."""
        text = """
        The derivative of f(x) = x² is f'(x) = 2x
        The integral ∫₀¹ x² dx = 1/3
        The limit lim(x→0) sin(x)/x = 1
        """
        
        expressions = self.extractor.extract_from_text(text)
        
        # Should find multiple calculus expressions
        self.assertGreaterEqual(len(expressions), 3)
        
        # Check specific patterns
        derivative_found = any("f'(x) = 2x" in expr.original_text 
                             for expr in expressions)
        integral_found = any("∫" in expr.original_text 
                           for expr in expressions)
        
        self.assertTrue(derivative_found, "Should detect derivative")
        self.assertTrue(integral_found, "Should detect integral")
    
    def test_confidence_scoring(self):
        """Test confidence scoring for mathematical expressions."""
        # High confidence mathematical expressions
        high_conf_text = "The quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)"
        high_expressions = self.extractor.extract_from_text(high_conf_text)
        
        # Low confidence (possibly mathematical) expressions
        low_conf_text = "The meeting is at 3:30 PM and costs $25.50"
        low_expressions = self.extractor.extract_from_text(low_conf_text, min_confidence=0.1)
        
        if high_expressions:
            high_conf = max(expr.confidence for expr in high_expressions)
            self.assertGreater(high_conf, 0.7, "High-confidence math should score > 0.7")
        
        if low_expressions:
            low_conf = max(expr.confidence for expr in low_expressions)
            self.assertLess(low_conf, 0.6, "Low-confidence expressions should score < 0.6")
    
    def test_line_number_tracking(self):
        """Test that line numbers are correctly tracked."""
        text = """Line 1: Regular text
Line 2: The equation is E = mc²
Line 3: More text
Line 4: Another formula: F = ma"""
        
        expressions = self.extractor.extract_from_text(text)
        
        # Should find expressions on lines 2 and 4
        line_numbers = [expr.line_number for expr in expressions]
        self.assertIn(2, line_numbers, "Should find expression on line 2")
        self.assertIn(4, line_numbers, "Should find expression on line 4")
    
    def test_expression_types(self):
        """Test classification of expression types."""
        text = """
        Equation: x² + 2x + 1 = 0
        Function: f(x) = sin(x)
        Integral: ∫ x dx
        Derivative: d/dx(x²)
        """
        
        expressions = self.extractor.extract_from_text(text)
        
        # Check that different types are identified
        types = [expr.expression_type for expr in expressions]
        self.assertIn('equation', types, "Should identify equations")
        self.assertIn('function', types, "Should identify functions")


class TestDocumentExporters(unittest.TestCase):
    """Test the document export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_math = [
            {
                'original_text': 'x = (-b ± √(b² - 4ac)) / (2a)',
                'mathml': '<math><mi>x</mi><mo>=</mo><mfrac><mrow><mo>-</mo><mi>b</mi><mo>±</mo><msqrt><mrow><msup><mi>b</mi><mn>2</mn></msup><mo>-</mo><mn>4</mn><mi>a</mi><mi>c</mi></mrow></msqrt></mrow><mrow><mn>2</mn><mi>a</mi></mrow></mfrac></math>',
                'latex': 'x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}',
                'line_number': 1
            }
        ]
        
        self.sample_text = "The quadratic formula is:\n[MATH_PLACEHOLDER_0]\nThis solves quadratic equations."
    
    def test_markdown_exporter(self):
        """Test Markdown export functionality."""
        exporter = MarkdownExporter()
        
        result = exporter.export(
            text_content=self.sample_text,
            math_expressions=self.sample_math,
            metadata={'title': 'Test Document', 'author': 'Test Author'}
        )
        
        # Check that the result contains expected elements
        self.assertIn('# Test Document', result)
        self.assertIn('Test Author', result)
        self.assertIn('quadratic formula', result)
        self.assertIn('$$', result)  # LaTeX math blocks
    
    def test_html_exporter(self):
        """Test HTML export functionality."""
        exporter = HTMLExporter()
        
        result = exporter.export(
            text_content=self.sample_text,
            math_expressions=self.sample_math,
            metadata={'title': 'Test Document'}
        )
        
        # Check HTML structure
        self.assertIn('<html>', result)
        self.assertIn('<title>Test Document</title>', result)
        self.assertIn('MathJax', result)  # Should include MathJax for rendering
        self.assertIn('<math', result)   # Should contain MathML
    
    def test_pdf_exporter_availability(self):
        """Test PDF exporter availability and basic functionality."""
        exporter = PDFExporter()
        
        # Test if any PDF library is available
        has_reportlab = exporter._has_reportlab()
        has_weasyprint = exporter._has_weasyprint()
        
        # At least log what's available
        if not (has_reportlab or has_weasyprint):
            self.skipTest("No PDF export libraries available")
        
        # If available, test basic export
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.pdf"
            
            success = exporter.export_to_file(
                text_content=self.sample_text,
                math_expressions=self.sample_math,
                output_path=output_path,
                metadata={'title': 'Test PDF'}
            )
            
            if success:
                self.assertTrue(output_path.exists(), "PDF file should be created")
                self.assertGreater(output_path.stat().st_size, 0, "PDF should not be empty")


class TestDocumentProcessor(unittest.TestCase):
    """Test the MathDocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.options = ProcessingOptions(
            confidence_threshold=0.6,
            math_format='mathml',
            title='Test Document',
            author='Test Suite'
        )
        self.processor = MathDocumentProcessor(self.options)
        
        self.sample_document = """
Mathematics Test Document
========================

This document contains various mathematical expressions.

1. Basic Algebra
The quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)

2. Calculus
The derivative: f'(x) = lim(h→0) [f(x+h) - f(x)] / h
The integral: ∫ f(x) dx

3. Statistics
Normal distribution: f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
        """
    
    def test_math_preview(self):
        """Test mathematical expression preview functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.sample_document)
            temp_path = Path(f.name)
        
        try:
            expressions = self.processor.preview_math_extraction(temp_path)
            
            # Should find multiple mathematical expressions
            self.assertGreater(len(expressions), 3)
            
            # Check that we get reasonable confidence scores
            confidences = [expr.confidence for expr in expressions]
            avg_confidence = sum(confidences) / len(confidences)
            self.assertGreater(avg_confidence, 0.5, "Average confidence should be reasonable")
            
            # Check that line numbers are tracked
            line_numbers = [expr.line_number for expr in expressions]
            self.assertTrue(all(num > 0 for num in line_numbers), "All line numbers should be positive")
            
        finally:
            temp_path.unlink()
    
    def test_document_processing(self):
        """Test complete document processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input file
            input_file = temp_path / "test_document.txt"
            input_file.write_text(self.sample_document)
            
            # Process document
            result = self.processor.process_document(
                input_file,
                export_formats=['markdown', 'html'],
                output_dir=temp_path
            )
            
            # Check result structure
            self.assertIsInstance(result, ProcessingResult)
            self.assertTrue(result.success, f"Processing should succeed: {result.errors}")
            self.assertEqual(result.input_file, input_file)
            
            # Check exports were created
            self.assertIn('markdown', result.export_paths)
            self.assertIn('html', result.export_paths)
            
            for format_type, output_path in result.export_paths.items():
                self.assertTrue(output_path.exists(), f"{format_type} file should exist")
                self.assertGreater(output_path.stat().st_size, 0, f"{format_type} file should not be empty")
            
            # Check math summary
            summary = result.math_summary
            self.assertIn('total_expressions', summary)
            self.assertIn('successful_conversions', summary)
            self.assertIn('failed_conversions', summary)
            self.assertGreater(summary['total_expressions'], 0)
    
    def test_directory_processing(self):
        """Test batch processing of multiple documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Create multiple test files
            documents = {
                'math1.txt': """Basic math: a² + b² = c²""",
                'math2.txt': """Calculus: ∫ x dx = x²/2 + C""",
                'nomath.txt': """This file has no mathematical content.""",
                'math3.md': """# Markdown Math\n\nEuler's identity: e^(iπ) + 1 = 0"""
            }
            
            for filename, content in documents.items():
                (input_dir / filename).write_text(content)
            
            # Process directory with pattern filtering
            results = self.processor.process_directory(
                input_dir,
                export_formats=['markdown'],
                output_dir=output_dir,
                file_patterns=['*.txt', '*.md']
            )
            
            # Should process all files matching patterns
            self.assertEqual(len(results), 4)
            
            # Check individual results
            successful_results = [r for r in results if r.success]
            self.assertGreater(len(successful_results), 0, "Some files should process successfully")
            
            # Check processing summary
            summary = self.processor.get_processing_summary(results)
            self.assertIn('files_processed', summary)
            self.assertIn('files_successful', summary)
            self.assertIn('total_math_expressions', summary)
            self.assertEqual(summary['files_processed'], 4)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete document processing pipeline."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end document processing workflow."""
        # Create a realistic mathematical document
        document_content = """
Advanced Calculus Study Guide
============================

Chapter 1: Limits and Continuity
--------------------------------

The formal definition of a limit:
lim(x→a) f(x) = L

ε-δ definition: For every ε > 0, there exists δ > 0 such that
if 0 < |x - a| < δ, then |f(x) - L| < ε

Chapter 2: Derivatives
---------------------

The derivative as a limit:
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

Chain rule: (f∘g)'(x) = f'(g(x)) · g'(x)

Product rule: (fg)' = f'g + fg'

Chapter 3: Integration
---------------------

Fundamental theorem of calculus:
∫ₐᵇ f'(x) dx = f(b) - f(a)

Integration by parts:
∫ u dv = uv - ∫ v du

Substitution method:
∫ f(g(x))g'(x) dx = ∫ f(u) du, where u = g(x)
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input file
            input_file = temp_path / "calculus_guide.txt"
            input_file.write_text(document_content)
            
            # Set up processing options
            options = ProcessingOptions(
                confidence_threshold=0.6,
                math_format='mathml',
                title='Advanced Calculus Study Guide',
                author='Integration Test Suite'
            )
            
            # Create processor
            processor = MathDocumentProcessor(options=options)
            
            # Step 1: Preview mathematical expressions
            expressions = processor.preview_math_extraction(input_file)
            self.assertGreater(len(expressions), 5, "Should find multiple mathematical expressions")
            
            # Step 2: Process and export to all formats
            export_formats = ['markdown', 'html']
            # Add PDF if available
            pdf_exporter = PDFExporter()
            if pdf_exporter._has_reportlab() or pdf_exporter._has_weasyprint():
                export_formats.append('pdf')
            
            result = processor.process_document(
                input_file,
                export_formats,
                temp_path
            )
            
            # Verify successful processing
            self.assertTrue(result.success, f"End-to-end processing failed: {result.errors}")
            
            # Verify all requested formats were generated
            for format_type in export_formats:
                self.assertIn(format_type, result.export_paths)
                output_file = result.export_paths[format_type]
                self.assertTrue(output_file.exists())
                self.assertGreater(output_file.stat().st_size, 100)  # Reasonable file size
            
            # Verify math conversion statistics
            summary = result.math_summary
            self.assertGreater(summary['total_expressions'], 0)
            
            # At least some expressions should convert successfully
            if summary['total_expressions'] > 0:
                success_rate = summary['successful_conversions'] / summary['total_expressions']
                self.assertGreater(success_rate, 0.5, "At least 50% of expressions should convert")
    
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        options = ProcessingOptions()
        processor = MathDocumentProcessor(options)
        
        # Test with non-existent file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            non_existent = temp_path / "does_not_exist.txt"
            
            expressions = processor.preview_math_extraction(non_existent)
            self.assertEqual(len(expressions), 0, "Should handle non-existent files gracefully")
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            empty_file = Path(f.name)
        
        try:
            result = processor.process_document(
                empty_file,
                ['markdown'],
                empty_file.parent
            )
            
            # Should not crash, might succeed with empty content
            self.assertIsInstance(result, ProcessingResult)
            
        finally:
            empty_file.unlink()


def run_tests():
    """Run all document processing tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMathExtractor,
        TestDocumentExporters,
        TestDocumentProcessor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)