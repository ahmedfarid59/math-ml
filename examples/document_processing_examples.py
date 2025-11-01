#!/usr/bin/env python3
"""
Examples demonstrating document processing capabilities of the MathML parser.

This module shows how to extract mathematical expressions from text documents
and export them to various formats (PDF, Markdown, HTML) with proper MathML
or LaTeX formatting.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from mathml_parser import MathMLParser
from mathml_parser.core.document_processor import MathDocumentProcessor, ProcessingOptions
from mathml_parser.core.text_processor import MathExtractor
from mathml_parser.core.document_exporters import MarkdownExporter, HTMLExporter, PDFExporter


def create_sample_documents() -> Dict[str, str]:
    """Create sample documents with mathematical content for testing."""
    
    documents = {
        "physics_notes.txt": """
Physics Study Notes
==================

1. Kinematics Equations

The basic kinematic equations for constant acceleration are:

v = v₀ + at
x = x₀ + v₀t + ½at²
v² = v₀² + 2a(x - x₀)

where v is velocity, a is acceleration, t is time, and x is position.

2. Energy Conservation

The conservation of energy states that:
E_total = KE + PE = ½mv² + mgh = constant

3. Wave Equation

The general wave equation is:
∂²y/∂t² = (v²)(∂²y/∂x²)

For a sinusoidal wave: y(x,t) = A sin(kx - ωt + φ)

4. Maxwell's Equations

∇·E = ρ/ε₀
∇·B = 0
∇×E = -∂B/∂t
∇×B = μ₀J + μ₀ε₀(∂E/∂t)

5. Schrödinger Equation

iℏ(∂ψ/∂t) = Ĥψ

For a particle in a box: ψₙ(x) = √(2/L) sin(nπx/L)
        """,
        
        "calculus_homework.txt": """
Calculus Problem Set #5
======================

Problem 1: Find the derivative of f(x) = x³ + 2x² - 5x + 1

Solution: f'(x) = 3x² + 4x - 5

Problem 2: Evaluate the integral ∫₀² (x² + 1) dx

Solution: 
∫(x² + 1) dx = x³/3 + x + C
∫₀² (x² + 1) dx = [x³/3 + x]₀² = (8/3 + 2) - (0) = 14/3

Problem 3: Find the limit lim(x→0) (sin(x)/x)

Using L'Hôpital's rule:
lim(x→0) (sin(x)/x) = lim(x→0) (cos(x)/1) = 1

Problem 4: Taylor series expansion

The Taylor series for eˣ around x = 0 is:
eˣ = 1 + x + x²/2! + x³/3! + ... = Σ(n=0 to ∞) xⁿ/n!

Problem 5: Vector calculus

For a vector field F = ⟨P, Q, R⟩:
div(F) = ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z
curl(F) = ∇×F = ⟨∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y⟩
        """,
        
        "statistics_summary.txt": """
Statistics Quick Reference
=========================

1. Descriptive Statistics

Mean: μ = (Σx)/n
Variance: σ² = Σ(x - μ)²/n
Standard deviation: σ = √(σ²)

Sample statistics:
Sample mean: x̄ = (Σx)/n
Sample variance: s² = Σ(x - x̄)²/(n-1)

2. Probability Distributions

Normal distribution: f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))

Binomial: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)

Poisson: P(X = k) = (λ^k * e^(-λ))/k!

3. Hypothesis Testing

Test statistic for mean: t = (x̄ - μ₀)/(s/√n)

Chi-square: χ² = Σ((O_i - E_i)²/E_i)

Confidence interval: x̄ ± t_(α/2) * (s/√n)

4. Regression Analysis

Linear regression: y = β₀ + β₁x + ε

Coefficient of determination: R² = 1 - (SS_res/SS_tot)

Multiple regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
        """
    }
    
    return documents


def example_1_basic_text_processing():
    """Example 1: Basic text processing and math extraction."""
    print("=" * 60)
    print("Example 1: Basic Text Processing and Math Extraction")
    print("=" * 60)
    
    # Create a sample document
    sample_text = """
    The quadratic formula is: x = (-b ± √(b² - 4ac)) / (2a)
    
    For the equation x² + 5x + 6 = 0, we have:
    a = 1, b = 5, c = 6
    
    The discriminant is: Δ = b² - 4ac = 25 - 24 = 1
    
    So the solutions are: x = (-5 ± 1) / 2 = -2 or -3
    """
    
    # Extract mathematical expressions
    extractor = MathExtractor()
    math_expressions = extractor.extract_from_text(sample_text)
    
    print(f"Found {len(math_expressions)} mathematical expressions:")
    print()
    
    for i, expr in enumerate(math_expressions, 1):
        print(f"{i}. Line {expr.line_number}: '{expr.original_text}'")
        print(f"   Confidence: {expr.confidence:.2f}")
        print(f"   Type: {expr.expression_type}")
        print()


def example_2_document_processing():
    """Example 2: Complete document processing workflow."""
    print("=" * 60)
    print("Example 2: Complete Document Processing Workflow")
    print("=" * 60)
    
    # Create sample documents
    documents = create_sample_documents()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write sample documents to temporary files
        input_files = []
        for filename, content in documents.items():
            file_path = temp_path / filename
            file_path.write_text(content, encoding='utf-8')
            input_files.append(file_path)
        
        # Set up processing options
        options = ProcessingOptions(
            confidence_threshold=0.6,
            math_format='mathml',
            title="Mathematical Document Collection",
            author="MathML Parser Examples"
        )
        
        # Create document processor
        processor = MathDocumentProcessor(options=options)
        
        # Process each document
        for input_file in input_files:
            print(f"\nProcessing: {input_file.name}")
            print("-" * 40)
            
            # Preview math extraction
            expressions = processor.preview_math_extraction(input_file)
            print(f"Found {len(expressions)} mathematical expressions")
            
            # Process and export to multiple formats
            export_formats = ['markdown', 'html', 'pdf']
            result = processor.process_document(
                input_file,
                export_formats,
                temp_path / "output"
            )
            
            if result.success:
                print("✓ Successfully processed!")
                print("Generated files:")
                for format_type, output_path in result.export_paths.items():
                    print(f"  - {format_type.upper()}: {output_path.name}")
                
                # Show math summary
                summary = result.math_summary
                print(f"Math conversion summary:")
                print(f"  - Total expressions: {summary['total_expressions']}")
                print(f"  - Successful: {summary['successful_conversions']}")
                print(f"  - Failed: {summary['failed_conversions']}")
            else:
                print("✗ Processing failed!")
                for error in result.errors:
                    print(f"  Error: {error}")


def example_3_custom_exporters():
    """Example 3: Using individual exporters for custom workflows."""
    print("=" * 60)
    print("Example 3: Custom Export Workflows")
    print("=" * 60)
    
    # Sample mathematical content
    math_content = [
        {
            'original_text': 'f(x) = x² + 2x + 1',
            'mathml': '<math><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn></mrow></math>',
            'latex': 'f(x) = x^2 + 2x + 1',
            'line_number': 1
        },
        {
            'original_text': '∫₀¹ x² dx = 1/3',
            'mathml': '<math><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msup><mi>x</mi><mn>2</mn></msup><mi>d</mi><mi>x</mi><mo>=</mo><mfrac><mn>1</mn><mn>3</mn></mfrac></mrow></math>',
            'latex': '\\int_0^1 x^2 dx = \\frac{1}{3}',
            'line_number': 3
        }
    ]
    
    # Text content
    text_content = """Introduction to Calculus

This document covers basic calculus concepts.

Function Definition:
[MATH_PLACEHOLDER_0]

Integration Example:
[MATH_PLACEHOLDER_1]

These examples demonstrate fundamental calculus operations."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Example: Custom Markdown export
        print("Creating custom Markdown export...")
        markdown_exporter = MarkdownExporter()
        
        markdown_content = markdown_exporter.export(
            text_content=text_content,
            math_expressions=math_content,
            metadata={
                'title': 'Custom Calculus Guide',
                'author': 'Math Expert',
                'date': '2024'
            }
        )
        
        markdown_file = temp_path / "custom_guide.md"
        markdown_file.write_text(markdown_content, encoding='utf-8')
        print(f"✓ Markdown file created: {markdown_file.name}")
        
        # Example: Custom HTML export with styling
        print("\nCreating custom HTML export...")
        html_exporter = HTMLExporter()
        
        html_content = html_exporter.export(
            text_content=text_content,
            math_expressions=math_content,
            metadata={
                'title': 'Interactive Calculus Guide',
                'author': 'Math Expert'
            }
        )
        
        html_file = temp_path / "interactive_guide.html"
        html_file.write_text(html_content, encoding='utf-8')
        print(f"✓ HTML file created: {html_file.name}")
        
        # Example: PDF export (if dependencies available)
        print("\nAttempting PDF export...")
        try:
            pdf_exporter = PDFExporter()
            
            pdf_path = temp_path / "calculus_guide.pdf"
            success = pdf_exporter.export_to_file(
                text_content=text_content,
                math_expressions=math_content,
                output_path=pdf_path,
                metadata={
                    'title': 'Calculus Reference Guide',
                    'author': 'MathML Parser'
                }
            )
            
            if success:
                print(f"✓ PDF file created: {pdf_path.name}")
            else:
                print("⚠ PDF export failed (missing dependencies?)")
                
        except Exception as e:
            print(f"⚠ PDF export not available: {e}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple documents."""
    print("=" * 60)
    print("Example 4: Batch Processing Multiple Documents")
    print("=" * 60)
    
    # Create multiple sample documents
    documents = create_sample_documents()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Write sample documents
        for filename, content in documents.items():
            file_path = input_dir / filename
            file_path.write_text(content, encoding='utf-8')
        
        # Set up processing options
        options = ProcessingOptions(
            confidence_threshold=0.7,
            math_format='mathml',
            title="Mathematics Document Collection",
            author="Batch Processor"
        )
        
        # Create processor
        processor = MathDocumentProcessor(options=options)
        
        # Process entire directory
        print(f"Processing all .txt files in {input_dir}")
        print("-" * 40)
        
        export_formats = ['markdown', 'html']
        results = processor.process_directory(
            input_dir,
            export_formats,
            output_dir,
            file_patterns=['*.txt']
        )
        
        # Show results
        print(f"\nProcessed {len(results)} files:")
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.input_file.name}")
            
            if result.success:
                summary = result.math_summary
                print(f"   Math expressions: {summary['total_expressions']}")
                print(f"   Exports: {', '.join(result.export_paths.keys())}")
        
        # Show processing summary
        summary = processor.get_processing_summary(results)
        print(f"\nBatch Processing Summary:")
        print(f"  Files processed: {summary['files_processed']}")
        print(f"  Success rate: {summary['files_successful']}/{summary['files_processed']}")
        print(f"  Total math expressions: {summary['total_math_expressions']}")
        print(f"  Math conversion rate: {summary['conversion_success_rate']:.1f}%")


def example_5_confidence_tuning():
    """Example 5: Tuning confidence thresholds for math detection."""
    print("=" * 60)
    print("Example 5: Confidence Threshold Tuning")
    print("=" * 60)
    
    # Text with mixed mathematical and non-mathematical content
    mixed_text = """
Research Paper on Mathematical Modeling
======================================

The study examines various equations and formulas.

Clear mathematical expressions:
- The area of a circle: A = πr²
- Quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)
- Euler's identity: e^(iπ) + 1 = 0

Ambiguous cases:
- The temperature was 25°C (not really math)
- Version 2.1.3 was released (version number)
- He scored 90% on the test (percentage, but contextually not mathematical)
- The ratio was 3:1 (could be mathematical)

Clear non-mathematical:
- The meeting is at 3:30 PM
- Call me at 555-1234
- The price is $29.99
    """
    
    # Test different confidence thresholds
    extractor = MathExtractor()
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    print("Testing different confidence thresholds:")
    print()
    
    for threshold in thresholds:
        expressions = extractor.extract_from_text(mixed_text, min_confidence=threshold)
        
        print(f"Threshold {threshold}: Found {len(expressions)} expressions")
        for expr in expressions:
            print(f"  - '{expr.original_text}' (confidence: {expr.confidence:.2f})")
        print()
    
    print("Recommendation: Use threshold 0.6-0.7 for balanced precision/recall")


def run_all_examples():
    """Run all document processing examples."""
    print("MathML Parser - Document Processing Examples")
    print("=" * 80)
    print()
    
    examples = [
        example_1_basic_text_processing,
        example_2_document_processing,
        example_3_custom_exporters,
        example_4_batch_processing,
        example_5_confidence_tuning
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_all_examples()