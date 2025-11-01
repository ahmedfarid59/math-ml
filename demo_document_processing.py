#!/usr/bin/env python3
"""
Simple demonstration of the MathML parser's document processing capabilities.

This script shows how to extract mathematical expressions from text files
and export them to different formats.
"""

import tempfile
from pathlib import Path

# Sample mathematical text
SAMPLE_TEXT = """
Mathematical Analysis Notes
==========================

1. Calculus Fundamentals

The derivative of f(x) = x² is f'(x) = 2x

The integral ∫ x² dx = x³/3 + C

2. Linear Algebra

Matrix multiplication: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ

Eigenvalue equation: Av = λv

3. Statistics

Normal distribution: f(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))

Standard error: SE = σ/√n

4. Physics

Einstein's equation: E = mc²

Wave equation: ∂²y/∂t² = c²(∂²y/∂x²)
"""

def demo_basic_extraction():
    """Demonstrate basic mathematical expression extraction."""
    print("=" * 50)
    print("Demo: Mathematical Expression Extraction")
    print("=" * 50)
    
    from mathml_parser.core.text_processor import MathExtractor
    
    # Extract mathematical expressions
    extractor = MathExtractor()
    expressions = extractor.extract_from_text(SAMPLE_TEXT)
    
    print(f"Found {len(expressions)} mathematical expressions:\n")
    
    for i, expr in enumerate(expressions, 1):
        print(f"{i}. '{expr.original_text}'")
        print(f"   Line: {expr.line_number}, Confidence: {expr.confidence:.2f}")
        print(f"   Type: {expr.expression_type}")
        print()

def demo_document_export():
    """Demonstrate document processing and export."""
    print("=" * 50)
    print("Demo: Document Processing and Export")
    print("=" * 50)
    
    from mathml_parser.core.document_processor import MathDocumentProcessor, ProcessingOptions
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input file
        input_file = temp_path / "math_notes.txt"
        input_file.write_text(SAMPLE_TEXT)
        
        # Set up processing options
        options = ProcessingOptions(
            confidence_threshold=0.6,
            math_format='mathml',
            title="Mathematics Demo",
            author="MathML Parser"
        )
        
        # Create processor and process document
        processor = MathDocumentProcessor(options=options)
        
        # Export to multiple formats
        result = processor.process_document(
            input_file,
            export_formats=['markdown', 'html'],
            output_dir=temp_path
        )
        
        if result.success:
            print("✓ Document processing successful!")
            print("\nGenerated files:")
            for format_type, output_path in result.export_paths.items():
                print(f"  - {format_type.upper()}: {output_path}")
                
                # Show a preview of the content
                if format_type == 'markdown':
                    content = output_path.read_text()[:500]
                    print(f"\nMarkdown preview:\n{'-' * 30}")
                    print(content + "..." if len(content) == 500 else content)
            
            # Show math summary
            summary = result.math_summary
            print(f"\nMath Summary:")
            print(f"  Total expressions: {summary['total_expressions']}")
            print(f"  Successful conversions: {summary['successful_conversions']}")
            print(f"  Failed conversions: {summary['failed_conversions']}")
        else:
            print("✗ Document processing failed!")
            for error in result.errors:
                print(f"  Error: {error}")

def demo_cli_usage():
    """Show CLI usage examples."""
    print("=" * 50)
    print("Demo: CLI Usage Examples")
    print("=" * 50)
    
    print("Command-line usage examples:")
    print()
    
    print("1. Extract and preview math from a text file:")
    print("   python -m mathml_parser --preview-math document.txt")
    print()
    
    print("2. Process a document and export to multiple formats:")
    print("   python -m mathml_parser --process-doc notes.txt --export markdown,html,pdf")
    print()
    
    print("3. Process all text files in a directory:")
    print("   python -m mathml_parser --process-dir ./documents --file-patterns '*.txt,*.md'")
    print()
    
    print("4. Set confidence threshold and output format:")
    print("   python -m mathml_parser --process-doc math.txt --confidence-threshold 0.8 --export html")
    print()
    
    print("5. Add document metadata:")
    print("   python -m mathml_parser --process-doc report.txt --doc-title 'Math Report' --doc-author 'John Doe'")

def main():
    """Run the complete demonstration."""
    print("MathML Parser - Document Processing Demo")
    print("=" * 60)
    print()
    
    # Run individual demos
    demo_basic_extraction()
    print("\n")
    
    demo_document_export()
    print("\n")
    
    demo_cli_usage()
    print("\n")
    
    print("=" * 60)
    print("Demo completed! Try the CLI commands with your own files.")

if __name__ == "__main__":
    main()