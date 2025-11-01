#!/usr/bin/env python3
"""
Enhanced Command Line Interface for MathML Parser
================================================

Provides a comprehensive command-line tool for parsing mathematical expressions
with support for LaTeX input, optimization, multiple output formats, and advanced features.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from mathml_parser import (
    MathMLParser, MathParseError, parse, parse_safe, parse_latex, 
    parse_and_optimize, parse_to_format, get_all_formats,
    optimize_expression, latex_to_standard,
    # Document processing
    process_document, process_documents, MathDocumentProcessor,
    ProcessingOptions, ProcessingResult
)
from mathml_parser.core.latex_parser import LaTeXParser
from mathml_parser.core.optimizer import ExpressionOptimizer
from mathml_parser.core.multi_format import MultiFormatRenderer


def create_parser() -> argparse.ArgumentParser:
    """Create the enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mathml-parse",
        description="Enhanced mathematical expression parser with advanced features",
        epilog="""
Examples:
  # Basic parsing
  mathml-parse "x^2 + 2*x + 1"
  
  # LaTeX input
  mathml-parse --latex "\\frac{x^2}{2}" 
  
  # Multiple formats
  mathml-parse "sin(π/2)" --format latex,html,ascii
  
  # With optimization
  mathml-parse "x + 0" --optimize
  
  # File processing
  mathml-parse --file input.txt --output output.xml --format mathml
  
  # Advanced features
  mathml-parse "∫₀^π sin(x) dx" --format all --optimize --verbose
  
    # Interactive mode with all features
    mathml-parse --interactive --optimize --format all
    
    # Document processing (NEW!)
    mathml-parse --process-doc input.txt --export pdf,html,markdown
    mathml-parse --process-dir /path/to/docs --export html --output-dir /path/to/output
    mathml-parse --preview-math document.txt  # Preview math extraction
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "expression",
        nargs="?",
        help="Mathematical expression to parse"
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="File containing expressions to parse (one per line)"
    )
    input_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode for entering multiple expressions"
    )
    
    # Document processing options (NEW!)
    input_group.add_argument(
        "--process-doc",
        type=Path,
        help="Process a text document to extract and convert math content"
    )
    input_group.add_argument(
        "--process-dir", 
        type=Path,
        help="Process all text files in a directory"
    )
    input_group.add_argument(
        "--preview-math",
        type=Path,
        help="Preview mathematical expressions that would be extracted from a file"
    )
    
    # Input format options
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Input is in LaTeX format"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--format",
        default="mathml",
        help="Output format(s): mathml, latex, html, svg, ascii, plain, all (comma-separated for multiple)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print output"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )
    
    # Document processing options
    parser.add_argument(
        "--export",
        type=str,
        default="html",
        help="Export formats for document processing: pdf,html,markdown (comma-separated)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for math expression detection (0.0-1.0)"
    )
    parser.add_argument(
        "--file-patterns",
        type=str,
        help="File patterns for directory processing (comma-separated, e.g., '*.txt,*.md')"
    )
    parser.add_argument(
        "--doc-title",
        type=str,
        help="Title for exported documents"
    )
    parser.add_argument(
        "--doc-author", 
        type=str,
        help="Author for exported documents"
    )
    
    # Processing options
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply expression optimization"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable input validation"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict parsing mode"
    )
    
    # Information options
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Include performance metrics in output"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Show detected mathematical features"
    )
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Show optimization suggestions (requires --optimize)"
    )
    
    # Utility options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (only output results)"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.0.0"
    )
    
    return parser


def setup_components(args: argparse.Namespace):
    """Set up parser components based on arguments."""
    # Main parser
    parser_instance = MathMLParser(
        enable_validation=not args.no_validate,
        enable_metrics=args.metrics or args.features,
        strict_mode=args.strict,
        cache_grammar=True
    )
    
    # LaTeX parser (if needed)
    latex_parser = LaTeXParser() if args.latex else None
    
    # Optimizer (if needed)
    optimizer = ExpressionOptimizer() if args.optimize else None
    
    # Multi-format renderer
    renderer = MultiFormatRenderer()
    
    return parser_instance, latex_parser, optimizer, renderer


def parse_formats(format_string: str) -> List[str]:
    """Parse format string into list of formats."""
    if format_string.lower() == 'all':
        return ['mathml', 'latex', 'html', 'svg', 'ascii', 'plain']
    return [f.strip().lower() for f in format_string.split(',')]


def parse_expression(expression: str, components, args: argparse.Namespace) -> Dict[str, Any]:
    """Parse a single expression with all requested features."""
    parser_instance, latex_parser, optimizer, renderer = components
    
    result = {
        "expression": expression,
        "original": expression,
        "success": False,
        "outputs": {},
        "metrics": None,
        "features": None,
        "optimization": None,
        "error": None,
        "parse_time": 0
    }
    
    start_time = time.time()
    
    try:
        # Handle LaTeX input
        if args.latex and latex_parser:
            try:
                converted = latex_parser.convert_to_standard(expression)
                if args.verbose and not args.quiet:
                    print(f"LaTeX converted: {expression} → {converted}", file=sys.stderr)
                expression = converted
            except Exception as e:
                result["error"] = f"LaTeX conversion error: {str(e)}"
                result["parse_time"] = time.time() - start_time
                return result
        
        # Optimization
        optimized_expr = expression
        if args.optimize and optimizer:
            try:
                optimized_expr = optimizer.optimize_expression(expression)
                optimization_data = {
                    "optimized": optimized_expr,
                    "changed": optimized_expr != expression
                }
                
                if args.suggestions:
                    optimization_data["suggestions"] = optimizer.suggest_optimizations(expression)
                
                result["optimization"] = optimization_data
                
                if args.verbose and not args.quiet and optimized_expr != expression:
                    print(f"Optimized: {expression} → {optimized_expr}", file=sys.stderr)
                    
            except Exception as e:
                if args.verbose and not args.quiet:
                    print(f"Optimization warning: {str(e)}", file=sys.stderr)
        
        # Parse with safe method
        parse_result = parser_instance.parse_safe(optimized_expr)
        
        if not parse_result.success:
            result["error"] = {
                "message": str(parse_result.error),
                "type": getattr(parse_result.error, 'error_type', 'unknown'),
                "position": getattr(parse_result.error, 'position', None),
                "suggestions": getattr(parse_result.error, 'suggestions', [])
            }
            result["parse_time"] = time.time() - start_time
            return result
        
        result["success"] = True
        
        # Collect metrics and features
        if args.metrics and hasattr(parse_result, 'metrics') and parse_result.metrics:
            metrics = parse_result.metrics
            result["metrics"] = {
                "parse_time": getattr(metrics, 'parse_time', None),
                "validation_time": getattr(metrics, 'validation_time', None),
                "transformation_time": getattr(metrics, 'transformation_time', None),
                "input_length": getattr(metrics, 'input_length', len(optimized_expr)),
                "output_length": getattr(metrics, 'output_length', None),
                "complexity_score": getattr(metrics, 'complexity_score', 0),
                "features_used": getattr(metrics, 'features_used', [])
            }
            
            if args.features:
                result["features"] = metrics.features_used
        
        # Generate requested output formats
        formats = parse_formats(args.format)
        
        for format_name in formats:
            try:
                if format_name == 'mathml':
                    result["outputs"][format_name] = parse_result.mathml
                else:
                    # Use multi-format renderer
                    formatted = renderer.render(optimized_expr, format_name)
                    result["outputs"][format_name] = formatted
            except Exception as e:
                result["outputs"][format_name] = f"Error: {str(e)}"
        
        result["parse_time"] = time.time() - start_time
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        result["parse_time"] = time.time() - start_time
        return result


def format_output(results: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    """Format results according to the specified output format."""
    if args.json_output:
        if len(results) == 1:
            data = results[0]
        else:
            data = {"results": results}
        
        if args.pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, ensure_ascii=False)
    
    # Text-based output
    output_lines = []
    
    for i, result in enumerate(results):
        if len(results) > 1 and not args.quiet:
            output_lines.append(f"=== Expression {i+1}: {result['original']} ===")
        
        if not result["success"]:
            output_lines.append(f"ERROR: {result['error']}")
            continue
        
        # Show optimization info
        if result.get("optimization") and args.verbose and not args.quiet:
            opt = result["optimization"]
            if opt["changed"]:
                output_lines.append(f"Optimized: {opt['optimized']}")
            if "suggestions" in opt and opt["suggestions"]:
                output_lines.append(f"Suggestions: {'; '.join(opt['suggestions'])}")
        
        # Show metrics
        if result.get("metrics") and (args.metrics or args.verbose) and not args.quiet:
            metrics = result["metrics"]
            output_lines.append(f"Parse time: {result['parse_time']:.4f}s")
            if metrics.get("complexity_score") is not None:
                output_lines.append(f"Complexity: {metrics['complexity_score']:.2f}")
            if args.features and result.get("features"):
                output_lines.append(f"Features: {', '.join(result['features'])}")
        
        # Show outputs
        for format_name, content in result["outputs"].items():
            if len(result["outputs"]) > 1 and not args.quiet:
                output_lines.append(f"\n{format_name.upper()}:")
            output_lines.append(content)
            if len(result["outputs"]) > 1 and not args.quiet:
                output_lines.append("")
        
        if len(results) > 1 and not args.quiet:
            output_lines.append("")
    
    return '\n'.join(output_lines)


def interactive_mode(components, args: argparse.Namespace) -> None:
    """Run in enhanced interactive mode."""
    print("Enhanced MathML Parser Interactive Mode")
    print("Features enabled:", end="")
    features = []
    if args.latex: features.append("LaTeX input")
    if args.optimize: features.append("optimization")
    if args.metrics: features.append("metrics")
    formats = parse_formats(args.format)
    if len(formats) > 1: features.append(f"multiple formats ({', '.join(formats)})")
    
    if features:
        print(" " + ", ".join(features))
    else:
        print(" basic parsing")
    
    print("Enter mathematical expressions (type 'quit' or 'exit' to stop):")
    print("Special commands: :help, :formats, :optimize on/off, :latex on/off")
    print("-" * 60)
    
    while True:
        try:
            expression = input(">>> ").strip()
            
            if expression.lower() in ["quit", "exit", "q"]:
                break
            elif expression == ":help":
                print("Available commands:")
                print("  :help - Show this help")
                print("  :formats - Show available output formats")
                print("  :optimize on/off - Toggle optimization")
                print("  :latex on/off - Toggle LaTeX input mode")
                print("  quit/exit - Exit interactive mode")
                continue
            elif expression == ":formats":
                print("Available output formats:")
                print("  mathml, latex, html, svg, ascii, plain, all")
                continue
            elif expression.startswith(":optimize"):
                parts = expression.split()
                if len(parts) > 1:
                    args.optimize = parts[1].lower() == "on"
                    # Update components
                    components = setup_components(args)
                print(f"Optimization: {'enabled' if args.optimize else 'disabled'}")
                continue
            elif expression.startswith(":latex"):
                parts = expression.split()
                if len(parts) > 1:
                    args.latex = parts[1].lower() == "on"
                    # Update components
                    components = setup_components(args)
                print(f"LaTeX input mode: {'enabled' if args.latex else 'disabled'}")
                continue
            elif not expression:
                continue
            
            result = parse_expression(expression, components, args)
            output = format_output([result], args)
            print(output)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main() -> int:
    """Enhanced CLI entry point."""
    arg_parser = create_parser()
    args = arg_parser.parse_args()
    
    # Set up logging level
    if args.verbose and not args.quiet:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Handle document processing modes
        if args.process_doc or args.process_dir or args.preview_math:
            return handle_document_processing(args)
        
        # Handle expression parsing modes (existing functionality)
        # Create components
        components = setup_components(args)
        results = []
        
        if args.interactive:
            interactive_mode(components, args)
            return 0
        
        elif args.file:
            # Parse expressions from file
            if not args.file.exists():
                print(f"Error: File '{args.file}' not found", file=sys.stderr)
                return 1
            
            with open(args.file, 'r', encoding=args.encoding) as f:
                expressions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            
            if not expressions:
                print(f"Error: No expressions found in '{args.file}'", file=sys.stderr)
                return 1
            
            for expression in expressions:
                if not args.quiet:
                    print(f"Processing: {expression}", file=sys.stderr)
                result = parse_expression(expression, components, args)
                results.append(result)
        
        else:
            # Parse single expression
            if not args.quiet:
                print(f"Processing: {args.expression}", file=sys.stderr)
            result = parse_expression(args.expression, components, args)
            results.append(result)
        
        # Format and output results
        output = format_output(results, args)
        
        if args.output:
            with open(args.output, 'w', encoding=args.encoding) as f:
                f.write(output)
            if not args.quiet:
                print(f"Results written to '{args.output}'", file=sys.stderr)
        else:
            print(output)
        
        # Check for errors
        error_count = sum(1 for r in results if not r["success"])
        if error_count > 0:
            if not args.quiet:
                print(f"Warning: {error_count} expression(s) failed to parse", file=sys.stderr)
            return 1 if error_count == len(results) else 0
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_document_processing(args) -> int:
    """Handle document processing modes."""
    
    # Create processing options
    processing_options = ProcessingOptions(
        confidence_threshold=args.confidence_threshold,
        encoding=args.encoding,
        math_format=args.format if not args.json_output else 'mathml',
        title=args.doc_title,
        author=args.doc_author
    )
    
    # Preview mode
    if args.preview_math:
        if not args.preview_math.exists():
            print(f"Error: File '{args.preview_math}' not found", file=sys.stderr)
            return 1
        
        processor = MathDocumentProcessor(options=processing_options)
        
        try:
            expressions = processor.preview_math_extraction(args.preview_math)
            
            if args.json_output:
                import json
                preview_data = {
                    'file': str(args.preview_math),
                    'total_expressions': len(expressions),
                    'expressions': [
                        {
                            'text': expr.original_text,
                            'line': expr.line_number,
                            'confidence': expr.confidence,
                            'context_before': expr.context_before,
                            'context_after': expr.context_after
                        }
                        for expr in expressions
                    ]
                }
                print(json.dumps(preview_data, indent=2))
            else:
                print(f"Mathematical expressions found in '{args.preview_math}':")
                print("=" * 60)
                for i, expr in enumerate(expressions, 1):
                    print(f"{i}. Line {expr.line_number}: '{expr.original_text}' (confidence: {expr.confidence:.2f})")
                    if args.verbose:
                        print(f"   Context: ...{expr.context_before}[{expr.original_text}]{expr.context_after}...")
                print(f"\nTotal: {len(expressions)} expressions found")
            
            return 0
            
        except Exception as e:
            print(f"Error previewing math expressions: {e}", file=sys.stderr)
            return 1
    
    # Document processing mode
    export_formats = [fmt.strip() for fmt in args.export.split(',')]
    
    if args.process_doc:
        # Process single document
        if not args.process_doc.exists():
            print(f"Error: File '{args.process_doc}' not found", file=sys.stderr)
            return 1
        
        try:
            result = process_document(
                args.process_doc,
                export_formats,
                args.output.parent if args.output else None,
                **processing_options.__dict__
            )
            
            return handle_processing_result(result, args)
            
        except Exception as e:
            print(f"Error processing document: {e}", file=sys.stderr)
            return 1
    
    elif args.process_dir:
        # Process directory
        if not args.process_dir.exists() or not args.process_dir.is_dir():
            print(f"Error: Directory '{args.process_dir}' not found", file=sys.stderr)
            return 1
        
        # Parse file patterns
        file_patterns = None
        if args.file_patterns:
            file_patterns = [pattern.strip() for pattern in args.file_patterns.split(',')]
        
        try:
            processor = MathDocumentProcessor(options=processing_options)
            results = processor.process_directory(
                args.process_dir,
                export_formats, 
                args.output.parent if args.output else None,
                file_patterns
            )
            
            return handle_multiple_processing_results(results, args, processor)
            
        except Exception as e:
            print(f"Error processing directory: {e}", file=sys.stderr)
            return 1
    
    return 0


def handle_processing_result(result: ProcessingResult, args) -> int:
    """Handle the result of processing a single document."""
    if args.json_output:
        import json
        result_data = {
            'input_file': str(result.input_file),
            'success': result.success,
            'math_summary': result.math_summary,
            'export_paths': {k: str(v) for k, v in result.export_paths.items()},
            'errors': result.errors
        }
        print(json.dumps(result_data, indent=2))
    else:
        print(f"Processed: {result.input_file}")
        if result.success:
            print(f"✓ Success! Generated {len(result.export_paths)} output file(s):")
            for format_type, path in result.export_paths.items():
                print(f"  - {format_type.upper()}: {path}")
            
            # Show math summary
            summary = result.math_summary
            print(f"\nMath content summary:")
            print(f"  - Total expressions: {summary.get('total_expressions', 0)}")
            print(f"  - Successful conversions: {summary.get('successful_conversions', 0)}")
            print(f"  - Failed conversions: {summary.get('failed_conversions', 0)}")
            if summary.get('total_expressions', 0) > 0:
                success_rate = (summary.get('successful_conversions', 0) / 
                              summary.get('total_expressions', 1) * 100)
                print(f"  - Success rate: {success_rate:.1f}%")
        else:
            print(f"✗ Failed to process document")
            for error in result.errors:
                print(f"  Error: {error}")
    
    return 0 if result.success else 1


def handle_multiple_processing_results(results: List[ProcessingResult], args, processor: MathDocumentProcessor) -> int:
    """Handle the results of processing multiple documents."""
    if args.json_output:
        import json
        results_data = {
            'results': [
                {
                    'input_file': str(result.input_file),
                    'success': result.success,
                    'math_summary': result.math_summary,
                    'export_paths': {k: str(v) for k, v in result.export_paths.items()},
                    'errors': result.errors
                }
                for result in results
            ],
            'summary': processor.get_processing_summary(results)
        }
        print(json.dumps(results_data, indent=2))
    else:
        # Show individual results
        for result in results:
            print(f"{'✓' if result.success else '✗'} {result.input_file}")
            if not result.success:
                for error in result.errors:
                    print(f"    Error: {error}")
        
        # Show overall summary
        summary = processor.get_processing_summary(results)
        print(f"\nProcessing Summary:")
        print(f"  Files processed: {summary['files_processed']}")
        print(f"  Successful: {summary['files_successful']}")
        print(f"  Failed: {summary['files_failed']}")
        print(f"  Math expressions found: {summary['total_math_expressions']}")
        print(f"  Math conversion success rate: {summary['conversion_success_rate']:.1f}%")
        print(f"  Export formats generated: {', '.join(summary['export_formats_generated'])}")
        
        if summary['errors'] and args.verbose:
            print(f"\nErrors encountered:")
            for error in summary['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(summary['errors']) > 10:
                print(f"  ... and {len(summary['errors']) - 10} more errors")
    
    successful_files = sum(1 for r in results if r.success)
    return 0 if successful_files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())