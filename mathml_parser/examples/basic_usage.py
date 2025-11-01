"""
Basic Usage Examples for MathML Parser
=====================================

This file demonstrates basic usage of the MathML Parser package
for common mathematical expressions.
"""

from mathml_parser import parse, parse_safe, MathMLParser

def basic_arithmetic():
    """Examples of basic arithmetic operations."""
    print("=== Basic Arithmetic ===")
    
    expressions = [
        "2 + 3",
        "x - y", 
        "a * b",
        "p / q",
        "x^2",
        "2^(3+1)"
    ]
    
    for expr in expressions:
        try:
            mathml = parse(expr)
            print(f"Expression: {expr}")
            print(f"MathML: {mathml}")
            print()
        except Exception as e:
            print(f"Error parsing '{expr}': {e}")
            print()


def functions_and_variables():
    """Examples with functions and variables."""
    print("=== Functions and Variables ===")
    
    expressions = [
        "sin(x)",
        "cos(π/2)",
        "log(e)",
        "sqrt(16)",
        "abs(-5)",
        "max(a, b, c)"
    ]
    
    for expr in expressions:
        try:
            mathml = parse(expr)
            print(f"Expression: {expr}")
            print(f"MathML: {mathml}")
            print()
        except Exception as e:
            print(f"Error parsing '{expr}': {e}")
            print()


def greek_letters_and_constants():
    """Examples with Greek letters and mathematical constants."""
    print("=== Greek Letters and Constants ===")
    
    expressions = [
        "α + β",
        "sin(θ)",
        "Δx/Δt",
        "π * r^2",
        "e^(i*π)",
        "∞"
    ]
    
    for expr in expressions:
        try:
            mathml = parse(expr)
            print(f"Expression: {expr}")
            print(f"MathML: {mathml}")
            print()
        except Exception as e:
            print(f"Error parsing '{expr}': {e}")
            print()


def safe_parsing():
    """Examples of safe parsing with error handling."""
    print("=== Safe Parsing Examples ===")
    
    expressions = [
        "2 + 3 * 4",  # Valid
        "sin(cos(x))",  # Valid
        "2 + + 3",  # Invalid - syntax error
        "sin(",  # Invalid - missing closing parenthesis
        "unknown_function(x)"  # Valid but uses unknown function
    ]
    
    for expr in expressions:
        result = parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"Success: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
            if result.error.suggestions:
                print(f"Suggestions: {', '.join(result.error.suggestions)}")
        print()


def advanced_parser_usage():
    """Examples using the MathMLParser class directly."""
    print("=== Advanced Parser Usage ===")
    
    # Create parser with custom settings
    parser = MathMLParser(
        enable_validation=True,
        enable_metrics=True,
        strict_mode=False
    )
    
    expressions = [
        "∫(x^2)dx",
        "∑(i=1 to n) i^2",
        "[1, 2; 3, 4]",  # Matrix
        "{x | x > 0}"   # Set notation
    ]
    
    for expr in expressions:
        result = parser.parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
            if result.metrics:
                print(f"Parse time: {result.metrics.parse_time:.4f}s")
                print(f"Complexity: {result.metrics.complexity_score:.2f}")
                print(f"Features: {', '.join(result.metrics.features_used)}")
        else:
            print(f"Error: {result.error.message}")
        print()


def comparison_operators():
    """Examples with comparison operators."""
    print("=== Comparison Operators ===")
    
    expressions = [
        "x = y",
        "a != b",
        "p < q",
        "m > n",
        "x <= 5",
        "y >= 0"
    ]
    
    for expr in expressions:
        try:
            mathml = parse(expr)
            print(f"Expression: {expr}")
            print(f"MathML: {mathml}")
            print()
        except Exception as e:
            print(f"Error parsing '{expr}': {e}")
            print()


if __name__ == "__main__":
    print("MathML Parser - Basic Usage Examples")
    print("=" * 50)
    print()
    
    basic_arithmetic()
    functions_and_variables()
    greek_letters_and_constants()
    safe_parsing()
    advanced_parser_usage()
    comparison_operators()
    
    print("Examples completed successfully!")