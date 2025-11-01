"""
Advanced Features Demo for MathML Parser
========================================

This file demonstrates advanced features of the MathML Parser including
matrices, advanced functions, calculus notation, and complex expressions.
"""

from mathml_parser import MathMLParser, parse_safe

def advanced_functions():
    """Demonstrate advanced mathematical functions."""
    print("=== Advanced Functions ===")
    
    expressions = [
        "sinh(x)",
        "cosh(y)",
        "tanh(z)",
        "arcsin(0.5)",
        "arccos(1)",
        "arctan(π/4)",
        "ln(e^x)",
        "log(100)",
        "exp(i*π)",
        "floor(3.7)",
        "ceil(2.1)",
        "round(π)"
    ]
    
    parser = MathMLParser(enable_metrics=True)
    
    for expr in expressions:
        result = parser.parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
        print()


def matrix_operations():
    """Demonstrate matrix notation."""
    print("=== Matrix Operations ===")
    
    expressions = [
        "[1, 2, 3]",  # Row vector
        "[1; 2; 3]",  # Column vector  
        "[1, 2; 3, 4]",  # 2x2 matrix
        "[a, b, c; d, e, f; g, h, i]",  # 3x3 matrix
        "[sin(x), cos(x); -cos(x), sin(x)]"  # Matrix with functions
    ]
    
    for expr in expressions:
        result = parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
        print()


def calculus_notation():
    """Demonstrate calculus and advanced notation."""
    print("=== Calculus Notation ===")
    
    expressions = [
        "∫(x^2)dx",
        "∫(sin(x))dx from 0 to π",
        "∑(i=1 to n) i^2", 
        "∏(k=1 to m) k",
        "lim(x→0) sin(x)/x",
        "lim(n→∞) (1+1/n)^n",
        "∂f/∂x",
        "d²y/dx²"
    ]
    
    parser = MathMLParser(enable_validation=True)
    
    for expr in expressions:
        result = parser.parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
            if result.error.suggestions:
                print(f"Suggestions: {', '.join(result.error.suggestions)}")
        print()


def complex_expressions():
    """Demonstrate complex mathematical expressions."""
    print("=== Complex Expressions ===")
    
    expressions = [
        "e^(i*π) + 1 = 0",  # Euler's identity
        "x = (-b ± √(b²-4ac))/(2a)",  # Quadratic formula
        "∫(-∞ to ∞) e^(-x²) dx = √π",  # Gaussian integral
        "∑(n=0 to ∞) x^n/n! = e^x",  # Taylor series for e^x
        "sin²(θ) + cos²(θ) = 1",  # Pythagorean identity
        "lim(h→0) (f(x+h) - f(x))/h",  # Derivative definition
        "|z|² = z * z̄",  # Complex modulus
        "∮_C f(z)dz = 2πi ∑ Res(f,a_k)"  # Residue theorem
    ]
    
    parser = MathMLParser(enable_metrics=True, enable_validation=True)
    
    for expr in expressions:
        result = parser.parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
            if result.metrics:
                print(f"Complexity Score: {result.metrics.complexity_score:.2f}")
                print(f"Features Used: {', '.join(result.metrics.features_used)}")
        else:
            print(f"Error: {result.error.message}")
            print(f"Error Type: {result.error.error_type}")
            if result.error.context:
                print(f"Context: {result.error.context}")
        print()


def set_and_logic_notation():
    """Demonstrate set theory and logic notation."""
    print("=== Set and Logic Notation ===")
    
    expressions = [
        "{1, 2, 3, 4, 5}",  # Set
        "{x | x ∈ ℝ, x > 0}",  # Set comprehension
        "A ∪ B",  # Union
        "A ∩ B",  # Intersection
        "A ⊆ B",  # Subset
        "∅",  # Empty set
        "ℕ, ℤ, ℚ, ℝ, ℂ",  # Number sets
        "∀x ∈ ℝ: x² ≥ 0",  # Universal quantifier
        "∃y ∈ ℕ: y > 100"  # Existential quantifier
    ]
    
    for expr in expressions:
        result = parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
        print()


def subscripts_and_superscripts():
    """Demonstrate subscript and superscript notation."""
    print("=== Subscripts and Superscripts ===")
    
    expressions = [
        "x₁ + x₂ + x₃",
        "a^n + b^n = c^n",
        "H₂O + NaCl",
        "E = mc²",
        "x^(n+1)",
        "a_{i,j}",
        "∑_{i=1}^n x_i",
        "f'(x) = df/dx",
        "f''(x) = d²f/dx²"
    ]
    
    for expr in expressions:
        result = parse_safe(expr)
        print(f"Expression: {expr}")
        if result.success:
            print(f"MathML: {result.mathml}")
        else:
            print(f"Error: {result.error.message}")
        print()


def performance_test():
    """Test parser performance with various expression types."""
    print("=== Performance Test ===")
    
    parser = MathMLParser(enable_metrics=True)
    
    # Test expressions of varying complexity
    test_cases = [
        ("Simple", "x + y"),
        ("Medium", "sin(x^2) + cos(y^2)"),
        ("Complex", "∫(e^(-x²))dx from -∞ to ∞"),
        ("Very Complex", "∑(n=1 to ∞) ((-1)^(n+1))/(2n-1) = π/4"),
        ("Matrix", "[sin(α), cos(α); -cos(α), sin(α)]"),
        ("Greek Heavy", "α²β + γ³δ - ε⁴ζ + η⁵θ")
    ]
    
    for name, expr in test_cases:
        result = parser.parse_safe(expr)
        print(f"{name} Expression: {expr}")
        if result.success:
            if result.metrics:
                print(f"  Parse Time: {result.metrics.parse_time:.4f}s")
                print(f"  Complexity: {result.metrics.complexity_score:.2f}")
                print(f"  Input Length: {result.metrics.input_length}")
                print(f"  Output Length: {result.metrics.output_length}")
                print(f"  Features: {', '.join(result.metrics.features_used)}")
        else:
            print(f"  Error: {result.error.message}")
        print()
    
    # Show overall metrics
    metrics_summary = parser.get_metrics_summary()
    print("=== Overall Performance Metrics ===")
    print(f"Total parses: {metrics_summary.get('total_parses', 0)}")
    if 'avg_parse_time' in metrics_summary:
        print(f"Average parse time: {metrics_summary['avg_parse_time']:.4f}s")
        print(f"Average complexity: {metrics_summary['avg_complexity']:.2f}")
        print(f"Max input length: {metrics_summary['max_input_length']}")
        print("Feature frequency:")
        for feature, count in metrics_summary.get('features_frequency', {}).items():
            print(f"  {feature}: {count}")


if __name__ == "__main__":
    print("MathML Parser - Advanced Features Demo")
    print("=" * 50)
    print()
    
    advanced_functions()
    matrix_operations()
    calculus_notation()
    complex_expressions()
    set_and_logic_notation()
    subscripts_and_superscripts()
    performance_test()
    
    print("Advanced features demo completed successfully!")