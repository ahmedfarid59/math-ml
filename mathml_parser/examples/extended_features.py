"""
Extended Feature Demonstration for MathML Parser
===============================================

This comprehensive demo showcases all the enhanced features including:
- Implicit multiplication
- Extended calculus notation (derivatives, integrals)
- Vector operations
- Number theory and logic
- LaTeX input support
- Expression optimization
- Multiple output formats
"""

from mathml_parser import (
    parse, parse_safe, parse_latex, parse_and_optimize, parse_to_format,
    get_all_formats, optimize_expression, latex_to_standard,
    MultiFormatRenderer, ExpressionOptimizer
)

def demonstrate_implicit_multiplication():
    """Demonstrate implicit multiplication features."""
    print("=== Implicit Multiplication ===")
    
    expressions = [
        "2x",           # Number times variable
        "3(x+1)",       # Number times parentheses
        "(a)(b)",       # Parentheses multiplication
        "sin x",        # Function with implicit argument
        "2π",           # Number times constant
        "xy",           # Variables multiplication
    ]
    
    for expr in expressions:
        try:
            result = parse_safe(expr)
            if result.success:
                print(f"✓ {expr} → MathML generated")
            else:
                print(f"✗ {expr} → {result.error.message}")
        except Exception as e:
            print(f"✗ {expr} → {str(e)}")
    print()


def demonstrate_calculus_notation():
    """Demonstrate extended calculus notation."""
    print("=== Extended Calculus Notation ===")
    
    expressions = [
        "d/dx(x^2)",                    # Derivative
        "∂/∂x(xy)",                     # Partial derivative
        "d²/dx²(sin(x))",               # Second derivative
        "∂²/∂x²(x²y)",                  # Second partial derivative
        "∫ x^2 dx",                     # Indefinite integral
        "∫ sin(x) dx from 0 to π",     # Definite integral
        "∬ xy dx dy",                   # Double integral
        "∭ xyz dx dy dz",               # Triple integral
    ]
    
    for expr in expressions:
        try:
            result = parse_safe(expr)
            if result.success:
                print(f"✓ {expr} → MathML generated")
            else:
                print(f"✗ {expr} → {result.error.message}")
        except Exception as e:
            print(f"✗ {expr} → {str(e)}")
    print()


def demonstrate_vector_operations():
    """Demonstrate vector operations."""
    print("=== Vector Operations ===")
    
    expressions = [
        "vec(v)",              # Vector notation
        "dot(a, b)",           # Dot product
        "cross(u, v)",         # Cross product
        "grad(f)",             # Gradient
        "div(F)",              # Divergence
        "curl(F)",             # Curl
    ]
    
    for expr in expressions:
        try:
            result = parse_safe(expr)
            if result.success:
                print(f"✓ {expr} → MathML generated")
            else:
                print(f"✗ {expr} → {result.error.message}")
        except Exception as e:
            print(f"✗ {expr} → {str(e)}")
    print()


def demonstrate_number_theory_logic():
    """Demonstrate number theory and logic operations."""
    print("=== Number Theory & Logic ===")
    
    expressions = [
        "gcd(12, 18)",                        # Greatest common divisor
        "lcm(4, 6)",                          # Least common multiple
        "∀x∈ℝ: x² ≥ 0",                      # Universal quantifier
        "∃y∈ℕ: y > 100",                     # Existential quantifier
        "P ∧ Q",                             # Logical AND
        "P ∨ Q",                             # Logical OR
        "¬P",                                # Logical NOT
        "P ⇒ Q",                             # Logical implies
        "P ⇔ Q",                             # Logical if and only if
    ]
    
    for expr in expressions:
        try:
            result = parse_safe(expr)
            if result.success:
                print(f"✓ {expr} → MathML generated")
            else:
                print(f"✗ {expr} → {result.error.message}")
        except Exception as e:
            print(f"✗ {expr} → {str(e)}")
    print()


def demonstrate_latex_support():
    """Demonstrate LaTeX input support."""
    print("=== LaTeX Input Support ===")
    
    latex_expressions = [
        r"\\frac{x^2 + 1}{x - 1}",          # Fraction
        r"\\sqrt{x^2 + y^2}",               # Square root
        r"\\sqrt[3]{x}",                    # Cube root
        r"\\alpha + \\beta = \\gamma",      # Greek letters
        r"\\sin(\\theta)",                  # Trigonometric functions
        r"\\int_0^{\\pi} \\sin(x) dx",     # Integral with limits
        r"\\sum_{n=1}^{\\infty} \\frac{1}{n^2}",  # Summation
        r"\\frac{\\partial f}{\\partial x}",      # Partial derivative
    ]
    
    for latex_expr in latex_expressions:
        try:
            # Convert LaTeX to standard notation
            standard = latex_to_standard(latex_expr)
            print(f"LaTeX: {latex_expr}")
            print(f"Standard: {standard}")
            
            # Parse to MathML
            mathml = parse_latex(latex_expr)
            print(f"✓ Successfully converted to MathML")
            print()
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            print()


def demonstrate_expression_optimization():
    """Demonstrate expression optimization."""
    print("=== Expression Optimization ===")
    
    expressions = [
        "x + 0",                    # Addition identity
        "x * 1",                    # Multiplication identity
        "sin²(x) + cos²(x)",        # Trigonometric identity
        "ln(exp(x))",               # Inverse functions
        "x^0",                      # Power identity
        "sqrt(x²)",                 # Root identity
        "2*x + 3*x",                # Like terms
    ]
    
    optimizer = ExpressionOptimizer()
    
    for expr in expressions:
        try:
            optimized = optimizer.optimize_expression(expr)
            suggestions = optimizer.suggest_optimizations(expr)
            
            print(f"Original: {expr}")
            print(f"Optimized: {optimized}")
            if suggestions:
                print(f"Suggestions: {'; '.join(suggestions)}")
            print()
        except Exception as e:
            print(f"Error optimizing {expr}: {str(e)}")
            print()


def demonstrate_multi_format_output():
    """Demonstrate multiple output formats."""
    print("=== Multiple Output Formats ===")
    
    expression = "∫₀^π sin(x) dx = 2"
    
    try:
        # Get all formats
        all_formats = get_all_formats(expression)
        
        print(f"Expression: {expression}")
        print("-" * 50)
        
        for format_name, rendered in all_formats.items():
            print(f"{format_name.upper()}:")
            print(rendered[:200] + "..." if len(rendered) > 200 else rendered)
            print()
            
    except Exception as e:
        print(f"Error generating formats: {str(e)}")


def demonstrate_advanced_expressions():
    """Demonstrate complex mathematical expressions."""
    print("=== Advanced Mathematical Expressions ===")
    
    advanced_expressions = [
        # Physics equations
        "E = mc²",
        "F = ma",
        "∇²φ = ρ/ε₀",
        
        # Famous mathematical identities
        "e^(iπ) + 1 = 0",                    # Euler's identity
        "∑(n=0 to ∞) x^n/n! = e^x",         # Taylor series
        "∫(-∞ to ∞) e^(-x²) dx = √π",       # Gaussian integral
        
        # Calculus theorems
        "d/dx(∫ᵃˣ f(t) dt) = f(x)",         # Fundamental theorem
        "∫∫_D (∂Q/∂x - ∂P/∂y) dx dy = ∮_C P dx + Q dy",  # Green's theorem
        
        # Linear algebra
        "det([a,b;c,d]) = ad - bc",
        "||v|| = √(v·v)",
        
        # Number theory
        "φ(n) = n∏(p|n)(1 - 1/p)",          # Euler's totient function
        "ζ(s) = ∑(n=1 to ∞) 1/n^s",         # Riemann zeta function
    ]
    
    parser_with_metrics = None
    try:
        from mathml_parser import MathMLParser
        parser_with_metrics = MathMLParser(enable_metrics=True)
    except:
        pass
    
    for expr in advanced_expressions:
        try:
            if parser_with_metrics:
                result = parser_with_metrics.parse_safe(expr)
                if result.success:
                    print(f"✓ {expr}")
                    if result.metrics:
                        print(f"  Complexity: {result.metrics.complexity_score:.2f}")
                        print(f"  Features: {', '.join(result.metrics.features_used)}")
                else:
                    print(f"✗ {expr} → {result.error.message}")
            else:
                result = parse_safe(expr)
                if result.success:
                    print(f"✓ {expr}")
                else:
                    print(f"✗ {expr} → {result.error.message}")
        except Exception as e:
            print(f"✗ {expr} → {str(e)}")
        print()


def performance_comparison():
    """Compare performance with and without optimization."""
    print("=== Performance Comparison ===")
    
    import time
    
    test_expressions = [
        "x^2 + 2*x + 1",
        "sin(π/2) + cos(0)",
        "∫ x² dx from 0 to 1",
        "α² + β² = γ²",
        "∑(i=1 to n) i²",
    ]
    
    # Test standard parsing
    start_time = time.time()
    for expr in test_expressions * 100:  # Repeat for timing
        try:
            parse(expr)
        except:
            pass
    standard_time = time.time() - start_time
    
    # Test optimized parsing
    start_time = time.time()
    for expr in test_expressions * 100:  # Repeat for timing
        try:
            parse_and_optimize(expr)
        except:
            pass
    optimized_time = time.time() - start_time
    
    print(f"Standard parsing: {standard_time:.4f}s")
    print(f"Optimized parsing: {optimized_time:.4f}s")
    print(f"Performance ratio: {optimized_time/standard_time:.2f}x")
    print()


if __name__ == "__main__":
    print("MathML Parser - Extended Feature Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_implicit_multiplication()
    demonstrate_calculus_notation()
    demonstrate_vector_operations()
    demonstrate_number_theory_logic()
    demonstrate_latex_support()
    demonstrate_expression_optimization()
    demonstrate_multi_format_output()
    demonstrate_advanced_expressions()
    performance_comparison()
    
    print("Extended feature demonstration completed!")
    print("\nFor more examples, check the documentation and test files.")
    print("Package now supports:")
    print("✓ Implicit multiplication")
    print("✓ Extended calculus notation") 
    print("✓ Vector operations")
    print("✓ Number theory & logic")
    print("✓ LaTeX input support")
    print("✓ Expression optimization")
    print("✓ Multiple output formats")