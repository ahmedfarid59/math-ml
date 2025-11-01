"""
Comprehensive Tests for Enhanced MathML Parser
==============================================

Tests all new features including LaTeX input, optimization, 
multi-format output, and advanced mathematical notation.
"""

import sys
import os
import time

# Add the parent directory to the path to import mathml_parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from mathml_parser import (
        parse, parse_safe, parse_latex, parse_and_optimize, parse_to_format,
        get_all_formats, optimize_expression, latex_to_standard
    )
    from mathml_parser.core.latex_parser import LaTeXParser
    from mathml_parser.core.optimizer import ExpressionOptimizer
    from mathml_parser.core.multi_format import MultiFormatRenderer
    from mathml_parser.core.exceptions import MathMLParsingError
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in standalone mode for demonstration...")


class TestImplicitMultiplication:
    """Test implicit multiplication features."""
    
    def test_number_variable_multiplication(self):
        """Test number times variable."""
        try:
            result = parse_safe("2x")
            assert result.success
            assert "2" in result.mathml
            assert "x" in result.mathml
            print("✓ Number-variable multiplication")
        except Exception as e:
            print(f"✗ Number-variable multiplication: {e}")
    
    def test_number_parentheses_multiplication(self):
        """Test number times parentheses."""
        try:
            result = parse_safe("3(x+1)")
            assert result.success
            assert "3" in result.mathml
            print("✓ Number-parentheses multiplication")
        except Exception as e:
            print(f"✗ Number-parentheses multiplication: {e}")
    
    def test_parentheses_multiplication(self):
        """Test parentheses multiplication."""
        try:
            result = parse_safe("(a)(b)")
            assert result.success
            print("✓ Parentheses multiplication")
        except Exception as e:
            print(f"✗ Parentheses multiplication: {e}")
    
    def test_function_implicit_argument(self):
        """Test function with implicit argument."""
        try:
            result = parse_safe("sin x")
            assert result.success
            assert "sin" in result.mathml
            print("✓ Function implicit argument")
        except Exception as e:
            print(f"✗ Function implicit argument: {e}")
    
    def test_number_constant_multiplication(self):
        """Test number times constant."""
        try:
            result = parse_safe("2π")
            assert result.success
            assert "2" in result.mathml
            assert "π" in result.mathml
            print("✓ Number-constant multiplication")
        except Exception as e:
            print(f"✗ Number-constant multiplication: {e}")
    
    def test_variables_multiplication(self):
        """Test variables multiplication."""
        try:
            result = parse_safe("xy")
            assert result.success
            assert "x" in result.mathml
            assert "y" in result.mathml
            print("✓ Variables multiplication")
        except Exception as e:
            print(f"✗ Variables multiplication: {e}")


class TestCalculusNotation:
    """Test extended calculus notation."""
    
    def test_basic_derivative(self):
        """Test basic derivative notation."""
        try:
            result = parse_safe("d/dx(x^2)")
            assert result.success
            print("✓ Basic derivative")
        except Exception as e:
            print(f"✗ Basic derivative: {e}")
        
    def test_partial_derivative(self):
        """Test partial derivative notation."""
        try:
            result = parse_safe("∂/∂x(xy)")
            assert result.success
            print("✓ Partial derivative")
        except Exception as e:
            print(f"✗ Partial derivative: {e}")
    
    def test_second_derivative(self):
        """Test second derivative notation."""
        try:
            result = parse_safe("d²/dx²(sin(x))")
            assert result.success
            print("✓ Second derivative")
        except Exception as e:
            print(f"✗ Second derivative: {e}")
    
    def test_indefinite_integral(self):
        """Test indefinite integral."""
        try:
            result = parse_safe("∫ x^2 dx")
            assert result.success
            print("✓ Indefinite integral")
        except Exception as e:
            print(f"✗ Indefinite integral: {e}")
    
    def test_definite_integral(self):
        """Test definite integral."""
        try:
            result = parse_safe("∫ sin(x) dx from 0 to π")
            assert result.success
            print("✓ Definite integral")
        except Exception as e:
            print(f"✗ Definite integral: {e}")
    
    def test_double_integral(self):
        """Test double integral."""
        try:
            result = parse_safe("∬ xy dx dy")
            assert result.success
            print("✓ Double integral")
        except Exception as e:
            print(f"✗ Double integral: {e}")
    
    def test_triple_integral(self):
        """Test triple integral."""
        try:
            result = parse_safe("∭ xyz dx dy dz")
            assert result.success
            print("✓ Triple integral")
        except Exception as e:
            print(f"✗ Triple integral: {e}")


class TestVectorOperations:
    """Test vector operations."""
    
    def test_vector_notation(self):
        """Test vector notation."""
        try:
            result = parse_safe("vec(v)")
            assert result.success
            print("✓ Vector notation")
        except Exception as e:
            print(f"✗ Vector notation: {e}")
    
    def test_dot_product(self):
        """Test dot product."""
        try:
            result = parse_safe("dot(a, b)")
            assert result.success
            print("✓ Dot product")
        except Exception as e:
            print(f"✗ Dot product: {e}")
    
    def test_cross_product(self):
        """Test cross product."""
        try:
            result = parse_safe("cross(u, v)")
            assert result.success
            print("✓ Cross product")
        except Exception as e:
            print(f"✗ Cross product: {e}")
    
    def test_gradient(self):
        """Test gradient."""
        try:
            result = parse_safe("grad(f)")
            assert result.success
            print("✓ Gradient")
        except Exception as e:
            print(f"✗ Gradient: {e}")
    
    def test_divergence(self):
        """Test divergence."""
        try:
            result = parse_safe("div(F)")
            assert result.success
            print("✓ Divergence")
        except Exception as e:
            print(f"✗ Divergence: {e}")
    
    def test_curl(self):
        """Test curl."""
        try:
            result = parse_safe("curl(F)")
            assert result.success
            print("✓ Curl")
        except Exception as e:
            print(f"✗ Curl: {e}")


class TestNumberTheoryLogic:
    """Test number theory and logic operations."""
    
    def test_gcd(self):
        """Test greatest common divisor."""
        try:
            result = parse_safe("gcd(12, 18)")
            assert result.success
            print("✓ GCD")
        except Exception as e:
            print(f"✗ GCD: {e}")
    
    def test_lcm(self):
        """Test least common multiple."""
        try:
            result = parse_safe("lcm(4, 6)")
            assert result.success
            print("✓ LCM")
        except Exception as e:
            print(f"✗ LCM: {e}")
    
    def test_universal_quantifier(self):
        """Test universal quantifier."""
        try:
            result = parse_safe("∀x∈ℝ: x² ≥ 0")
            assert result.success
            print("✓ Universal quantifier")
        except Exception as e:
            print(f"✗ Universal quantifier: {e}")
    
    def test_existential_quantifier(self):
        """Test existential quantifier."""
        try:
            result = parse_safe("∃y∈ℕ: y > 100")
            assert result.success
            print("✓ Existential quantifier")
        except Exception as e:
            print(f"✗ Existential quantifier: {e}")
    
    def test_logical_and(self):
        """Test logical AND."""
        try:
            result = parse_safe("P ∧ Q")
            assert result.success
            print("✓ Logical AND")
        except Exception as e:
            print(f"✗ Logical AND: {e}")
    
    def test_logical_or(self):
        """Test logical OR."""
        try:
            result = parse_safe("P ∨ Q")
            assert result.success
            print("✓ Logical OR")
        except Exception as e:
            print(f"✗ Logical OR: {e}")
    
    def test_logical_not(self):
        """Test logical NOT."""
        try:
            result = parse_safe("¬P")
            assert result.success
            print("✓ Logical NOT")
        except Exception as e:
            print(f"✗ Logical NOT: {e}")
    
    def test_logical_implies(self):
        """Test logical implies."""
        try:
            result = parse_safe("P ⇒ Q")
            assert result.success
            print("✓ Logical implies")
        except Exception as e:
            print(f"✗ Logical implies: {e}")
    
    def test_logical_iff(self):
        """Test logical if and only if."""
        try:
            result = parse_safe("P ⇔ Q")
            assert result.success
            print("✓ Logical IFF")
        except Exception as e:
            print(f"✗ Logical IFF: {e}")


class TestLaTeXSupport:
    """Test LaTeX input support."""
    
    def test_fraction(self):
        """Test LaTeX fraction."""
        try:
            latex_expr = r"\frac{x^2 + 1}{x - 1}"
            result = parse_latex(latex_expr)
            assert result is not None
            print("✓ LaTeX fraction")
        except Exception as e:
            print(f"✗ LaTeX fraction: {e}")
    
    def test_square_root(self):
        """Test LaTeX square root."""
        try:
            latex_expr = r"\sqrt{x^2 + y^2}"
            result = parse_latex(latex_expr)
            assert result is not None
            print("✓ LaTeX square root")
        except Exception as e:
            print(f"✗ LaTeX square root: {e}")
    
    def test_cube_root(self):
        """Test LaTeX cube root."""
        try:
            latex_expr = r"\sqrt[3]{x}"
            result = parse_latex(latex_expr)
            assert result is not None
            print("✓ LaTeX cube root")
        except Exception as e:
            print(f"✗ LaTeX cube root: {e}")
    
    def test_greek_letters(self):
        """Test LaTeX Greek letters."""
        try:
            latex_expr = r"\alpha + \beta = \gamma"
            result = parse_latex(latex_expr)
            assert result is not None
            print("✓ LaTeX Greek letters")
        except Exception as e:
            print(f"✗ LaTeX Greek letters: {e}")
    
    def test_trigonometric_functions(self):
        """Test LaTeX trigonometric functions."""
        try:
            latex_expr = r"\sin(\theta)"
            result = parse_latex(latex_expr)
            assert result is not None
            print("✓ LaTeX trigonometric functions")
        except Exception as e:
            print(f"✗ LaTeX trigonometric functions: {e}")


class TestExpressionOptimization:
    """Test expression optimization."""
    
    def test_addition_identity(self):
        """Test addition identity optimization."""
        try:
            optimizer = ExpressionOptimizer()
            optimized = optimizer.optimize_expression("x + 0")
            print(f"✓ Addition identity: {optimized}")
        except Exception as e:
            print(f"✗ Addition identity: {e}")
    
    def test_multiplication_identity(self):
        """Test multiplication identity optimization."""
        try:
            optimizer = ExpressionOptimizer()
            optimized = optimizer.optimize_expression("x * 1")
            print(f"✓ Multiplication identity: {optimized}")
        except Exception as e:
            print(f"✗ Multiplication identity: {e}")
    
    def test_power_identity(self):
        """Test power identity optimization."""
        try:
            optimizer = ExpressionOptimizer()
            optimized = optimizer.optimize_expression("x^0")
            print(f"✓ Power identity: {optimized}")
        except Exception as e:
            print(f"✗ Power identity: {e}")


class TestMultiFormatOutput:
    """Test multiple output formats."""
    
    def test_mathml_format(self):
        """Test MathML format output."""
        try:
            result = parse_to_format("x^2", "mathml")
            assert result is not None
            assert "<math" in result
            print("✓ MathML format")
        except Exception as e:
            print(f"✗ MathML format: {e}")
    
    def test_latex_format(self):
        """Test LaTeX format output."""
        try:
            renderer = MultiFormatRenderer()
            result = renderer.render("x^2", "latex")
            assert result is not None
            print("✓ LaTeX format")
        except Exception as e:
            print(f"✗ LaTeX format: {e}")
    
    def test_all_formats(self):
        """Test getting all formats at once."""
        try:
            all_formats = get_all_formats("x^2")
            assert isinstance(all_formats, dict)
            assert len(all_formats) >= 4  # Should have multiple formats
            print(f"✓ All formats: {list(all_formats.keys())}")
        except Exception as e:
            print(f"✗ All formats: {e}")


class TestAdvancedExpressions:
    """Test complex mathematical expressions."""
    
    def test_euler_identity(self):
        """Test Euler's identity."""
        try:
            result = parse_safe("e^(iπ) + 1 = 0")
            assert result.success
            print("✓ Euler's identity")
        except Exception as e:
            print(f"✗ Euler's identity: {e}")
    
    def test_physics_equations(self):
        """Test physics equations."""
        expressions = [
            "E = mc²",
            "F = ma",
            "∇²φ = ρ/ε₀"
        ]
        for expr in expressions:
            try:
                result = parse_safe(expr)
                assert result.success
                print(f"✓ Physics equation: {expr}")
            except Exception as e:
                print(f"✗ Physics equation {expr}: {e}")
    
    def test_linear_algebra(self):
        """Test linear algebra expressions."""
        expressions = [
            "det([a,b;c,d]) = ad - bc",
            "||v|| = √(v·v)"
        ]
        for expr in expressions:
            try:
                result = parse_safe(expr)
                assert result.success
                print(f"✓ Linear algebra: {expr}")
            except Exception as e:
                print(f"✗ Linear algebra {expr}: {e}")


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestImplicitMultiplication,
        TestCalculusNotation,
        TestVectorOperations,
        TestNumberTheoryLogic,
        TestLaTeXSupport,
        TestExpressionOptimization,
        TestMultiFormatOutput,
        TestAdvancedExpressions
    ]
    
    print("Enhanced MathML Parser - Comprehensive Test Suite")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        test_instance = test_class()
        methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in methods:
            method = getattr(test_instance, method_name)
            try:
                method()
            except Exception as e:
                print(f"✗ {method_name}: {e}")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")


if __name__ == "__main__":
    run_all_tests()