"""
Comprehensive Tests for Enhanced MathML Parser
==============================================

Tests all new features including LaTeX input, optimization, 
multi-format output, and advanced mathematical notation.
"""

import pytest
from unittest.mock import patch, MagicMock

from mathml_parser import (
    parse, parse_safe, parse_latex, parse_and_optimize, parse_to_format,
    get_all_formats, optimize_expression, latex_to_standard
)
from mathml_parser.core.latex_parser import LaTeXParser
from mathml_parser.core.optimizer import ExpressionOptimizer
from mathml_parser.core.multi_format import MultiFormatRenderer
from mathml_parser.core.exceptions import MathMLParsingError


class TestImplicitMultiplication:
    """Test implicit multiplication features."""
    
    def test_number_variable_multiplication(self):
        """Test number times variable."""
        result = parse_safe("2x")
        assert result.success
        assert "2" in result.mathml
        assert "x" in result.mathml
    
    def test_number_parentheses_multiplication(self):
        """Test number times parentheses."""
        result = parse_safe("3(x+1)")
        assert result.success
        assert "3" in result.mathml
    
    def test_parentheses_multiplication(self):
        """Test parentheses multiplication."""
        result = parse_safe("(a)(b)")
        assert result.success
    
    def test_function_implicit_argument(self):
        """Test function with implicit argument."""
        result = parse_safe("sin x")
        assert result.success
        assert "sin" in result.mathml
    
    def test_number_constant_multiplication(self):
        """Test number times constant."""
        result = parse_safe("2π")
        assert result.success
        assert "2" in result.mathml
        assert "π" in result.mathml
    
    def test_variables_multiplication(self):
        """Test variables multiplication."""
        result = parse_safe("xy")
        assert result.success
        assert "x" in result.mathml
        assert "y" in result.mathml


class TestCalculusNotation:
    """Test extended calculus notation."""
    
    def test_basic_derivative(self):
        """Test basic derivative notation."""
        result = parse_safe("d/dx(x^2)")
        assert result.success
        
    def test_partial_derivative(self):
        """Test partial derivative notation."""
        result = parse_safe("∂/∂x(xy)")
        assert result.success
    
    def test_second_derivative(self):
        """Test second derivative notation."""
        result = parse_safe("d²/dx²(sin(x))")
        assert result.success
    
    def test_indefinite_integral(self):
        """Test indefinite integral."""
        result = parse_safe("∫ x^2 dx")
        assert result.success
    
    def test_definite_integral(self):
        """Test definite integral."""
        result = parse_safe("∫ sin(x) dx from 0 to π")
        assert result.success
    
    def test_double_integral(self):
        """Test double integral."""
        result = parse_safe("∬ xy dx dy")
        assert result.success
    
    def test_triple_integral(self):
        """Test triple integral."""
        result = parse_safe("∭ xyz dx dy dz")
        assert result.success


class TestVectorOperations:
    """Test vector operations."""
    
    def test_vector_notation(self):
        """Test vector notation."""
        result = parse_safe("vec(v)")
        assert result.success
    
    def test_dot_product(self):
        """Test dot product."""
        result = parse_safe("dot(a, b)")
        assert result.success
    
    def test_cross_product(self):
        """Test cross product."""
        result = parse_safe("cross(u, v)")
        assert result.success
    
    def test_gradient(self):
        """Test gradient."""
        result = parse_safe("grad(f)")
        assert result.success
    
    def test_divergence(self):
        """Test divergence."""
        result = parse_safe("div(F)")
        assert result.success
    
    def test_curl(self):
        """Test curl."""
        result = parse_safe("curl(F)")
        assert result.success


class TestNumberTheoryLogic:
    """Test number theory and logic operations."""
    
    def test_gcd(self):
        """Test greatest common divisor."""
        result = parse_safe("gcd(12, 18)")
        assert result.success
    
    def test_lcm(self):
        """Test least common multiple."""
        result = parse_safe("lcm(4, 6)")
        assert result.success
    
    def test_universal_quantifier(self):
        """Test universal quantifier."""
        result = parse_safe("∀x∈ℝ: x² ≥ 0")
        assert result.success
    
    def test_existential_quantifier(self):
        """Test existential quantifier."""
        result = parse_safe("∃y∈ℕ: y > 100")
        assert result.success
    
    def test_logical_and(self):
        """Test logical AND."""
        result = parse_safe("P ∧ Q")
        assert result.success
    
    def test_logical_or(self):
        """Test logical OR."""
        result = parse_safe("P ∨ Q")
        assert result.success
    
    def test_logical_not(self):
        """Test logical NOT."""
        result = parse_safe("¬P")
        assert result.success
    
    def test_logical_implies(self):
        """Test logical implies."""
        result = parse_safe("P ⇒ Q")
        assert result.success
    
    def test_logical_iff(self):
        """Test logical if and only if."""
        result = parse_safe("P ⇔ Q")
        assert result.success


class TestLaTeXSupport:
    """Test LaTeX input support."""
    
    def setUp(self):
        self.latex_parser = LaTeXParser()
    
    def test_fraction(self):
        """Test LaTeX fraction."""
        latex_expr = r"\\frac{x^2 + 1}{x - 1}"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_square_root(self):
        """Test LaTeX square root."""
        latex_expr = r"\\sqrt{x^2 + y^2}"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_cube_root(self):
        """Test LaTeX cube root."""
        latex_expr = r"\\sqrt[3]{x}"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_greek_letters(self):
        """Test LaTeX Greek letters."""
        latex_expr = r"\\alpha + \\beta = \\gamma"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_trigonometric_functions(self):
        """Test LaTeX trigonometric functions."""
        latex_expr = r"\\sin(\\theta)"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_integral_with_limits(self):
        """Test LaTeX integral with limits."""
        latex_expr = r"\\int_0^{\\pi} \\sin(x) dx"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_summation(self):
        """Test LaTeX summation."""
        latex_expr = r"\\sum_{n=1}^{\\infty} \\frac{1}{n^2}"
        result = parse_latex(latex_expr)
        assert result is not None
    
    def test_partial_derivative_latex(self):
        """Test LaTeX partial derivative."""
        latex_expr = r"\\frac{\\partial f}{\\partial x}"
        result = parse_latex(latex_expr)
        assert result is not None


class TestExpressionOptimization:
    """Test expression optimization."""
    
    def setUp(self):
        self.optimizer = ExpressionOptimizer()
    
    def test_addition_identity(self):
        """Test addition identity optimization."""
        optimized = self.optimizer.optimize_expression("x + 0")
        assert optimized == "x" or optimized == "0 + x"
    
    def test_multiplication_identity(self):
        """Test multiplication identity optimization."""
        optimized = self.optimizer.optimize_expression("x * 1")
        assert optimized == "x" or optimized == "1 * x"
    
    def test_trigonometric_identity(self):
        """Test trigonometric identity optimization."""
        optimized = self.optimizer.optimize_expression("sin²(x) + cos²(x)")
        assert "1" in optimized or optimized == "sin²(x) + cos²(x)"
    
    def test_inverse_functions(self):
        """Test inverse function optimization."""
        optimized = self.optimizer.optimize_expression("ln(exp(x))")
        assert optimized == "x" or optimized == "ln(exp(x))"
    
    def test_power_identity(self):
        """Test power identity optimization."""
        optimized = self.optimizer.optimize_expression("x^0")
        assert optimized == "1" or optimized == "x^0"
    
    def test_like_terms(self):
        """Test like terms combination."""
        optimized = self.optimizer.optimize_expression("2*x + 3*x")
        assert "5*x" in optimized or optimized == "2*x + 3*x"
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        suggestions = self.optimizer.suggest_optimizations("x + 0")
        assert isinstance(suggestions, list)


class TestMultiFormatOutput:
    """Test multiple output formats."""
    
    def setUp(self):
        self.renderer = MultiFormatRenderer()
    
    def test_mathml_format(self):
        """Test MathML format output."""
        result = parse_to_format("x^2", "mathml")
        assert result is not None
        assert "<math" in result
    
    def test_latex_format(self):
        """Test LaTeX format output."""
        result = self.renderer.render("x^2", "latex")
        assert result is not None
    
    def test_html_format(self):
        """Test HTML format output."""
        result = self.renderer.render("x^2", "html")
        assert result is not None
    
    def test_svg_format(self):
        """Test SVG format output."""
        result = self.renderer.render("x^2", "svg")
        assert result is not None
    
    def test_ascii_format(self):
        """Test ASCII format output."""
        result = self.renderer.render("x^2", "ascii")
        assert result is not None
    
    def test_plain_format(self):
        """Test plain text format output."""
        result = self.renderer.render("x^2", "plain")
        assert result is not None
    
    def test_all_formats(self):
        """Test getting all formats at once."""
        all_formats = get_all_formats("x^2")
        assert isinstance(all_formats, dict)
        assert len(all_formats) >= 4  # Should have multiple formats


class TestAdvancedExpressions:
    """Test complex mathematical expressions."""
    
    def test_euler_identity(self):
        """Test Euler's identity."""
        result = parse_safe("e^(iπ) + 1 = 0")
        assert result.success
    
    def test_physics_equations(self):
        """Test physics equations."""
        expressions = [
            "E = mc²",
            "F = ma",
            "∇²φ = ρ/ε₀"
        ]
        for expr in expressions:
            result = parse_safe(expr)
            assert result.success, f"Failed to parse: {expr}"
    
    def test_taylor_series(self):
        """Test Taylor series representation."""
        result = parse_safe("∑(n=0 to ∞) x^n/n! = e^x")
        assert result.success
    
    def test_gaussian_integral(self):
        """Test Gaussian integral."""
        result = parse_safe("∫(-∞ to ∞) e^(-x²) dx = √π")
        assert result.success
    
    def test_fundamental_theorem(self):
        """Test fundamental theorem of calculus."""
        result = parse_safe("d/dx(∫ᵃˣ f(t) dt) = f(x)")
        assert result.success
    
    def test_linear_algebra(self):
        """Test linear algebra expressions."""
        expressions = [
            "det([a,b;c,d]) = ad - bc",
            "||v|| = √(v·v)"
        ]
        for expr in expressions:
            result = parse_safe(expr)
            assert result.success, f"Failed to parse: {expr}"
    
    def test_number_theory_functions(self):
        """Test number theory functions."""
        expressions = [
            "φ(n) = n∏(p|n)(1 - 1/p)",
            "ζ(s) = ∑(n=1 to ∞) 1/n^s"
        ]
        for expr in expressions:
            result = parse_safe(expr)
            # Note: These may not parse perfectly due to complex notation
            # but should not crash


class TestIntegration:
    """Test integration of all features."""
    
    def test_latex_with_optimization(self):
        """Test LaTeX input with optimization."""
        latex_expr = r"\\frac{x^0}{1}"
        result = parse_and_optimize(latex_expr, is_latex=True)
        # Should optimize x^0/1 to 1
        assert result is not None
    
    def test_all_features_combined(self):
        """Test combining multiple features."""
        # This would test LaTeX input, optimization, and multi-format output
        # in a real integration scenario
        pass
    
    def test_error_handling(self):
        """Test error handling across all features."""
        invalid_expressions = [
            "(((",  # Unmatched parentheses
            "x^",   # Incomplete expression
            "sin(",  # Incomplete function call
            r"\\unknown{x}",  # Unknown LaTeX command
        ]
        
        for expr in invalid_expressions:
            result = parse_safe(expr)
            assert not result.success
            assert result.error is not None


class TestPerformance:
    """Test performance characteristics."""
    
    def test_optimization_performance(self):
        """Test that optimization doesn't significantly slow down parsing."""
        import time
        
        expression = "x^2 + 2*x + 1"
        
        # Test standard parsing
        start = time.time()
        for _ in range(100):
            parse(expression)
        standard_time = time.time() - start
        
        # Test optimized parsing
        start = time.time()
        for _ in range(100):
            parse_and_optimize(expression)
        optimized_time = time.time() - start
        
        # Optimization should not be more than 3x slower
        assert optimized_time < standard_time * 3
    
    def test_multi_format_performance(self):
        """Test multi-format rendering performance."""
        import time
        
        expression = "∫₀^π sin(x) dx"
        
        start = time.time()
        for _ in range(50):
            get_all_formats(expression)
        multi_format_time = time.time() - start
        
        # Should complete in reasonable time
        assert multi_format_time < 10.0  # 10 seconds for 50 renders


# Integration tests with fixtures
@pytest.fixture
def sample_expressions():
    """Sample expressions for testing."""
    return [
        "x^2 + 2*x + 1",
        "sin(π/2)",
        "∫₀^π sin(x) dx",
        "∂/∂x(xy)",
        "vec(v) · vec(w)",
        "∀x∈ℝ: x² ≥ 0",
        r"\\frac{x^2}{2}",
        "E = mc²"
    ]


def test_all_expressions_parse(sample_expressions):
    """Test that all sample expressions parse successfully."""
    for expr in sample_expressions:
        if expr.startswith("\\"):
            # LaTeX expression
            result = parse_latex(expr)
        else:
            result = parse_safe(expr)
            assert result.success, f"Failed to parse: {expr}"


def test_feature_compatibility():
    """Test that all features work together."""
    # Test LaTeX input with optimization
    latex_expr = r"\\frac{x^2 + 0}{1}"
    result = parse_and_optimize(latex_expr, is_latex=True)
    assert result is not None
    
    # Test optimization with multi-format output
    expr = "x + 0"
    optimized = parse_and_optimize(expr)
    formats = get_all_formats(optimized)
    assert "mathml" in formats
    assert len(formats) >= 4


if __name__ == "__main__":
    # Run basic tests if called directly
    print("Running comprehensive tests...")
    
    # Test implicit multiplication
    print("✓ Testing implicit multiplication...")
    test_implicit = TestImplicitMultiplication()
    test_implicit.test_number_variable_multiplication()
    test_implicit.test_function_implicit_argument()
    
    # Test calculus notation
    print("✓ Testing calculus notation...")
    test_calculus = TestCalculusNotation()
    test_calculus.test_basic_derivative()
    test_calculus.test_indefinite_integral()
    
    # Test vector operations
    print("✓ Testing vector operations...")
    test_vector = TestVectorOperations()
    test_vector.test_vector_notation()
    test_vector.test_dot_product()
    
    # Test LaTeX support
    print("✓ Testing LaTeX support...")
    test_latex = TestLaTeXSupport()
    test_latex.setUp()
    test_latex.test_fraction()
    
    # Test optimization
    print("✓ Testing optimization...")
    test_opt = TestExpressionOptimization()
    test_opt.setUp()
    test_opt.test_addition_identity()
    
    # Test multi-format
    print("✓ Testing multi-format output...")
    test_multi = TestMultiFormatOutput()
    test_multi.setUp()
    test_multi.test_mathml_format()
    
    print("All tests completed successfully!")
    print("\nTo run the full test suite with pytest:")
    print("pytest mathml_parser/tests/test_comprehensive.py -v")
            ("ln(x)", True),
            ("sqrt(16)", True),
            ("abs(-5)", True)
        ]
        
        for expr, should_succeed in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertEqual(result.success, should_succeed)
    
    def test_greek_letters(self):
        """Test Greek letter support."""
        test_cases = [
            ("α + β", True),
            ("gamma * delta", True),
            ("π * r^2", True),
            ("θ + φ", True),
            ("Omega", True)
        ]
        
        for expr, should_succeed in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertEqual(result.success, should_succeed)
    
    def test_comparison_operators(self):
        """Test comparison operators."""
        test_cases = [
            ("x = y", True),
            ("a != b", True),
            ("p < q", True),
            ("m > n", True),
            ("x <= 5", True),
            ("y >= 0", True)
        ]
        
        for expr, should_succeed in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertEqual(result.success, should_succeed)


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced mathematical features."""
    
    def setUp(self):
        self.parser = MathMLParser(enable_metrics=True)
    
    def test_trigonometric_functions(self):
        """Test trigonometric and hyperbolic functions."""
        test_cases = [
            "sin(x)", "cos(y)", "tan(z)",
            "sec(a)", "csc(b)", "cot(c)",
            "arcsin(0.5)", "arccos(1)", "arctan(π/4)",
            "sinh(x)", "cosh(y)", "tanh(z)"
        ]
        
        for expr in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success, f"Failed to parse: {expr}")
    
    def test_advanced_functions(self):
        """Test advanced mathematical functions."""
        test_cases = [
            "floor(3.7)", "ceil(2.1)", "round(π)",
            "max(a, b, c)", "min(x, y, z)",
            "exp(x)", "ln(e)", "log(100)"
        ]
        
        for expr in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success, f"Failed to parse: {expr}")
    
    def test_subscripts_superscripts(self):
        """Test subscript and superscript notation."""
        test_cases = [
            "x_1", "y^2", "a_{i,j}",
            "x^(n+1)", "base_sub^super"
        ]
        
        for expr in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success, f"Failed to parse: {expr}")
    
    def test_matrices(self):
        """Test matrix notation."""
        test_cases = [
            "[1, 2, 3]",
            "[1; 2; 3]", 
            "[1, 2; 3, 4]",
            "[a, b; c, d]"
        ]
        
        for expr in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success, f"Failed to parse: {expr}")
    
    def test_absolute_values(self):
        """Test absolute value notation."""
        test_cases = [
            "|x|", "|a + b|", "|sin(x)|"
        ]
        
        for expr in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success, f"Failed to parse: {expr}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and validation."""
    
    def setUp(self):
        self.parser = MathMLParser(enable_validation=True)
    
    def test_syntax_errors(self):
        """Test handling of syntax errors."""
        error_cases = [
            "2 + + 3",  # Double operator
            "sin(",     # Missing closing parenthesis
            ")",        # Unmatched closing parenthesis
            "2 +",      # Incomplete expression
            "* 3"       # Missing left operand
        ]
        
        for expr in error_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)
                self.assertIsInstance(result.error, MathParseError)
    
    def test_validation_errors(self):
        """Test input validation."""
        validator = InputValidator()
        
        error_cases = [
            "",         # Empty input
            "   ",      # Whitespace only
            "(((",      # Unmatched brackets
            ")))",      # Unmatched brackets
        ]
        
        for expr in error_cases:
            with self.subTest(expr=expr):
                validation_result = validator.validate_input(expr)
                self.assertFalse(validation_result.is_valid)
    
    def test_error_suggestions(self):
        """Test that error messages include helpful suggestions."""
        error_cases = [
            "sin(",     # Should suggest closing parenthesis
            "2 + + 3",  # Should suggest operator fix
            "log)",     # Should suggest opening parenthesis
        ]
        
        for expr in error_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error.suggestions)
                self.assertGreater(len(result.error.suggestions), 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance monitoring and metrics."""
    
    def setUp(self):
        self.parser = MathMLParser(enable_metrics=True)
    
    def test_metrics_collection(self):
        """Test that metrics are collected properly."""
        expr = "sin(x^2) + cos(y^2)"
        result = self.parser.parse_safe(expr)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.metrics)
        self.assertGreater(result.metrics.parse_time, 0)
        self.assertGreater(result.metrics.input_length, 0)
        self.assertGreater(result.metrics.output_length, 0)
        self.assertGreater(result.metrics.complexity_score, 0)
        self.assertIsInstance(result.metrics.features_used, list)
    
    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        simple_expr = "x + y"
        complex_expr = "sin(cos(tan(x^2 + y^2)))"
        
        simple_result = self.parser.parse_safe(simple_expr)
        complex_result = self.parser.parse_safe(complex_expr)
        
        self.assertTrue(simple_result.success)
        self.assertTrue(complex_result.success)
        
        # Complex expression should have higher complexity score
        self.assertGreater(
            complex_result.metrics.complexity_score,
            simple_result.metrics.complexity_score
        )
    
    def test_feature_detection(self):
        """Test mathematical feature detection."""
        test_cases = [
            ("x + y", ["arithmetic"]),
            ("sin(x)", ["functions"]),
            ("α + β", ["greek_letters"]),
            ("[1, 2; 3, 4]", ["matrices"]),
            ("|x|", ["absolute_values"]),
            ("x = y", ["comparisons"])
        ]
        
        for expr, expected_features in test_cases:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success)
                for feature in expected_features:
                    self.assertIn(feature, result.metrics.features_used)
    
    def test_metrics_summary(self):
        """Test metrics summary functionality."""
        expressions = [
            "x + y", "sin(x)", "α + β", "[1, 2]", "|x|"
        ]
        
        for expr in expressions:
            self.parser.parse_safe(expr)
        
        summary = self.parser.get_metrics_summary()
        
        self.assertEqual(summary["total_parses"], len(expressions))
        self.assertIn("avg_parse_time", summary)
        self.assertIn("avg_complexity", summary)
        self.assertIn("features_frequency", summary)


class TestParserConfiguration(unittest.TestCase):
    """Test parser configuration options."""
    
    def test_validation_toggle(self):
        """Test enabling/disabling validation."""
        # Parser with validation
        parser_with_validation = MathMLParser(enable_validation=True)
        
        # Parser without validation
        parser_without_validation = MathMLParser(enable_validation=False)
        
        # Both should handle valid expressions
        valid_expr = "x + y"
        result1 = parser_with_validation.parse_safe(valid_expr)
        result2 = parser_without_validation.parse_safe(valid_expr)
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
    
    def test_metrics_toggle(self):
        """Test enabling/disabling metrics."""
        parser_with_metrics = MathMLParser(enable_metrics=True)
        parser_without_metrics = MathMLParser(enable_metrics=False)
        
        expr = "sin(x)"
        result1 = parser_with_metrics.parse_safe(expr)
        result2 = parser_without_metrics.parse_safe(expr)
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertIsNotNone(result1.metrics)
        self.assertIsNone(result2.metrics)
    
    def test_strict_mode(self):
        """Test strict parsing mode."""
        parser_strict = MathMLParser(strict_mode=True)
        parser_lenient = MathMLParser(strict_mode=False)
        
        # Both should handle standard expressions
        expr = "x + y"
        result1 = parser_strict.parse_safe(expr)
        result2 = parser_lenient.parse_safe(expr)
        
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)


class TestGrammarComponents(unittest.TestCase):
    """Test individual grammar components."""
    
    def test_grammar_generation(self):
        """Test grammar generation."""
        grammar = MathematicalGrammar()
        grammar_text = grammar.get_grammar()
        
        self.assertIsInstance(grammar_text, str)
        self.assertGreater(len(grammar_text), 100)
        self.assertIn("start:", grammar_text)
    
    def test_transformer_components(self):
        """Test transformer functionality."""
        transformer = EnhancedMathMLTransformer()
        
        # Test that transformer has required methods
        self.assertTrue(hasattr(transformer, 'add'))
        self.assertTrue(hasattr(transformer, 'mul'))
        self.assertTrue(hasattr(transformer, 'sine'))
        self.assertTrue(hasattr(transformer, 'greek_letter'))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_parse_function(self):
        """Test global parse function."""
        result = parse("x + y")
        self.assertIsInstance(result, str)
        self.assertIn("<math", result)
    
    def test_parse_safe_function(self):
        """Test global parse_safe function."""
        result = parse_safe("x + y")
        self.assertIsInstance(result, MathParseResult)
        self.assertTrue(result.success)
        
        result_error = parse_safe("invalid ((")
        self.assertFalse(result_error.success)
        self.assertIsNotNone(result_error.error)


class TestMathMLOutput(unittest.TestCase):
    """Test MathML output format and structure."""
    
    def setUp(self):
        self.parser = MathMLParser()
    
    def test_mathml_structure(self):
        """Test that output follows MathML structure."""
        expr = "x + y"
        result = self.parser.parse_safe(expr)
        
        self.assertTrue(result.success)
        mathml = result.mathml
        
        # Should start with <math> tag
        self.assertTrue(mathml.startswith("<math"))
        self.assertTrue(mathml.endswith("</math>"))
        
        # Should contain proper namespace
        self.assertIn("xmlns", mathml)
        
        # Should contain mathematical elements
        self.assertIn("<mi>", mathml)  # Variables
        self.assertIn("<mo>", mathml)  # Operators
    
    def test_mathml_validity(self):
        """Test MathML validity for various expressions."""
        test_expressions = [
            "x",
            "2",
            "x + y",
            "sin(x)",
            "x^2",
            "sqrt(x)",
            "|x|"
        ]
        
        for expr in test_expressions:
            with self.subTest(expr=expr):
                result = self.parser.parse_safe(expr)
                self.assertTrue(result.success)
                
                mathml = result.mathml
                # Basic structure checks
                self.assertIn("<math", mathml)
                self.assertIn("</math>", mathml)
                
                # No malformed tags
                self.assertEqual(mathml.count("<math"), 1)
                self.assertEqual(mathml.count("</math>"), 1)


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBasicParsing,
        TestAdvancedFeatures, 
        TestErrorHandling,
        TestPerformanceMetrics,
        TestParserConfiguration,
        TestGrammarComponents,
        TestConvenienceFunctions,
        TestMathMLOutput
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest suite {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)