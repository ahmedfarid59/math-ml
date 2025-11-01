"""
Enhanced MathML Transformer for Mathematical Expressions
========================================================

This module provides comprehensive MathML transformation capabilities
for all supported mathematical notation including advanced functions,
Greek letters, matrices, and comparison operators.
"""

from lark import Transformer
from typing import List, Dict, Any, Optional


class EnhancedMathMLTransformer(Transformer):
    """
    Comprehensive MathML transformer supporting extensive mathematical notation.
    
    Transforms parsed mathematical expressions into proper MathML format
    with support for all advanced features including Greek letters,
    matrices, advanced functions, and comparison operators.
    """
    
    # Greek letter mapping to Unicode entities
    GREEK_LETTERS = {
        'alpha': '&alpha;', 'beta': '&beta;', 'gamma': '&gamma;', 'delta': '&delta;',
        'epsilon': '&epsilon;', 'zeta': '&zeta;', 'eta': '&eta;', 'theta': '&theta;',
        'iota': '&iota;', 'kappa': '&kappa;', 'lambda': '&lambda;', 'mu': '&mu;',
        'nu': '&nu;', 'xi': '&xi;', 'omicron': '&omicron;', 'pi': '&pi;',
        'rho': '&rho;', 'sigma': '&sigma;', 'tau': '&tau;', 'upsilon': '&upsilon;',
        'phi': '&phi;', 'chi': '&chi;', 'psi': '&psi;', 'omega': '&omega;',
        'Alpha': '&Alpha;', 'Beta': '&Beta;', 'Gamma': '&Gamma;', 'Delta': '&Delta;',
        'Epsilon': '&Epsilon;', 'Zeta': '&Zeta;', 'Eta': '&Eta;', 'Theta': '&Theta;',
        'Iota': '&Iota;', 'Kappa': '&Kappa;', 'Lambda': '&Lambda;', 'Mu': '&Mu;',
        'Nu': '&Nu;', 'Xi': '&Xi;', 'Omicron': '&Omicron;', 'Pi': '&Pi;',
        'Rho': '&Rho;', 'Sigma': '&Sigma;', 'Tau': '&Tau;', 'Upsilon': '&Upsilon;',
        'Phi': '&Phi;', 'Chi': '&Chi;', 'Psi': '&Psi;', 'Omega': '&Omega;'
    }
    
    # Mathematical constant mapping
    MATH_CONSTANTS = {
        'e': '<mi>e</mi>',
        'π': '<mi>&pi;</mi>',
        'pi': '<mi>&pi;</mi>',
        'infinity': '<mi>&infin;</mi>',
        'inf': '<mi>&infin;</mi>',
        '∞': '<mi>&infin;</mi>'
    }
    
    def __init__(self):
        """Initialize the transformer."""
        super().__init__()
    
    # Basic arithmetic operations
    def add(self, args):
        """Handle addition operation."""
        return f"""<mrow>
\t{args[0]}
\t<mo>+</mo>
\t{args[1]}
</mrow>"""

    def sub(self, args):
        """Handle subtraction operation."""
        return f"""<mrow>
\t{args[0]}
\t<mo>-</mo>
\t{args[1]}
</mrow>"""

    def mul(self, args):
        """Handle multiplication operation."""
        return f"""<mrow>
\t{args[0]}
\t<mo>*</mo>
\t{args[1]}
</mrow>"""

    def div(self, args):
        """Handle division operation using fraction notation."""
        return f"""<mfrac>
\t{args[0]}
\t{args[1]}
</mfrac>"""

    def mod(self, args):
        """Handle modulo operation."""
        return f"""<mrow>
\t{args[0]}
\t<mo>mod</mo>
\t{args[1]}
</mrow>"""

    def implicit_mul(self, args):
        """Handle implicit multiplication with invisible times."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&#x2062;</mo>
\t{args[1]}
</mrow>"""

    def pow(self, args):
        """Handle exponentiation using superscript."""
        return f"""<msup>
\t{args[0]}
\t{args[1]}
</msup>"""

    def unary_plus(self, args):
        """Handle unary plus operator."""
        return f"""<mrow>
\t<mo>+</mo>
\t{args[0]}
</mrow>"""

    def unary_minus(self, args):
        """Handle unary minus operator."""
        return f"""<mrow>
\t<mo>-</mo>
\t{args[0]}
</mrow>"""

    # Comparison operators
    def equal(self, args):
        """Handle equality comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>=</mo>
\t{args[1]}
</mrow>"""

    def not_equal(self, args):
        """Handle inequality comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&ne;</mo>
\t{args[1]}
</mrow>"""

    def less_than(self, args):
        """Handle less than comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&lt;</mo>
\t{args[1]}
</mrow>"""

    def greater_than(self, args):
        """Handle greater than comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&gt;</mo>
\t{args[1]}
</mrow>"""

    def less_equal(self, args):
        """Handle less than or equal comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&le;</mo>
\t{args[1]}
</mrow>"""

    def greater_equal(self, args):
        """Handle greater than or equal comparison."""
        return f"""<mrow>
\t{args[0]}
\t<mo>&ge;</mo>
\t{args[1]}
</mrow>"""

    # Basic elements
    def number(self, args):
        """Handle numeric values."""
        return f"<mn>{args[0]}</mn>"

    def variable(self, args):
        """Handle variable names."""
        return f"<mi>{args[0]}</mi>"

    def subscript_var(self, args):
        """Handle variables with subscripts."""
        var_with_sub = str(args[0])
        if '_' in var_with_sub:
            var_part, sub_part = var_with_sub.split('_', 1)
            return f"""<msub>
\t<mi>{var_part}</mi>
\t<mi>{sub_part}</mi>
</msub>"""
        return f"<mi>{var_with_sub}</mi>"

    def subscript(self, args):
        """Handle explicit subscript notation."""
        return f"""<msub>
\t<mi>{args[0]}</mi>
\t{args[1]}
</msub>"""

    def superscript(self, args):
        """Handle explicit superscript notation."""
        return f"""<msup>
\t<mi>{args[0]}</mi>
\t{args[1]}
</msup>"""

    def factorial(self, args):
        """Handle factorial operation."""
        return f"""<mrow>
\t{args[0]}
\t<mo>!</mo>
</mrow>"""

    def absolute_value(self, args):
        """Handle absolute value notation."""
        return f"""<mrow>
\t<mo>|</mo>
\t{args[0]}
\t<mo>|</mo>
</mrow>"""

    def group(self, args):
        """Handle parenthetical grouping."""
        return f"""<mrow>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    # Greek letters and constants
    def greek_letter(self, args):
        """Handle Greek letter notation."""
        letter = str(args[0])
        return f"<mi>{self.GREEK_LETTERS.get(letter, letter)}</mi>"

    def math_constant(self, args):
        """Handle mathematical constants."""
        constant = str(args[0])
        return self.MATH_CONSTANTS.get(constant, f'<mi>{constant}</mi>')

    # Basic mathematical functions
    def square_root(self, args):
        """Handle square root function."""
        return f"""<msqrt>
\t{args[0]}
</msqrt>"""

    def nth_root(self, args):
        """Handle nth root function."""
        return f"""<mroot>
\t{args[1]}
\t{args[0]}
</mroot>"""

    def abs_function(self, args):
        """Handle abs() function call."""
        return f"""<mrow>
\t<mo>|</mo>
\t{args[0]}
\t<mo>|</mo>
</mrow>"""

    # Advanced mathematical functions
    def natural_log(self, args):
        """Handle natural logarithm function."""
        return f"""<mrow>
\t<mi>ln</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def logarithm(self, args):
        """Handle logarithm function."""
        return f"""<mrow>
\t<mi>log</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def exponential(self, args):
        """Handle exponential function."""
        return f"""<mrow>
\t<mi>exp</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def floor_func(self, args):
        """Handle floor function."""
        return f"""<mrow>
\t<mo>&lfloor;</mo>
\t{args[0]}
\t<mo>&rfloor;</mo>
</mrow>"""

    def ceil_func(self, args):
        """Handle ceiling function."""
        return f"""<mrow>
\t<mo>&lceil;</mo>
\t{args[0]}
\t<mo>&rceil;</mo>
</mrow>"""

    def round_func(self, args):
        """Handle round function."""
        return f"""<mrow>
\t<mi>round</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def max_func(self, args):
        """Handle max function."""
        return f"""<mrow>
\t<mi>max</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def min_func(self, args):
        """Handle min function."""
        return f"""<mrow>
\t<mi>min</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    # Trigonometric functions
    def sine(self, args):
        """Handle sine function."""
        return f"""<mrow>
\t<mi>sin</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def cosine(self, args):
        """Handle cosine function."""
        return f"""<mrow>
\t<mi>cos</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def tangent(self, args):
        """Handle tangent function."""
        return f"""<mrow>
\t<mi>tan</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def secant(self, args):
        """Handle secant function."""
        return f"""<mrow>
\t<mi>sec</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def cosecant(self, args):
        """Handle cosecant function."""
        return f"""<mrow>
\t<mi>csc</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def cotangent(self, args):
        """Handle cotangent function."""
        return f"""<mrow>
\t<mi>cot</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    # Inverse trigonometric functions
    def arcsine(self, args):
        """Handle arcsine function."""
        return f"""<mrow>
\t<msup><mi>sin</mi><mn>-1</mn></msup>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def arccosine(self, args):
        """Handle arccosine function."""
        return f"""<mrow>
\t<msup><mi>cos</mi><mn>-1</mn></msup>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def arctangent(self, args):
        """Handle arctangent function."""
        return f"""<mrow>
\t<msup><mi>tan</mi><mn>-1</mn></msup>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    # Hyperbolic functions
    def hyperbolic_sine(self, args):
        """Handle hyperbolic sine function."""
        return f"""<mrow>
\t<mi>sinh</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def hyperbolic_cosine(self, args):
        """Handle hyperbolic cosine function."""
        return f"""<mrow>
\t<mi>cosh</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    def hyperbolic_tangent(self, args):
        """Handle hyperbolic tangent function."""
        return f"""<mrow>
\t<mi>tanh</mi>
\t<mo>(</mo>
\t{args[0]}
\t<mo>)</mo>
</mrow>"""

    # Advanced notation
    def summation(self, args):
        """Handle summation notation."""
        var, lower, upper, expr = args
        return f"""<mrow>
\t<munderover>
\t\t<mo>&sum;</mo>
\t\t<mrow><mi>{var}</mi><mo>=</mo>{lower}</mrow>
\t\t{upper}
\t</munderover>
\t{expr}
</mrow>"""

    def product(self, args):
        """Handle product notation."""
        var, lower, upper, expr = args
        return f"""<mrow>
\t<munderover>
\t\t<mo>&prod;</mo>
\t\t<mrow><mi>{var}</mi><mo>=</mo>{lower}</mrow>
\t\t{upper}
\t</munderover>
\t{expr}
</mrow>"""

    def integral(self, args):
        """Handle integral notation."""
        expr, var = args
        return f"""<mrow>
\t<mo>&int;</mo>
\t{expr}
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var}</mi>
</mrow>"""

    def limit(self, args):
        """Handle limit notation."""
        expr, var, approach = args
        return f"""<mrow>
\t<munder>
\t\t<mi>lim</mi>
\t\t<mrow><mi>{var}</mi><mo>&rarr;</mo>{approach}</mrow>
\t</munder>
\t{expr}
</mrow>"""

    # Calculus and derivatives
    def derivative(self, args):
        """Handle derivative notation d/dx(f)."""
        var, expr = args
        return f"""<mrow>
\t<mfrac>
\t\t<mi>d</mi>
\t\t<mrow><mi>d</mi><mi>{var}</mi></mrow>
\t</mfrac>
\t<mo>(</mo>
\t{expr}
\t<mo>)</mo>
</mrow>"""

    def partial_derivative(self, args):
        """Handle partial derivative notation ∂/∂x(f)."""
        var, expr = args
        return f"""<mrow>
\t<mfrac>
\t\t<mo>&part;</mo>
\t\t<mrow><mo>&part;</mo><mi>{var}</mi></mrow>
\t</mfrac>
\t<mo>(</mo>
\t{expr}
\t<mo>)</mo>
</mrow>"""

    def second_derivative(self, args):
        """Handle second derivative notation d²/dx²(f)."""
        var, expr = args
        return f"""<mrow>
\t<mfrac>
\t\t<msup><mi>d</mi><mn>2</mn></msup>
\t\t<mrow><mi>d</mi><msup><mi>{var}</mi><mn>2</mn></msup></mrow>
\t</mfrac>
\t<mo>(</mo>
\t{expr}
\t<mo>)</mo>
</mrow>"""

    def second_partial_derivative(self, args):
        """Handle second partial derivative notation ∂²/∂x²(f)."""
        var, expr = args
        return f"""<mrow>
\t<mfrac>
\t\t<msup><mo>&part;</mo><mn>2</mn></msup>
\t\t<mrow><mo>&part;</mo><msup><mi>{var}</mi><mn>2</mn></msup></mrow>
\t</mfrac>
\t<mo>(</mo>
\t{expr}
\t<mo>)</mo>
</mrow>"""

    def indefinite_integral(self, args):
        """Handle indefinite integral ∫f dx."""
        expr, var = args
        return f"""<mrow>
\t<mo>&int;</mo>
\t{expr}
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var}</mi>
</mrow>"""

    def definite_integral(self, args):
        """Handle definite integral ∫f dx from a to b."""
        expr, var, lower, upper = args
        return f"""<mrow>
\t<msubsup>
\t\t<mo>&int;</mo>
\t\t{lower}
\t\t{upper}
\t</msubsup>
\t{expr}
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var}</mi>
</mrow>"""

    def double_integral(self, args):
        """Handle double integral ∬f dx dy."""
        expr, var1, var2 = args
        return f"""<mrow>
\t<mo>&iint;</mo>
\t{expr}
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var1}</mi>
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var2}</mi>
</mrow>"""

    def triple_integral(self, args):
        """Handle triple integral ∭f dx dy dz."""
        expr, var1, var2, var3 = args
        return f"""<mrow>
\t<mo>&iiint;</mo>
\t{expr}
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var1}</mi>
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var2}</mi>
\t<mspace width="0.2em"/>
\t<mi>d</mi><mi>{var3}</mi>
</mrow>"""

    # Vector operations
    def vector(self, args):
        """Handle vector notation vec(v)."""
        var = args[0]
        return f"""<mover>
\t<mi>{var}</mi>
\t<mo>&rarr;</mo>
</mover>"""

    def dot_product(self, args):
        """Handle dot product a·b."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&middot;</mo>
\t{b}
</mrow>"""

    def cross_product(self, args):
        """Handle cross product a×b."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&times;</mo>
\t{b}
</mrow>"""

    def gradient(self, args):
        """Handle gradient ∇f."""
        expr = args[0]
        return f"""<mrow>
\t<mo>&nabla;</mo>
\t{expr}
</mrow>"""

    def divergence(self, args):
        """Handle divergence ∇·f."""
        expr = args[0]
        return f"""<mrow>
\t<mo>&nabla;</mo>
\t<mo>&middot;</mo>
\t{expr}
</mrow>"""

    def curl(self, args):
        """Handle curl ∇×f."""
        expr = args[0]
        return f"""<mrow>
\t<mo>&nabla;</mo>
\t<mo>&times;</mo>
\t{expr}
</mrow>"""

    # Number theory and logic
    def greatest_common_divisor(self, args):
        """Handle greatest common divisor gcd(a,b)."""
        a, b = args
        return f"""<mrow>
\t<mi>gcd</mi>
\t<mo>(</mo>
\t{a}
\t<mo>,</mo>
\t{b}
\t<mo>)</mo>
</mrow>"""

    def least_common_multiple(self, args):
        """Handle least common multiple lcm(a,b)."""
        a, b = args
        return f"""<mrow>
\t<mi>lcm</mi>
\t<mo>(</mo>
\t{a}
\t<mo>,</mo>
\t{b}
\t<mo>)</mo>
</mrow>"""

    def universal_quantifier(self, args):
        """Handle universal quantifier ∀x∈S:P(x)."""
        var, set_expr, predicate = args
        return f"""<mrow>
\t<mo>&forall;</mo>
\t<mi>{var}</mi>
\t<mo>&in;</mo>
\t{set_expr}
\t<mo>:</mo>
\t{predicate}
</mrow>"""

    def existential_quantifier(self, args):
        """Handle existential quantifier ∃x∈S:P(x)."""
        var, set_expr, predicate = args
        return f"""<mrow>
\t<mo>&exist;</mo>
\t<mi>{var}</mi>
\t<mo>&in;</mo>
\t{set_expr}
\t<mo>:</mo>
\t{predicate}
</mrow>"""

    def logical_and(self, args):
        """Handle logical AND ∧."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&and;</mo>
\t{b}
</mrow>"""

    def logical_or(self, args):
        """Handle logical OR ∨."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&or;</mo>
\t{b}
</mrow>"""

    def logical_not(self, args):
        """Handle logical NOT ¬."""
        expr = args[0]
        return f"""<mrow>
\t<mo>&not;</mo>
\t{expr}
</mrow>"""

    def logical_implies(self, args):
        """Handle logical implication ⇒."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&rArr;</mo>
\t{b}
</mrow>"""

    def logical_iff(self, args):
        """Handle logical if and only if ⇔."""
        a, b = args
        return f"""<mrow>
\t{a}
\t<mo>&hArr;</mo>
\t{b}
</mrow>"""

    # Matrix and set notation
    def matrix(self, args):
        """Handle matrix notation."""
        return f"""<mrow>
\t<mo>[</mo>
\t<mtable>
\t{args[0]}
\t</mtable>
\t<mo>]</mo>
</mrow>"""

    def matrix_rows(self, args):
        """Handle matrix rows."""
        rows = ""
        for row in args:
            rows += f"""\t<mtr>
{row}
\t</mtr>
"""
        return rows

    def matrix_row(self, args):
        """Handle matrix row elements."""
        cells = ""
        for cell in args:
            cells += f"""\t\t<mtd>
\t\t{cell}
\t\t</mtd>
"""
        return cells

    def set_notation(self, args):
        """Handle set notation."""
        return f"""<mrow>
\t<mo>{{</mo>
\t{args[0]}
\t<mo>}}</mo>
</mrow>"""

    # Function calls
    def function_call(self, args):
        """Handle general function calls."""
        func_name = args[0]
        arguments = args[1] if len(args) > 1 else ""
        return f"""<mrow>
\t<mi>{func_name}</mi>
\t<mo>(</mo>
\t{arguments}
\t<mo>)</mo>
</mrow>"""

    def function_call_no_args(self, args):
        """Handle function calls with no arguments."""
        func_name = args[0]
        return f"""<mrow>
\t<mi>{func_name}</mi>
\t<mo>(</mo>
\t<mo>)</mo>
</mrow>"""

    def expression_list(self, args):
        """Handle comma-separated expression lists."""
        if len(args) == 1:
            return args[0]
        else:
            result = args[0]
            for arg in args[1:]:
                result += f"""<mo>,</mo>
\t{arg}"""
            return result

    def start(self, args):
        """Handle the root of the parse tree."""
        return f"""<math xmlns='http://www.w3.org/1998/Math/MathML'>
\t{args[0]}
</math>"""


# Backward compatibility alias
MathMLTransformer = EnhancedMathMLTransformer