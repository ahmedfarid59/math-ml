"""
Comprehensive Mathematical Grammar for MathML Parser
===================================================

This module contains the complete Lark grammar definition for parsing
mathematical expressions with extensive notation support.

Features supported:
- Basic arithmetic operations (+, -, *, /, %, ^)
- Comparison operators (=, !=, <, >, <=, >=) 
- Implicit multiplication (2x, 3(x+1))
- Mathematical functions (sin, cos, log, ln, exp, etc.)
- Greek letters and mathematical constants
- Matrix notation [a,b;c,d]
- Advanced functions (floor, ceil, max, min)
- Subscripts and superscripts
- Absolute values and factorials
- Root operations and summation
"""

class MathematicalGrammar:
    """Comprehensive mathematical grammar for expression parsing."""
    
    @staticmethod
    def get_grammar():
        """
        Get the complete Lark grammar for mathematical expressions.
        
        Returns:
            str: Lark grammar definition
        """
        return r"""
        ?start: comparison -> start

        // Precedence from lowest to highest (comparison -> expression -> term -> factor -> power -> atom)
        ?comparison: expression
            | comparison "=" expression -> equal
            | comparison "!=" expression -> not_equal
            | comparison "<" expression -> less_than
            | comparison ">" expression -> greater_than
            | comparison "<=" expression -> less_equal
            | comparison ">=" expression -> greater_equal

        ?expression: term
            | expression "+" term -> add
            | expression "-" term -> sub

        ?term: factor  
            | term "*" factor -> mul
            | term "/" factor -> div
            | term "%" factor -> mod
            | NUMBER factor -> implicit_mul  // 2x, 2(x+1)
            | factor factor -> implicit_mul  // (x)(y), x y when not ambiguous

        ?factor: power
            | "+" factor -> unary_plus
            | "-" factor -> unary_minus

        ?power: atom
            | atom "^" factor -> pow  // right associative

        ?atom: NUMBER -> number
            | SUBSCRIPT_VAR -> subscript_var
            | GREEK_LETTER -> greek_letter
            | MATH_CONSTANT -> math_constant
            
            // Basic mathematical functions
            | "sqrt" "(" expression ")" -> square_root
            | "root" "(" expression "," expression ")" -> nth_root
            | "abs" "(" expression ")" -> abs_function
            
            // Advanced mathematical functions
            | "ln" "(" expression ")" -> natural_log
            | "log" "(" expression ")" -> logarithm
            | "exp" "(" expression ")" -> exponential
            | "floor" "(" expression ")" -> floor_func
            | "ceil" "(" expression ")" -> ceil_func
            | "round" "(" expression ")" -> round_func
            | "max" "(" expression_list ")" -> max_func
            | "min" "(" expression_list ")" -> min_func
            
            // Trigonometric functions
            | "sin" "(" expression ")" -> sine
            | "cos" "(" expression ")" -> cosine  
            | "tan" "(" expression ")" -> tangent
            | "sec" "(" expression ")" -> secant
            | "csc" "(" expression ")" -> cosecant
            | "cot" "(" expression ")" -> cotangent
            
            // Inverse trigonometric functions
            | "arcsin" "(" expression ")" -> arcsine
            | "arccos" "(" expression ")" -> arccosine
            | "arctan" "(" expression ")" -> arctangent
            
            // Hyperbolic functions
            | "sinh" "(" expression ")" -> hyperbolic_sine
            | "cosh" "(" expression ")" -> hyperbolic_cosine
            | "tanh" "(" expression ")" -> hyperbolic_tangent
            
            // Advanced notation
            | "sum" "(" VARIABLE "," expression "," expression "," expression ")" -> summation
            | "prod" "(" VARIABLE "," expression "," expression "," expression ")" -> product
            | "int" "(" expression "," VARIABLE ")" -> integral
            | "lim" "(" expression "," VARIABLE "," expression ")" -> limit
            
            // Calculus and derivatives
            | "d" "/" "d" VARIABLE "(" expression ")" -> derivative
            | "∂" "/" "∂" VARIABLE "(" expression ")" -> partial_derivative
            | "d²" "/" "d" VARIABLE "²" "(" expression ")" -> second_derivative
            | "∂²" "/" "∂" VARIABLE "²" "(" expression ")" -> second_partial_derivative
            | "∫" expression "d" VARIABLE -> indefinite_integral
            | "∫" expression "d" VARIABLE "from" expression "to" expression -> definite_integral
            | "∬" expression "d" VARIABLE "d" VARIABLE -> double_integral
            | "∭" expression "d" VARIABLE "d" VARIABLE "d" VARIABLE -> triple_integral
            
            // Vector operations
            | "vec" "(" VARIABLE ")" -> vector
            | "dot" "(" expression "," expression ")" -> dot_product
            | "cross" "(" expression "," expression ")" -> cross_product
            | "grad" "(" expression ")" -> gradient
            | "div" "(" expression ")" -> divergence
            | "curl" "(" expression ")" -> curl
            
            // Number theory and logic
            | "gcd" "(" expression "," expression ")" -> greatest_common_divisor
            | "lcm" "(" expression "," expression ")" -> least_common_multiple
            | "∀" VARIABLE "∈" expression ":" expression -> universal_quantifier
            | "∃" VARIABLE "∈" expression ":" expression -> existential_quantifier
            | expression "∧" expression -> logical_and
            | expression "∨" expression -> logical_or
            | "¬" expression -> logical_not
            | expression "⇒" expression -> logical_implies
            | expression "⇔" expression -> logical_iff
            
            // Matrix and vector notation
            | "[" matrix_rows "]" -> matrix
            | "|" expression "|" -> absolute_value
            
            // Variable and function handling
            | VARIABLE "_" atom -> subscript
            | VARIABLE "^" atom -> superscript
            | VARIABLE "(" expression_list ")" -> function_call
            | VARIABLE "(" ")" -> function_call_no_args
            | VARIABLE -> variable
            
            // Grouping and special operators
            | "(" expression ")" -> group
            | atom "!" -> factorial
            | "{" expression_list "}" -> set_notation

        // Helper rules for lists and matrices
        expression_list: expression ("," expression)*
        matrix_rows: matrix_row (";" matrix_row)*
        matrix_row: expression ("," expression)*

        // Terminal definitions
        NUMBER: /[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/  // Scientific notation support
        VARIABLE: /[a-zA-Z][a-zA-Z_0-9]*/  // Variables start with letter
        SUBSCRIPT_VAR: /[a-zA-Z][a-zA-Z_0-9]*_[a-zA-Z_0-9]+/  // Variables with subscripts
        
        // Greek letters (comprehensive list)
        GREEK_LETTER: "alpha" | "beta" | "gamma" | "delta" | "epsilon" | "zeta" | "eta" | "theta" 
                    | "iota" | "kappa" | "lambda" | "mu" | "nu" | "xi" | "omicron" | "pi" 
                    | "rho" | "sigma" | "tau" | "upsilon" | "phi" | "chi" | "psi" | "omega"
                    | "Alpha" | "Beta" | "Gamma" | "Delta" | "Epsilon" | "Zeta" | "Eta" | "Theta"
                    | "Iota" | "Kappa" | "Lambda" | "Mu" | "Nu" | "Xi" | "Omicron" | "Pi"
                    | "Rho" | "Sigma" | "Tau" | "Upsilon" | "Phi" | "Chi" | "Psi" | "Omega"
        
        // Mathematical constants
        MATH_CONSTANT: "e" | "π" | "pi" | "infinity" | "inf" | "∞"

        // Ignore whitespace
        %ignore /\s+/
        """
    
    @staticmethod
    def get_simplified_grammar():
        """
        Get a simplified grammar for basic mathematical expressions.
        
        Returns:
            str: Simplified Lark grammar definition
        """
        return r"""
        ?start: expression -> start

        ?expression: term
            | expression "+" term -> add
            | expression "-" term -> sub

        ?term: factor  
            | term "*" factor -> mul
            | term "/" factor -> div
            | NUMBER factor -> implicit_mul
            | factor factor -> implicit_mul

        ?factor: power
            | "+" factor -> unary_plus
            | "-" factor -> unary_minus

        ?power: atom
            | atom "^" factor -> pow

        ?atom: NUMBER -> number
            | VARIABLE -> variable
            | "(" expression ")" -> group
            | atom "!" -> factorial

        NUMBER: /[0-9]+(\.[0-9]+)?/
        VARIABLE: /[a-zA-Z][a-zA-Z_0-9]*/

        %ignore /\s+/
        """
    
    @staticmethod
    def validate_grammar():
        """
        Validate that the grammar can be compiled by Lark.
        
        Returns:
            bool: True if grammar is valid, False otherwise
        """
        try:
            from lark import Lark
            Lark(MathematicalGrammar.get_grammar(), parser='lalr', start='start')
            return True
        except Exception:
            return False

# For backward compatibility
grammar = MathematicalGrammar.get_grammar()

if __name__ == "__main__":
    # Test grammar validation
    if MathematicalGrammar.validate_grammar():
        print("✓ Grammar validation successful")
    else:
        print("✗ Grammar validation failed")