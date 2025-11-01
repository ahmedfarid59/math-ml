# Examples and Use Cases

This section provides comprehensive examples demonstrating MathML Parser's capabilities across various domains and applications.

## Table of Contents

- [Basic Mathematics](#basic-mathematics)
- [Advanced Calculus](#advanced-calculus)
- [Linear Algebra](#linear-algebra)
- [Statistics and Probability](#statistics-and-probability)
- [Physics and Engineering](#physics-and-engineering)
- [Web Applications](#web-applications)
- [Educational Tools](#educational-tools)
- [Document Processing](#document-processing)
- [Research and Publishing](#research-and-publishing)

## Basic Mathematics

### Arithmetic and Algebra

```python
from mathml_parser import MathMLParser, render_expression

parser = MathMLParser()

# Basic arithmetic
arithmetic_examples = [
    "2 + 3 * 4",
    "(5 - 2)^2",
    "sqrt(16) + 3",
    "abs(-5) + 2"
]

print("=== Basic Arithmetic ===")
for expr in arithmetic_examples:
    latex = render_expression(expr, 'latex')
    ascii_math = render_expression(expr, 'ascii')
    print(f"Expression: {expr}")
    print(f"LaTeX:      {latex}")
    print(f"ASCII:      {ascii_math}")
    print()
```

### Polynomial Operations

```python
# Polynomial expressions
polynomials = [
    "x^3 + 2*x^2 - x + 1",
    "(x + 1)^2",
    "(a + b)(a - b)",
    "x^2 + y^2 + 2*x*y"
]

print("=== Polynomials ===")
for poly in polynomials:
    # Convert to different formats
    latex = render_expression(poly, 'latex')
    html = render_expression(poly, 'html')
    
    print(f"Polynomial: {poly}")
    print(f"LaTeX:      {latex}")
    print(f"HTML:       {html}")
    print()
```

### Rational Functions

```python
# Rational expressions
rational_functions = [
    "(x^2 - 1)/(x + 1)",
    "1/(1 + x^2)",
    "(2*x + 3)/(x^2 + x - 2)",
    "a/b + c/d"
]

print("=== Rational Functions ===")
for expr in rational_functions:
    formats = render_all_formats(expr)
    print(f"Expression: {expr}")
    for fmt_name, result in formats.items():
        print(f"  {fmt_name:8}: {result}")
    print()
```

## Advanced Calculus

### Derivatives

```python
# Derivative notation examples
derivatives = [
    "d/dx(x^2)",
    "∂f/∂x",
    "d²y/dx²",
    "f'(x)",
    "∂²u/∂x∂y"
]

print("=== Derivatives ===")
for deriv in derivatives:
    latex = render_expression(deriv, 'latex')
    print(f"Derivative: {deriv:15} → {latex}")
```

### Integrals

```python
# Integral examples
integrals = [
    "∫x dx",
    "∫(x^2 + 1, 0, 1)",
    "∫∫f(x,y) dx dy",
    "∮F·dr",
    "∫(sin(x), 0, π)"
]

print("=== Integrals ===")
for integral in integrals:
    latex = render_expression(integral, 'latex')
    ascii_math = render_expression(integral, 'ascii')
    print(f"Integral: {integral}")
    print(f"  LaTeX: {latex}")
    print(f"  ASCII: {ascii_math}")
    print()
```

### Limits and Series

```python
# Limits and series
limits_series = [
    "lim(x→0) sin(x)/x",
    "lim(n→∞) (1+1/n)^n",
    "∑(i=1 to n) i^2",
    "∑(n=0 to ∞) x^n/n!",
    "∏(i=1 to n) i"
]

print("=== Limits and Series ===")
for expr in limits_series:
    latex = render_expression(expr, 'latex')
    print(f"Expression: {expr}")
    print(f"LaTeX:      {latex}")
    print()
```

## Linear Algebra

### Matrices and Vectors

```python
# Matrix operations
matrices = [
    "[1,2;3,4]",              # 2x2 matrix
    "[a,b,c;d,e,f;g,h,i]",    # 3x3 matrix
    "det(A)",                 # Determinant
    "A^T",                    # Transpose
    "A^(-1)",                 # Inverse
    "trace(A)"                # Trace
]

print("=== Matrices ===")
for matrix in matrices:
    latex = render_expression(matrix, 'latex')
    html = render_expression(matrix, 'html')
    print(f"Matrix: {matrix}")
    print(f"LaTeX:  {latex}")
    print(f"HTML:   {html}")
    print()
```

### Vector Operations

```python
# Vector expressions
vectors = [
    "vec(v)",
    "||v||",
    "v · w",
    "v × w",
    "∇f",
    "div(F)",
    "curl(F)"
]

print("=== Vectors ===")
for vector in vectors:
    all_formats = render_all_formats(vector)
    print(f"Vector: {vector}")
    for fmt, result in all_formats.items():
        print(f"  {fmt:8}: {result}")
    print()
```

### Linear Systems

```python
# Linear system representation
system_example = """
System of equations example:
x + 2y = 5
3x - y = 1
"""

# Individual equations
equations = [
    "x + 2*y = 5",
    "3*x - y = 1",
    "A*x = b"
]

print("=== Linear Systems ===")
print(system_example)
for eq in equations:
    latex = render_expression(eq, 'latex')
    print(f"Equation: {eq:12} → {latex}")
```

## Statistics and Probability

### Statistical Notation

```python
# Statistical expressions
statistics = [
    "μ = E[X]",               # Mean
    "σ² = Var[X]",            # Variance
    "P(A ∩ B)",               # Intersection probability
    "P(A | B)",               # Conditional probability
    "∑(i=1 to n) x_i / n",    # Sample mean
    "√(∑(x_i - μ)² / n)"      # Standard deviation
]

print("=== Statistics ===")
for stat in statistics:
    latex = render_expression(stat, 'latex')
    ascii_math = render_expression(stat, 'ascii')
    print(f"Statistic: {stat}")
    print(f"  LaTeX: {latex}")
    print(f"  ASCII: {ascii_math}")
    print()
```

### Probability Distributions

```python
# Common probability distributions
distributions = [
    "f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))",  # Normal distribution
    "P(X = k) = (n choose k) * p^k * (1-p)^(n-k)",  # Binomial
    "f(x) = λe^(-λx)",                              # Exponential
    "Γ(n) = ∫(t^(n-1) * e^(-t), 0, ∞)"            # Gamma function
]

print("=== Probability Distributions ===")
for dist in distributions:
    latex = render_expression(dist, 'latex')
    print(f"Distribution: {dist}")
    print(f"LaTeX:        {latex}")
    print()
```

## Physics and Engineering

### Classical Mechanics

```python
# Physics formulas
physics_formulas = [
    "F = m*a",                    # Newton's second law
    "E = (1/2)*m*v²",            # Kinetic energy
    "F = G*m₁*m₂/r²",            # Gravitational force
    "ω = 2πf",                   # Angular frequency
    "τ = r × F",                 # Torque
    "L = r × p"                  # Angular momentum
]

print("=== Classical Mechanics ===")
for formula in physics_formulas:
    latex = render_expression(formula, 'latex')
    print(f"Formula: {formula:20} → {latex}")
```

### Electromagnetic Theory

```python
# Electromagnetic equations
em_equations = [
    "∇ · E = ρ/ε₀",             # Gauss's law
    "∇ × B = μ₀J + μ₀ε₀∂E/∂t", # Ampère's law
    "∇ × E = -∂B/∂t",           # Faraday's law
    "∇ · B = 0",                # No magnetic monopoles
    "F = q(E + v × B)"          # Lorentz force
]

print("=== Electromagnetic Theory ===")
for eq in em_equations:
    latex = render_expression(eq, 'latex')
    print(f"Equation: {eq}")
    print(f"LaTeX:    {latex}")
    print()
```

### Quantum Mechanics

```python
# Quantum mechanics expressions
quantum = [
    "Ĥψ = Eψ",                  # Schrödinger equation
    "[x̂, p̂] = iℏ",              # Canonical commutation relation
    "⟨ψ|Ô|ψ⟩",                  # Expectation value
    "e^(iĤt/ℏ)",                # Time evolution operator
    "ψ(x,t) = ∑c_n φ_n(x)e^(-iE_n t/ℏ)"  # Wave function expansion
]

print("=== Quantum Mechanics ===")
for expr in quantum:
    latex = render_expression(expr, 'latex')
    print(f"Expression: {expr}")
    print(f"LaTeX:      {latex}")
    print()
```

## Web Applications

### Real-time Math Renderer

```python
from flask import Flask, render_template, request, jsonify
from mathml_parser import render_expression, render_all_formats

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('math_converter.html')

@app.route('/api/convert', methods=['POST'])
def convert_math():
    data = request.json
    expression = data.get('expression', '')
    target_format = data.get('format', 'latex')
    
    try:
        if target_format == 'all':
            result = render_all_formats(expression)
        else:
            result = render_expression(expression, target_format)
        
        return jsonify({
            'success': True,
            'result': result,
            'expression': expression
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'expression': expression
        })

@app.route('/api/preview', methods=['POST'])
def preview_math():
    """Generate live preview in multiple formats"""
    data = request.json
    expression = data.get('expression', '')
    
    try:
        all_formats = render_all_formats(expression)
        
        # Add HTML preview with CSS
        from mathml_parser.formatters import HTMLFormatter
        html_formatter = HTMLFormatter()
        styled_html = html_formatter.get_css_styles() + all_formats['html']
        all_formats['html_styled'] = styled_html
        
        return jsonify({
            'success': True,
            'formats': all_formats,
            'expression': expression
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
```

### Math Expression Validator

```python
from mathml_parser import MathMLParser, MathMLParseError

class MathValidator:
    def __init__(self):
        self.parser = MathMLParser()
    
    def validate_expression(self, expression):
        """Validate mathematical expression and return detailed feedback"""
        result = {
            'valid': False,
            'expression': expression,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            # Attempt to parse
            parsed = self.parser.parse(expression)
            result['valid'] = True
            result['parsed'] = str(parsed)
            
            # Check for common issues
            self._check_warnings(expression, result)
            
        except MathMLParseError as e:
            result['errors'].append(str(e))
            result['suggestions'] = self._suggest_fixes(expression, e)
        
        return result
    
    def _check_warnings(self, expression, result):
        """Check for potential issues that aren't errors"""
        if '**' in expression:
            result['warnings'].append("Consider using '^' for exponentiation instead of '**'")
        
        if expression.count('(') != expression.count(')'):
            result['warnings'].append("Unmatched parentheses")
        
        # Check for common mathematical constants
        if 'pi' in expression.lower() and 'π' not in expression:
            result['suggestions'].append("Consider using π instead of 'pi'")
    
    def _suggest_fixes(self, expression, error):
        """Suggest potential fixes for common errors"""
        suggestions = []
        
        if "unexpected" in str(error).lower():
            suggestions.append("Check for typos in function names or operators")
        
        if "parenthes" in str(error).lower():
            suggestions.append("Check that all parentheses are properly matched")
        
        return suggestions

# Usage example
validator = MathValidator()

test_expressions = [
    "x^2 + 1",           # Valid
    "sin(x",             # Invalid: missing parenthesis
    "x ** 2",            # Valid but warning
    "cos(pi/2)",         # Valid with suggestion
    "x + + 1"            # Invalid: double operator
]

print("=== Expression Validation ===")
for expr in test_expressions:
    result = validator.validate_expression(expr)
    print(f"Expression: {expr}")
    print(f"Valid: {result['valid']}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    if result['suggestions']:
        print(f"Suggestions: {result['suggestions']}")
    print()
```

## Educational Tools

### Step-by-Step Solutions

```python
from mathml_parser import MathMLParser
from mathml_parser.educational import StepByStepper

class MathTutor:
    def __init__(self):
        self.parser = MathMLParser()
        self.stepper = StepByStepper()
    
    def solve_quadratic(self, a, b, c):
        """Demonstrate solving quadratic equation step by step"""
        print(f"Solving {a}x² + {b}x + {c} = 0")
        print("=" * 40)
        
        # Step 1: Show the equation
        equation = f"{a}*x^2 + {b}*x + {c} = 0"
        latex_eq = render_expression(equation, 'latex')
        print(f"Step 1: Start with the equation")
        print(f"        {latex_eq}")
        print()
        
        # Step 2: Show quadratic formula
        formula = "x = (-b ± sqrt(b² - 4ac)) / (2a)"
        latex_formula = render_expression(formula, 'latex')
        print(f"Step 2: Apply the quadratic formula")
        print(f"        {latex_formula}")
        print()
        
        # Step 3: Substitute values
        substituted = f"x = (-({b}) ± sqrt(({b})² - 4({a})({c}))) / (2({a}))"
        latex_sub = render_expression(substituted, 'latex')
        print(f"Step 3: Substitute a={a}, b={b}, c={c}")
        print(f"        {latex_sub}")
        print()
        
        # Step 4: Calculate discriminant
        discriminant = b**2 - 4*a*c
        disc_calc = f"x = ({-b} ± sqrt({discriminant})) / {2*a}"
        latex_disc = render_expression(disc_calc, 'latex')
        print(f"Step 4: Calculate the discriminant")
        print(f"        Δ = {b}² - 4({a})({c}) = {discriminant}")
        print(f"        {latex_disc}")
        print()
        
        # Step 5: Final solutions
        if discriminant >= 0:
            import math
            sqrt_disc = math.sqrt(discriminant)
            x1 = (-b + sqrt_disc) / (2*a)
            x2 = (-b - sqrt_disc) / (2*a)
            
            solution1 = f"x₁ = ({-b} + {sqrt_disc:.3f}) / {2*a} = {x1:.3f}"
            solution2 = f"x₂ = ({-b} - {sqrt_disc:.3f}) / {2*a} = {x2:.3f}"
            
            print(f"Step 5: Calculate the solutions")
            print(f"        {solution1}")
            print(f"        {solution2}")
        else:
            print(f"Step 5: Complex solutions (discriminant < 0)")
            print(f"        Solutions involve imaginary numbers")

# Example usage
tutor = MathTutor()
tutor.solve_quadratic(1, -5, 6)  # x² - 5x + 6 = 0
```

### Mathematical Concept Visualization

```python
def create_function_table(expression, x_values):
    """Create a table of function values"""
    from mathml_parser import MathMLParser
    
    parser = MathMLParser()
    
    print(f"Function: f(x) = {expression}")
    print(f"LaTeX:    {render_expression(expression, 'latex')}")
    print()
    print("x     | f(x)")
    print("------|--------")
    
    for x in x_values:
        # This is a simplified example - real implementation would need
        # expression evaluation capabilities
        try:
            # Substitute x value (simplified - real implementation more complex)
            substituted = expression.replace('x', str(x))
            result = eval(substituted)  # WARNING: Only for demo - use safe evaluation
            print(f"{x:5} | {result:8.3f}")
        except:
            print(f"{x:5} | undefined")

# Example
create_function_table("x**2 - 4*x + 3", range(-2, 6))
```

## Document Processing

### LaTeX Document Integration

```python
def create_latex_document(expressions, title="Mathematical Expressions"):
    """Generate a complete LaTeX document with mathematical expressions"""
    
    # Document header
    latex_doc = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

\title{""" + title + r"""}
\author{Generated by MathML Parser}
\date{\today}

\begin{document}
\maketitle

\section{Mathematical Expressions}

"""
    
    # Add expressions
    for i, expr in enumerate(expressions, 1):
        latex_expr = render_expression(expr, 'latex')
        latex_doc += f"""
\\subsection{{Expression {i}}}

Original notation:
\\begin{{verbatim}}
{expr}
\\end{{verbatim}}

Rendered LaTeX:
\\[{latex_expr.strip('$')}\\]

"""
    
    # Document footer
    latex_doc += r"""
\end{document}
"""
    
    return latex_doc

# Example usage
expressions = [
    "∫(x^2 + 1, 0, π)",
    "lim(x→0) sin(x)/x",
    "∑(n=1 to ∞) 1/n²",
    "[1,2;3,4] * [x;y]"
]

latex_document = create_latex_document(expressions, "Calculus Examples")
print(latex_document)

# Save to file
with open('math_expressions.tex', 'w') as f:
    f.write(latex_document)
```

### Markdown with Math Integration

```python
def create_markdown_with_math(expressions, descriptions):
    """Create Markdown document with mathematical expressions"""
    
    markdown = "# Mathematical Expressions\n\n"
    
    for expr, desc in zip(expressions, descriptions):
        # Get different formats
        latex = render_expression(expr, 'latex')
        ascii_math = render_expression(expr, 'ascii')
        
        markdown += f"## {desc}\n\n"
        markdown += f"**Original notation:** `{expr}`\n\n"
        markdown += f"**LaTeX rendering:** {latex}\n\n"
        markdown += f"**ASCII Math:** `{ascii_math}`\n\n"
        markdown += "---\n\n"
    
    return markdown

# Example
expressions = [
    "E = mc²",
    "∇²φ = 0",
    "∫∫∫ρ dV"
]

descriptions = [
    "Einstein's Mass-Energy Equivalence",
    "Laplace's Equation",
    "Volume Integral of Density"
]

markdown_doc = create_markdown_with_math(expressions, descriptions)
print(markdown_doc)
```

## Research and Publishing

### Batch Processing for Papers

```python
import re
from mathml_parser import render_expression

def process_academic_paper(text):
    """Process academic paper text and convert inline math to LaTeX"""
    
    # Pattern to match mathematical expressions in text
    # This is a simplified pattern - real implementation would be more sophisticated
    math_pattern = r'\$([^$]+)\$'
    
    def math_replacer(match):
        expression = match.group(1)
        try:
            # Convert to proper LaTeX
            latex_output = render_expression(expression, 'latex')
            return latex_output
        except:
            # Return original if conversion fails
            return match.group(0)
    
    # Replace mathematical expressions
    processed_text = re.sub(math_pattern, math_replacer, text)
    
    return processed_text

# Example academic text
paper_text = """
The fundamental theorem states that if $f(x) = x^2 + 1$, then the derivative
$f'(x) = 2x$. This can be proven using the limit definition:
$lim(h→0) (f(x+h) - f(x))/h$.

For the integral, we have $∫(x^2 + 1, 0, 1) = 4/3$.
"""

processed = process_academic_paper(paper_text)
print("=== Processed Academic Text ===")
print(processed)
```

### Journal Submission Helper

```python
class JournalFormatter:
    """Helper class for formatting mathematical expressions for different journals"""
    
    def __init__(self):
        self.journal_styles = {
            'ieee': {
                'inline_delim': '$',
                'display_delim': '$$',
                'packages': ['amsmath', 'amssymb']
            },
            'springer': {
                'inline_delim': '$',
                'display_delim': r'\begin{equation}',
                'packages': ['amsmath', 'amssymb', 'amsthm']
            },
            'elsevier': {
                'inline_delim': '$',
                'display_delim': r'\begin{align}',
                'packages': ['amsmath']
            }
        }
    
    def format_for_journal(self, expressions, journal='ieee'):
        """Format expressions according to journal style"""
        if journal not in self.journal_styles:
            raise ValueError(f"Unsupported journal: {journal}")
        
        style = self.journal_styles[journal]
        formatted = []
        
        for expr in expressions:
            latex = render_expression(expr, 'latex')
            
            # Apply journal-specific formatting
            if journal == 'springer' and '=' in expr:
                # Use equation environment for equations
                formatted_expr = f"\\begin{{equation}}\n{latex.strip('$')}\n\\end{{equation}}"
            else:
                formatted_expr = latex
            
            formatted.append(formatted_expr)
        
        return formatted

# Example usage
formatter = JournalFormatter()

equations = [
    "E = mc²",
    "∇ × B = μ₀J",
    "ψ(x,t) = Ae^(i(kx - ωt))"
]

for journal in ['ieee', 'springer', 'elsevier']:
    print(f"=== {journal.upper()} Format ===")
    formatted = formatter.format_for_journal(equations, journal)
    for eq in formatted:
        print(eq)
    print()
```

## Performance Examples

### Large-Scale Processing

```python
import time
from mathml_parser import MathMLParser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def benchmark_parsing():
    """Benchmark parsing performance with different approaches"""
    
    parser = MathMLParser()
    
    # Generate test expressions
    test_expressions = [
        f"x^{i} + {i}*y + {i**2}" for i in range(1, 1001)
    ]
    
    # Sequential processing
    start_time = time.time()
    results_sequential = []
    for expr in test_expressions:
        result = parser.parse(expr)
        results_sequential.append(result)
    sequential_time = time.time() - start_time
    
    # Parallel processing with threads
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_parallel = list(executor.map(parser.parse, test_expressions))
    parallel_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.2f} seconds")
    print(f"Parallel processing:   {parallel_time:.2f} seconds")
    print(f"Speedup:              {sequential_time/parallel_time:.2f}x")
    print(f"Expressions per second (sequential): {len(test_expressions)/sequential_time:.0f}")
    print(f"Expressions per second (parallel):   {len(test_expressions)/parallel_time:.0f}")

# Run benchmark
benchmark_parsing()
```

### Memory-Efficient Processing

```python
def process_large_math_file(filename, output_format='latex'):
    """Process large files of mathematical expressions efficiently"""
    
    from mathml_parser import MathMLParser
    
    parser = MathMLParser(cache_size=1000)  # Limit cache size
    
    with open(filename, 'r') as input_file, \
         open(f'output_{output_format}.txt', 'w') as output_file:
        
        for line_num, line in enumerate(input_file, 1):
            expression = line.strip()
            if expression:
                try:
                    result = render_expression(expression, output_format)
                    output_file.write(f"{result}\n")
                    
                    # Clear cache periodically to manage memory
                    if line_num % 1000 == 0:
                        parser.clear_cache()
                        print(f"Processed {line_num} expressions...")
                        
                except Exception as e:
                    output_file.write(f"ERROR: {e}\n")
    
    print(f"Processing complete. Output saved to output_{output_format}.txt")

# Example usage (would need actual file)
# process_large_math_file('mathematical_expressions.txt', 'latex')
```

This comprehensive examples section demonstrates the versatility and power of MathML Parser across various domains and applications. Each example is designed to be practical and immediately usable in real-world scenarios.