# Quick Start Guide

Get up and running with MathML Parser in minutes! This guide covers the most common use cases and provides practical examples.

## Basic Usage

### Parsing Mathematical Expressions

```python
from mathml_parser import MathMLParser

# Create a parser instance
parser = MathMLParser()

# Parse a simple expression
expression = "x^2 + 2*x + 1"
result = parser.parse(expression)

print(f"Original: {expression}")
print(f"Parsed: {result}")
```

### Multi-Format Output

Convert expressions to different formats:

```python
from mathml_parser import render_expression

expression = "x^2 + sqrt(y)"

# Convert to LaTeX
latex = render_expression(expression, 'latex')
print(f"LaTeX: {latex}")
# Output: $x^{2} + \sqrt{y}$

# Convert to ASCII Math
ascii_math = render_expression(expression, 'ascii')
print(f"ASCII: {ascii_math}")
# Output: x^2 + sqrt(y)

# Convert to HTML
html = render_expression(expression, 'html')
print(f"HTML: {html}")
# Output: <div class="math-expression">x<span class="math-superscript">2</span> + √y</div>
```

### Getting All Formats

```python
from mathml_parser import render_all_formats

expression = "∫(x^2, 0, 1)"
all_formats = render_all_formats(expression)

for format_name, output in all_formats.items():
    print(f"{format_name:10}: {output}")
```

## Common Mathematical Expressions

### Basic Arithmetic

```python
from mathml_parser import MathMLParser

parser = MathMLParser()

# Simple arithmetic
expressions = [
    "2 + 3",
    "x - y",
    "a * b",
    "p / q",
    "2^3",
    "sqrt(16)"
]

for expr in expressions:
    result = parser.parse(expr)
    print(f"{expr:12} → {result}")
```

### Algebraic Expressions

```python
# Polynomials
polynomial = "3*x^2 + 2*x - 5"
factored = "(x + 1)(x - 2)"

# Rational expressions
rational = "(x^2 - 1)/(x + 1)"

# Radical expressions
radical = "sqrt(x^2 + y^2)"

for expr in [polynomial, factored, rational, radical]:
    latex_output = render_expression(expr, 'latex')
    print(f"{expr:20} → {latex_output}")
```

### Trigonometric Functions

```python
# Basic trig functions
trig_expressions = [
    "sin(x)",
    "cos(π/4)",
    "tan(45°)",
    "arcsin(1/2)",
    "sec(θ)",
    "sin^2(x) + cos^2(x)"
]

for expr in trig_expressions:
    latex = render_expression(expr, 'latex')
    print(f"{expr:20} → {latex}")
```

### Calculus Notation

```python
# Derivatives
derivatives = [
    "d/dx(x^2)",
    "∂f/∂x",
    "f'(x)",
    "d²y/dx²"
]

# Integrals
integrals = [
    "∫x dx",
    "∫(x^2, 0, 1)",
    "∬f(x,y) dA",
    "∮F·dr"
]

# Limits
limits = [
    "lim(x→0) sin(x)/x",
    "lim(n→∞) (1+1/n)^n"
]

all_calc = derivatives + integrals + limits
for expr in all_calc:
    latex = render_expression(expr, 'latex')
    print(f"{expr:20} → {latex}")
```

### Linear Algebra

```python
# Matrices
matrix_expressions = [
    "[1,2;3,4]",  # 2x2 matrix
    "[a,b,c]",    # Row vector
    "det(A)",     # Determinant
    "A^T",        # Transpose
    "A^(-1)"      # Inverse
]

# Vectors
vector_expressions = [
    "vec(v)",
    "||v||",
    "v · w",
    "v × w"
]

for expr in matrix_expressions + vector_expressions:
    latex = render_expression(expr, 'latex')
    print(f"{expr:12} → {latex}")
```

## Advanced Features

### Expression Optimization

```python
from mathml_parser.optimizers import AlgebraicOptimizer

# Create optimizer
optimizer = AlgebraicOptimizer()

# Simplify expressions
expressions = [
    "x + x",           # → 2x
    "x * 1",           # → x
    "0 + x",           # → x
    "(x + 1)^2",       # → x² + 2x + 1
    "sin^2(x) + cos^2(x)"  # → 1
]

for expr in expressions:
    simplified = optimizer.simplify(expr)
    print(f"{expr:20} → {simplified}")
```

### Custom Formatting

```python
from mathml_parser.formatters import LaTeXFormatter

# Create custom formatter
formatter = LaTeXFormatter()

# Configure options
formatter.configure({
    'use_amsmath': True,
    'decimal_places': 3,
    'fraction_style': 'inline'  # or 'display'
})

expression = "1/2 + 3/4"
custom_output = formatter.format_expression(expression)
print(f"Custom LaTeX: {custom_output}")
```

### Batch Processing

```python
from mathml_parser import MathMLParser

parser = MathMLParser()

# Process multiple expressions
expressions = [
    "x^2 + 1",
    "sin(π/2)",
    "∫(e^x, 0, 1)",
    "lim(x→0) (sin(x)/x)"
]

# Batch parse
results = parser.parse_batch(expressions)

for original, parsed in zip(expressions, results):
    print(f"{original:20} → {parsed}")
```

### Error Handling

```python
from mathml_parser import MathMLParser, MathMLParseError

parser = MathMLParser()

expressions = [
    "x^2 + 1",      # Valid
    "x^^ + 1",      # Invalid: double exponent
    "sin(",         # Invalid: unclosed parenthesis
    "2 + + 3"       # Invalid: double operator
]

for expr in expressions:
    try:
        result = parser.parse(expr)
        print(f"✓ {expr:15} → {result}")
    except MathMLParseError as e:
        print(f"✗ {expr:15} → Error: {e}")
```

## Interactive Usage

### IPython/Jupyter Integration

```python
# In Jupyter notebook
from mathml_parser import MathMLParser
from IPython.display import display, Math, HTML

parser = MathMLParser()

def show_math(expression):
    """Display expression in multiple formats"""
    latex = render_expression(expression, 'latex')
    html = render_expression(expression, 'html')
    
    print(f"Input: {expression}")
    display(Math(latex.strip('$')))  # Remove $ delimiters for Math()
    display(HTML(html))

# Usage
show_math("∫(x^2 + 1, 0, π)")
```

### Command Line Interface

```bash
# Parse single expression
mathml-parse "x^2 + 1" --format latex

# Convert file of expressions
mathml-parse expressions.txt --format html --output results.html

# Interactive mode
mathml-parse --interactive
```

```python
# Programmatic CLI access
from mathml_parser.cli import main
import sys

# Simulate command line arguments
sys.argv = ['mathml-parse', 'x^2 + 1', '--format', 'latex']
main()
```

## Configuration

### Parser Settings

```python
from mathml_parser import MathMLParser

# Configure parser with custom settings
parser = MathMLParser(
    strict_mode=False,          # Allow flexible parsing
    auto_simplify=True,         # Automatic simplification
    preserve_formatting=True,   # Keep original spacing
    timeout=30,                 # Parse timeout in seconds
    cache_size=1000            # Expression cache size
)

# Or configure after creation
parser.configure({
    'decimal_notation': 'american',  # vs 'european'
    'angle_units': 'radians',        # vs 'degrees'
    'matrix_style': 'brackets'       # vs 'parentheses'
})
```

### Output Formatting

```python
from mathml_parser import render_expression

# Configure global formatting options
from mathml_parser.config import set_global_config

set_global_config({
    'latex_packages': ['amsmath', 'amssymb'],
    'html_css_class': 'my-math',
    'svg_font_family': 'Times New Roman',
    'ascii_unicode': False  # Use only ASCII characters
})

# Now all rendering uses these settings
result = render_expression("α + β", 'ascii')
print(result)  # Output: alpha + beta (not α + β)
```

## Performance Tips

### Caching

```python
from mathml_parser import MathMLParser

# Enable caching for repeated expressions
parser = MathMLParser(cache_size=10000)

# Same expression parsed multiple times uses cache
for i in range(1000):
    result = parser.parse("x^2 + 1")  # Only parsed once
```

### Parallel Processing

```python
from mathml_parser import MathMLParser
import concurrent.futures

parser = MathMLParser()

expressions = ["x^2 + 1", "sin(x)", "∫(e^x, 0, 1)"] * 100

# Process in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(parser.parse, expr) for expr in expressions]
    results = [future.result() for future in futures]
```

### Memory Management

```python
# For large batches, use streaming
from mathml_parser.streaming import StreamingParser

parser = StreamingParser()

# Process large file without loading everything into memory
with open('large_expressions.txt') as f:
    for line_num, result in parser.parse_stream(f):
        print(f"Line {line_num}: {result}")
        
        # Optional: limit memory usage
        if line_num % 1000 == 0:
            parser.clear_cache()
```

## Next Steps

Now that you've learned the basics:

1. **[Examples](examples.md)** - More complex real-world examples
2. **[User Guide](user_guide/index.md)** - In-depth feature documentation
3. **[API Reference](api/index.md)** - Complete API documentation
4. **[Advanced Topics](advanced/index.md)** - Performance optimization and extensions

## Common Patterns

### Web Application Integration

```python
from flask import Flask, request, jsonify
from mathml_parser import render_expression

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_expression():
    data = request.json
    expression = data.get('expression')
    format_type = data.get('format', 'latex')
    
    try:
        result = render_expression(expression, format_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
```

### File Processing

```python
from mathml_parser import MathMLParser
import json

parser = MathMLParser()

def process_math_file(input_file, output_file, format_type='latex'):
    """Convert mathematical expressions from file"""
    with open(input_file, 'r') as f:
        expressions = f.readlines()
    
    results = []
    for i, expr in enumerate(expressions):
        expr = expr.strip()
        if expr:
            try:
                result = render_expression(expr, format_type)
                results.append({
                    'line': i + 1,
                    'input': expr,
                    'output': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'line': i + 1,
                    'input': expr,
                    'error': str(e),
                    'success': False
                })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# Usage
process_math_file('expressions.txt', 'results.json', 'latex')
```

### Testing Mathematical Equivalence

```python
from mathml_parser import MathMLParser
from mathml_parser.comparison import expressions_equivalent

parser = MathMLParser()

# Test if expressions are mathematically equivalent
expr1 = "x^2 - 1"
expr2 = "(x-1)(x+1)"

if expressions_equivalent(expr1, expr2):
    print("Expressions are equivalent!")
else:
    print("Expressions are different.")

# Simplify and compare
simplified1 = parser.simplify(expr1)
simplified2 = parser.simplify(expr2)
print(f"{expr1} → {simplified1}")
print(f"{expr2} → {simplified2}")
```