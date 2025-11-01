# MathML Parser

A comprehensive mathematical expression to MathML parser with advanced features, robust error handling, and extensive mathematical notation support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version 2.0.0](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/mathml-parser/mathml-parser)

## Features

### 🧮 Comprehensive Mathematical Notation
- **Basic Arithmetic**: Addition, subtraction, multiplication, division, exponentiation, modulo
- **Advanced Functions**: Trigonometric, hyperbolic, logarithmic, exponential
- **Greek Letters**: Full support for α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω
- **Mathematical Constants**: π, e, ∞ and more
- **Comparison Operators**: =, ≠, <, >, ≤, ≥
- **Matrix Notation**: Support for matrices and vectors
- **Set Notation**: Sets, unions, intersections
- **Calculus**: Integrals, summations, products, limits
- **Subscripts & Superscripts**: Full support for complex notation

### 🛡️ Robust Error Handling
- **Input Validation**: Comprehensive validation with typo detection
- **Detailed Error Messages**: Clear error descriptions with context
- **Smart Suggestions**: Helpful suggestions for fixing errors
- **Safe Parsing**: Non-throwing parse methods for production use

### ⚡ Performance & Monitoring
- **Performance Metrics**: Parse time, complexity analysis, feature detection
- **Grammar Caching**: Optimized parsing with grammar compilation caching
- **Scalable Architecture**: Designed for high-performance applications

### 🔧 Developer-Friendly
- **Type Hints**: Full type annotation support
- **Comprehensive Tests**: Extensive test suite with 95%+ coverage
- **CLI Interface**: Command-line tool for batch processing
- **Documentation**: Detailed API documentation and examples

## Installation

### From PyPI (Recommended)
```bash
pip install mathml-parser
```

### From Source
```bash
git clone https://github.com/mathml-parser/mathml-parser.git
cd mathml-parser
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/mathml-parser/mathml-parser.git
cd mathml-parser
pip install -e ".[dev,docs]"
```

## Quick Start

### Basic Usage

```python
from mathml_parser import parse, parse_safe

# Simple parsing
mathml = parse("x^2 + 2*x + 1")
print(mathml)

# Safe parsing with error handling
result = parse_safe("sin(π/2)")
if result.success:
    print(result.mathml)
else:
    print(f"Error: {result.error.message}")
```

### Advanced Usage

```python
from mathml_parser import MathMLParser

# Create parser with custom settings
parser = MathMLParser(
    enable_validation=True,
    enable_metrics=True,
    strict_mode=False
)

# Parse with detailed results
result = parser.parse_safe("∫(x^2)dx from 0 to π")
if result.success:
    print(f"MathML: {result.mathml}")
    print(f"Parse time: {result.metrics.parse_time:.4f}s")
    print(f"Complexity: {result.metrics.complexity_score:.2f}")
    print(f"Features: {', '.join(result.metrics.features_used)}")
```

### Command Line Interface

```bash
# Parse single expression
mathml-parse "x^2 + 2*x + 1"

# Parse from file
mathml-parse --file expressions.txt --output results.xml

# Interactive mode
mathml-parse --interactive

# With validation and metrics
mathml-parse "sin(π/2)" --validate --metrics --format json
```

## Supported Mathematical Notation

### Basic Operations
- **Arithmetic**: `2 + 3`, `x - y`, `a * b`, `p / q`, `x^2`, `x % 3`
- **Grouping**: `(a + b) * c`, `[matrix]`, `{set}`

### Functions
- **Trigonometric**: `sin(x)`, `cos(x)`, `tan(x)`, `sec(x)`, `csc(x)`, `cot(x)`
- **Inverse Trig**: `arcsin(x)`, `arccos(x)`, `arctan(x)`
- **Hyperbolic**: `sinh(x)`, `cosh(x)`, `tanh(x)`
- **Logarithmic**: `log(x)`, `ln(x)`, `exp(x)`
- **Other**: `sqrt(x)`, `abs(x)`, `floor(x)`, `ceil(x)`, `round(x)`

### Greek Letters
```python
# All Greek letters supported
parse("α + β = γ")
parse("sin(θ) = cos(π/2 - θ)")
parse("Δx/Δt → dx/dt")
```

### Advanced Notation
```python
# Matrices
parse("[1, 2; 3, 4]")

# Calculus
parse("∫(x^2)dx")
parse("∑(i=1 to n) i^2")
parse("lim(x→0) sin(x)/x")

# Comparisons
parse("x ≥ 0")
parse("a ≠ b")

# Complex expressions
parse("e^(iπ) + 1 = 0")
```

## API Reference

### Main Functions

#### `parse(expression: str) -> str`
Parse mathematical expression and return MathML string.
- **Parameters**: `expression` - Mathematical expression to parse
- **Returns**: MathML string representation
- **Raises**: `MathParseError` on parsing failure

#### `parse_safe(expression: str) -> MathParseResult`
Safely parse expression with comprehensive error handling.
- **Parameters**: `expression` - Mathematical expression to parse  
- **Returns**: `MathParseResult` object with success/error information

### MathMLParser Class

#### `__init__(enable_validation=True, enable_metrics=False, strict_mode=False, cache_grammar=True)`
Initialize parser with custom configuration.

#### `parse(expression: str) -> str`
Parse expression and return MathML (throws on error).

#### `parse_safe(expression: str) -> MathParseResult`
Parse expression safely with error handling.

#### `get_metrics_summary() -> Dict[str, Any]`
Get performance metrics summary.

### Result Objects

#### `MathParseResult`
- `success: bool` - Whether parsing succeeded
- `mathml: str` - Generated MathML (if successful)
- `error: MathParseError` - Error information (if failed)
- `metrics: ParserMetrics` - Performance metrics (if enabled)

#### `MathParseError`
- `message: str` - Error description
- `error_type: str` - Type of error
- `position: int` - Error position in input
- `context: str` - Context around error
- `suggestions: List[str]` - Suggested fixes

## Examples

### Complex Mathematical Expressions

```python
from mathml_parser import parse

# Quadratic formula
mathml = parse("x = (-b ± √(b²-4ac))/(2a)")

# Euler's identity  
mathml = parse("e^(iπ) + 1 = 0")

# Gaussian integral
mathml = parse("∫(-∞ to ∞) e^(-x²) dx = √π")

# Taylor series
mathml = parse("∑(n=0 to ∞) x^n/n! = e^x")

# Matrix operations
mathml = parse("[cos(θ), -sin(θ); sin(θ), cos(θ)]")
```

### Error Handling

```python
from mathml_parser import parse_safe

result = parse_safe("2 + + 3")  # Invalid syntax
if not result.success:
    print(f"Error: {result.error.message}")
    print(f"Position: {result.error.position}")
    print(f"Suggestions: {', '.join(result.error.suggestions)}")
```

### Performance Monitoring

```python
from mathml_parser import MathMLParser

parser = MathMLParser(enable_metrics=True)

result = parser.parse_safe("sin(cos(tan(x^2)))")
if result.success and result.metrics:
    print(f"Parse time: {result.metrics.parse_time:.4f}s")
    print(f"Complexity score: {result.metrics.complexity_score:.2f}")
    print(f"Features used: {', '.join(result.metrics.features_used)}")
```

## CLI Usage

### Basic Commands

```bash
# Parse single expression
mathml-parse "x^2 + 2*x + 1"

# Parse from file (one expression per line)
mathml-parse --file expressions.txt

# Save output to file
mathml-parse "sin(x)" --output result.xml

# Interactive mode
mathml-parse --interactive
```

### Output Formats

```bash
# MathML output (default)
mathml-parse "x^2"

# JSON output with metadata
mathml-parse "x^2" --format json --pretty

# Text output with details
mathml-parse "x^2" --format text --metrics
```

### Validation and Error Handling

```bash
# Enable validation and metrics
mathml-parse "sin(π/2)" --validate --metrics

# Strict parsing mode
mathml-parse "x + y" --strict

# Disable validation for performance
mathml-parse "x + y" --no-validate
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest mathml_parser/tests/

# Run with coverage
python -m pytest mathml_parser/tests/ --cov=mathml_parser

# Run specific test file
python mathml_parser/tests/test_comprehensive.py
```

### Code Quality

```bash
# Format code
black mathml_parser/

# Sort imports
isort mathml_parser/

# Type checking
mypy mathml_parser/

# Linting
flake8 mathml_parser/
```

### Building Documentation

```bash
cd docs/
make html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 2.0.0
- Complete rewrite with enhanced architecture
- Added comprehensive mathematical notation support
- Implemented robust error handling with suggestions
- Added performance monitoring and metrics
- Created CLI interface
- Full type hint support
- Extensive test suite (95%+ coverage)

### Version 1.0.0
- Initial release with basic parsing functionality

## Support

- **Documentation**: [Read the Docs](https://mathml-parser.readthedocs.io/)
- **Issue Tracker**: [GitHub Issues](https://github.com/mathml-parser/mathml-parser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mathml-parser/mathml-parser/discussions)

## Acknowledgments

- Built with [Lark](https://github.com/lark-parser/lark) parsing toolkit
- Inspired by mathematical notation standards
- Thanks to all contributors and users