# MathML Parser

A comprehensive mathematical expression to MathML parser with advanced features, robust error handling, and extensive mathematical notation support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version 2.0.0](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/ahmedfarid59/math-ml)

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

### From Source (Recommended)
```bash
git clone https://github.com/ahmedfarid59/math-ml.git
cd math-ml
python -m pip install -r requirements.txt
```

### Development Installation
```bash
git clone https://github.com/ahmedfarid59/math-ml.git
cd math-ml
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # Development dependencies
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
python -m mathml_parser.cli "x^2 + 2*x + 1"

# Parse from file
python -m mathml_parser.cli --file expressions.txt --output results.xml

# Interactive mode
python -m mathml_parser.cli --interactive

# With validation and metrics
python -m mathml_parser.cli "sin(π/2)" --validate --metrics --format json
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
python -m mathml_parser.cli "x^2 + 2*x + 1"

# Parse from file (one expression per line)
python -m mathml_parser.cli --file expressions.txt

# Save output to file
python -m mathml_parser.cli "sin(x)" --output result.xml

# Interactive mode
python -m mathml_parser.cli --interactive
```

### Output Formats

```bash
# MathML output (default)
python -m mathml_parser.cli "x^2"

# JSON output with metadata
python -m mathml_parser.cli "x^2" --format json --pretty

# Text output with details
python -m mathml_parser.cli "x^2" --format text --metrics
```

### Validation and Error Handling

```bash
# Enable validation and metrics
python -m mathml_parser.cli "sin(π/2)" --validate --metrics

# Strict parsing mode
python -m mathml_parser.cli "x + y" --strict

# Disable validation for performance
python -m mathml_parser.cli "x + y" --no-validate
```

## Testing

### Test Suite Overview

The MathML Parser includes a comprehensive test suite with 95%+ code coverage, featuring:

- **Unit Tests**: Individual component testing for all modules
- **Integration Tests**: End-to-end parsing workflow validation
- **Performance Tests**: Benchmarking and optimization validation
- **Domain Tests**: Specialized testing for mathematical domains
- **Web Interface Tests**: API and WebSocket functionality testing
- **Regression Tests**: Preventing regressions in core functionality

### Running Tests

#### Basic Test Execution
```bash
# Run all tests with coverage report
python -m pytest mathml_parser/tests/ --cov=mathml_parser --cov-report=html

# Run specific test modules
python -m pytest mathml_parser/tests/test_core.py
python -m pytest mathml_parser/tests/test_domains.py
python -m pytest mathml_parser/tests/test_performance.py

# Run tests with verbose output
python -m pytest mathml_parser/tests/ -v

# Run tests and stop on first failure
python -m pytest mathml_parser/tests/ -x
```

#### Advanced Testing Options
```bash
# Run performance benchmarks
python -m pytest mathml_parser/tests/test_performance.py::test_performance_benchmarks

# Run domain-specific tests
python -m pytest mathml_parser/tests/test_domains.py::TestComplexNumbers
python -m pytest mathml_parser/tests/test_domains.py::TestDifferentialEquations
python -m pytest mathml_parser/tests/test_domains.py::TestProbability
python -m pytest mathml_parser/tests/test_domains.py::TestSetTheory

# Run web interface tests (requires Flask dependencies)
python -m pytest web_interface/tests/ --cov=web_interface

# Run tests with different Python versions (using tox)
tox
```

#### Test Configuration
```bash
# Run tests with specific markers
python -m pytest -m "not slow"  # Skip slow tests
python -m pytest -m "integration"  # Run only integration tests
python -m pytest -m "performance"  # Run only performance tests

# Run tests with custom settings
python -m pytest --timeout=30  # Set timeout for long-running tests
python -m pytest --maxfail=5   # Stop after 5 failures
```

### Test Structure

```
mathml_parser/tests/
├── conftest.py                 # Test configuration and fixtures
├── test_core.py               # Core parser functionality
├── test_multi_format.py       # Multi-format output testing
├── test_domains.py            # Domain-specific testing
├── test_performance.py        # Performance and optimization
├── test_error_handling.py     # Error handling and validation
├── test_cli.py                # Command-line interface
├── test_integration.py        # End-to-end integration tests
├── fixtures/                  # Test data and fixtures
│   ├── expressions.json       # Sample expressions
│   ├── expected_outputs.json  # Expected parsing results
│   └── performance_data.json  # Performance benchmarks
└── utils/                     # Testing utilities
    ├── test_helpers.py        # Helper functions
    └── mock_objects.py        # Mock objects for testing
```

### Writing Tests

#### Example Unit Test
```python
import pytest
from mathml_parser import MathMLParser, parse_safe

def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    parser = MathMLParser()
    
    # Test addition
    result = parser.parse_safe("2 + 3")
    assert result.success
    assert "2" in result.mathml
    assert "3" in result.mathml
    
    # Test complex expression
    result = parser.parse_safe("x^2 + 2*x + 1")
    assert result.success
    assert result.metrics.complexity_score > 0

def test_error_handling():
    """Test error handling and validation."""
    result = parse_safe("2 + + 3")  # Invalid syntax
    assert not result.success
    assert result.error is not None
    assert "syntax" in result.error.message.lower()
```

#### Example Integration Test
```python
def test_domain_integration():
    """Test integration between domains and core parser."""
    from mathml_parser.domains import ComplexNumberProcessor
    
    processor = ComplexNumberProcessor()
    result = processor.parse_complex("3 + 4i")
    
    assert result is not None
    assert result.real_part == 3
    assert result.imaginary_part == 4
    assert abs(result.magnitude - 5.0) < 1e-10
```

### Test Data and Fixtures

#### Using Test Fixtures
```python
@pytest.fixture
def sample_expressions():
    """Provide sample mathematical expressions for testing."""
    return [
        "x^2 + 2*x + 1",
        "sin(π/2)",
        "∫(x^2)dx",
        "3 + 4i",
        "dy/dx = x + y"
    ]

def test_multiple_expressions(sample_expressions):
    """Test parsing multiple expressions."""
    parser = MathMLParser()
    
    for expr in sample_expressions:
        result = parser.parse_safe(expr)
        assert result.success, f"Failed to parse: {expr}"
```

## Logging

### Logging Architecture

The MathML Parser implements comprehensive logging across all components:

- **Core Parser**: Expression parsing events and errors
- **Domain Processors**: Domain-specific analysis and results
- **Performance Engine**: Caching, optimization, and metrics
- **Web Interface**: HTTP requests, WebSocket events, and user interactions
- **Error Handling**: Detailed error tracking and debugging information

### Logging Configuration

#### Basic Logging Setup
```python
import logging
from mathml_parser import MathMLParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mathml_parser.log'),
        logging.StreamHandler()
    ]
)

# Create parser with logging enabled
parser = MathMLParser(enable_logging=True)
```

#### Advanced Logging Configuration
```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'mathml_parser.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'INFO',
        },
        'error_file': {
            'class': 'logging.FileHandler',
            'filename': 'mathml_errors.log',
            'formatter': 'detailed',
            'level': 'ERROR',
        },
    },
    'loggers': {
        'mathml_parser': {
            'handlers': ['file', 'console', 'error_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'mathml_parser.performance': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False,
        },
        'mathml_parser.web': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Logging Examples

#### Core Parser Logging
```python
import logging
from mathml_parser import MathMLParser

logger = logging.getLogger('mathml_parser')

parser = MathMLParser()
result = parser.parse_safe("sin(π/2)")

if result.success:
    logger.info(f"Successfully parsed expression: {result.mathml}")
else:
    logger.error(f"Parse failed: {result.error.message}")
```

#### Performance Logging
```python
from mathml_parser.performance import PerformanceOptimizer
import logging

logger = logging.getLogger('mathml_parser.performance')

optimizer = PerformanceOptimizer()

@optimizer.monitor_performance("complex_parsing")
def parse_complex_expression(expr):
    # Performance metrics automatically logged
    return parser.parse_safe(expr)

result = parse_complex_expression("∫(sin(x^2))dx")
```

#### Web Interface Logging
```python
from flask import request
import logging

logger = logging.getLogger('mathml_parser.web')

@app.route('/api/parse', methods=['POST'])
def api_parse():
    expression = request.json.get('expression')
    
    logger.info(f"API request from {request.remote_addr}: {expression}")
    
    result = processor.process_expression(expression)
    
    if result.error_message:
        logger.warning(f"Parse error for '{expression}': {result.error_message}")
    else:
        logger.info(f"Successfully processed expression in {result.processing_time:.3f}s")
    
    return jsonify(result.to_dict())
```

### Log Analysis and Monitoring

#### Log File Structure
```
logs/
├── mathml_parser.log           # Main application log
├── mathml_errors.log           # Error-specific log
├── performance.log             # Performance metrics
├── web_access.log              # Web interface access log
├── cache_operations.log        # Caching operations
└── domain_analysis.log         # Domain-specific analysis
```

#### Log Analysis Commands
```bash
# Monitor real-time logs
tail -f logs/mathml_parser.log

# Search for errors
grep "ERROR" logs/mathml_parser.log

# Analyze performance metrics
grep "Performance" logs/mathml_parser.log | tail -100

# Monitor web interface usage
grep "API request" logs/mathml_parser.log | wc -l

# Find slow parsing operations
grep "processing_time.*[5-9]\.[0-9]" logs/mathml_parser.log
```

#### Performance Monitoring
```python
import logging
from mathml_parser.performance import PerformanceBenchmark

logger = logging.getLogger('mathml_parser.performance')

# Log performance metrics
benchmark = PerformanceBenchmark()
results = benchmark.run_comprehensive_benchmark()

for test_name, metrics in results.items():
    logger.info(f"Performance test {test_name}: {metrics['avg_time']:.4f}s average")
    
    if metrics['avg_time'] > 1.0:  # Alert for slow operations
        logger.warning(f"Slow performance detected in {test_name}: {metrics['avg_time']:.4f}s")
```

### Debug Logging

#### Enable Debug Mode
```python
import logging
from mathml_parser import MathMLParser

# Enable debug logging
logging.getLogger('mathml_parser').setLevel(logging.DEBUG)

parser = MathMLParser(debug_mode=True)
result = parser.parse_safe("complex expression here")

# Debug information automatically logged:
# - Token parsing steps
# - Grammar rule applications  
# - Domain processor invocations
# - Cache hit/miss ratios
# - Performance measurements
```

#### Custom Debug Information
```python
import logging
from mathml_parser.core.parser import MathMLParser

logger = logging.getLogger('mathml_parser.debug')

class DebugMathMLParser(MathMLParser):
    def parse_safe(self, expression):
        logger.debug(f"Starting parse for: {expression}")
        
        result = super().parse_safe(expression)
        
        if result.success:
            logger.debug(f"Parse successful: {len(result.mathml)} chars generated")
        else:
            logger.debug(f"Parse failed at position {result.error.position}")
        
        return result
```

### Production Logging

#### Production Configuration
```python
# production_logging.py
import logging.config
import os

PRODUCTION_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'production': {
            'format': '%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': '/var/log/mathml_parser/app.log',
            'when': 'midnight',
            'interval': 1,
            'backupCount': 30,
            'formatter': 'production',
            'level': 'INFO',
        },
        'error_file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': '/var/log/mathml_parser/error.log',
            'when': 'midnight',
            'interval': 1,
            'backupCount': 90,
            'formatter': 'production',
            'level': 'ERROR',
        },
        'syslog': {
            'class': 'logging.handlers.SysLogHandler',
            'address': '/dev/log',
            'formatter': 'production',
            'level': 'WARNING',
        },
    },
    'loggers': {
        'mathml_parser': {
            'handlers': ['file', 'error_file', 'syslog'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Apply configuration
if os.environ.get('FLASK_ENV') == 'production':
    logging.config.dictConfig(PRODUCTION_LOGGING)
```

#### Monitoring and Alerting
```python
import logging
from logging.handlers import SMTPHandler

# Email alerts for critical errors
if not app.debug:
    mail_handler = SMTPHandler(
        mailhost='localhost',
        fromaddr='server-error@mathml-parser.com',
        toaddrs=['admin@mathml-parser.com'],
        subject='MathML Parser Error',
        credentials=None,
        secure=None
    )
    mail_handler.setLevel(logging.ERROR)
    mail_handler.setFormatter(logging.Formatter('''
    Message type:       %(levelname)s
    Location:           %(pathname)s:%(lineno)d
    Module:             %(module)s
    Function:           %(funcName)s
    Time:               %(asctime)s
    
    Message:
    
    %(message)s
    '''))
    
    app.logger.addHandler(mail_handler)
```

## Development

### Code Quality

```bash
# Format code
python -m black mathml_parser/

# Sort imports
python -m isort mathml_parser/

# Type checking
python -m mypy mathml_parser/

# Linting
python -m flake8 mathml_parser/
```

### Building Documentation

```bash
cd docs/
make html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository from [https://github.com/ahmedfarid59/math-ml](https://github.com/ahmedfarid59/math-ml)
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

- **Documentation**: [Project Documentation](https://github.com/ahmedfarid59/math-ml/wiki)
- **Issue Tracker**: [GitHub Issues](https://github.com/ahmedfarid59/math-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ahmedfarid59/math-ml/discussions)

## Acknowledgments

- Built with [Lark](https://github.com/lark-parser/lark) parsing toolkit
- Inspired by mathematical notation standards
- Thanks to all contributors and users