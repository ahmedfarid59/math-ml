# API Reference

Welcome to the comprehensive API reference for MathML Parser. This section provides detailed documentation for all public classes, functions, and modules.

## Core Modules

```{eval-rst}
.. toctree::
   :maxdepth: 2
   
   core
   parsers
   transformers
   formats
```

## Quick Reference

### Main Classes

- **[MathMLParser](core.md#mathmlparser)** - Main parser class for mathematical expressions
- **[MultiFormatRenderer](formats.md#multiformatrenderer)** - Multi-format output renderer
- **[ExpressionOptimizer](transformers.md#expressionoptimizer)** - Expression optimization and simplification

### Formatter Classes

- **[LaTeXFormatter](formats.md#latexformatter)** - LaTeX output formatting
- **[HTMLFormatter](formats.md#htmlformatter)** - HTML/CSS output formatting
- **[ASCIIMathFormatter](formats.md#asciimathformatter)** - ASCII Math formatting
- **[SVGFormatter](formats.md#svgformatter)** - SVG graphics formatting

### Utility Functions

- **[render_expression()](formats.md#render-expression)** - Convert expression to specific format
- **[render_all_formats()](formats.md#render-all-formats)** - Convert to all available formats
- **[parse_expression()](parsers.md#parse-expression)** - Parse mathematical notation

## Usage Patterns

### Basic Usage

```python
from mathml_parser import MathMLParser, render_expression

# Create parser
parser = MathMLParser()

# Parse expression
result = parser.parse("x^2 + 1")

# Render in different formats
latex = render_expression("x^2 + 1", 'latex')
html = render_expression("x^2 + 1", 'html')
```

### Advanced Configuration

```python
from mathml_parser import MathMLParser
from mathml_parser.formatters import LaTeXFormatter

# Configure parser
parser = MathMLParser(
    strict_mode=False,
    auto_simplify=True,
    cache_size=10000
)

# Configure custom formatter
formatter = LaTeXFormatter()
formatter.configure({
    'use_amsmath': True,
    'fraction_style': 'display'
})
```

### Error Handling

```python
from mathml_parser import MathMLParser, MathMLParseError

parser = MathMLParser()

try:
    result = parser.parse("invalid expression")
except MathMLParseError as e:
    print(f"Parse error: {e}")
    print(f"Error location: {e.location}")
    print(f"Suggestions: {e.suggestions}")
```

## Module Overview

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [mathml_parser.core](core.md) | Core parsing functionality | MathMLParser, ExpressionTree |
| [mathml_parser.parsers](parsers.md) | Specialized parsers | LaTeXParser, ASCIIMathParser |
| [mathml_parser.formatters](formats.md) | Output formatters | LaTeXFormatter, HTMLFormatter |
| [mathml_parser.transformers](transformers.md) | Expression transformations | Optimizer, Simplifier |
| [mathml_parser.utils](core.md#utilities) | Utility functions | Validator, Helper functions |

## Type Annotations

MathML Parser uses comprehensive type annotations for better IDE support and type checking:

```python
from typing import Dict, List, Optional, Union
from mathml_parser.types import (
    ExpressionType,
    FormatType,
    ParseResult,
    RenderResult
)

def render_expression(
    expression: ExpressionType,
    format_type: FormatType = 'latex'
) -> RenderResult:
    """Type-annotated render function"""
    pass
```

## Configuration Options

### Global Configuration

```python
from mathml_parser.config import configure_global_settings

configure_global_settings({
    'default_format': 'latex',
    'strict_parsing': False,
    'enable_caching': True,
    'cache_size': 10000,
    'timeout': 30,
    'parallel_processing': True
})
```

### Parser-Specific Configuration

```python
parser = MathMLParser()
parser.configure({
    'angle_units': 'radians',  # or 'degrees'
    'decimal_notation': 'american',  # or 'european'
    'matrix_style': 'brackets',  # or 'parentheses'
    'function_style': 'upright',  # or 'italic'
    'symbol_set': 'unicode'  # or 'ascii'
})
```

## Extension Points

### Custom Formatters

```python
from mathml_parser.formatters.base import OutputFormatter

class CustomFormatter(OutputFormatter):
    def format_expression(self, expression: str) -> str:
        # Custom formatting logic
        return formatted_result
    
    def get_format_name(self) -> str:
        return "custom"

# Register custom formatter
from mathml_parser import register_formatter
register_formatter(CustomFormatter())
```

### Custom Parsers

```python
from mathml_parser.parsers.base import BaseParser

class CustomParser(BaseParser):
    def parse(self, expression: str) -> ParseResult:
        # Custom parsing logic
        return parse_result

# Use custom parser
parser = MathMLParser(custom_parser=CustomParser())
```

## Performance Considerations

### Caching

```python
# Enable caching for repeated expressions
parser = MathMLParser(cache_size=10000)

# Clear cache when needed
parser.clear_cache()

# Get cache statistics
stats = parser.get_cache_stats()
print(f"Cache hits: {stats.hits}")
print(f"Cache misses: {stats.misses}")
```

### Parallel Processing

```python
from mathml_parser.parallel import parallel_render

expressions = ["x^2 + 1", "sin(x)", "∫(e^x, 0, 1)"]

# Process in parallel
results = parallel_render(expressions, format_type='latex', workers=4)
```

### Memory Management

```python
# For large-scale processing
from mathml_parser.streaming import StreamingProcessor

processor = StreamingProcessor()

with open('large_file.txt') as f:
    for result in processor.process_stream(f, format_type='latex'):
        # Process results one at a time
        handle_result(result)
```

## Error Reference

### Exception Hierarchy

```
MathMLError
├── MathMLParseError
│   ├── SyntaxError
│   ├── SemanticError
│   └── TimeoutError
├── MathMLRenderError
│   ├── FormatError
│   └── ConversionError
└── MathMLConfigError
    ├── InvalidConfigError
    └── MissingDependencyError
```

### Common Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| 1001 | SyntaxError | Invalid mathematical syntax |
| 1002 | SemanticError | Semantically incorrect expression |
| 1003 | TimeoutError | Parsing timeout exceeded |
| 2001 | FormatError | Unsupported output format |
| 2002 | ConversionError | Format conversion failed |
| 3001 | InvalidConfigError | Invalid configuration parameter |
| 3002 | MissingDependencyError | Required dependency not found |

## Version Compatibility

### API Stability

MathML Parser follows semantic versioning:

- **Major version** (x.0.0): Breaking API changes
- **Minor version** (0.x.0): New features, backward compatible
- **Patch version** (0.0.x): Bug fixes, backward compatible

### Deprecation Policy

Deprecated features are marked with warnings and removed in the next major version:

```python
import warnings
from mathml_parser import deprecated_function

# This will show a deprecation warning
with warnings.catch_warnings():
    warnings.simplefilter("always")
    result = deprecated_function()  # DeprecationWarning
```

### Migration Guide

When upgrading between major versions, refer to the [migration guide](../appendices/migration.md) for breaking changes and update instructions.

## Testing and Validation

### Built-in Validation

```python
from mathml_parser.validation import validate_expression

result = validate_expression("x^2 + 1")
print(f"Valid: {result.is_valid}")
print(f"Warnings: {result.warnings}")
print(f"Suggestions: {result.suggestions}")
```

### Test Utilities

```python
from mathml_parser.testing import (
    assert_expressions_equivalent,
    assert_format_matches,
    generate_test_expressions
)

# Test mathematical equivalence
assert_expressions_equivalent("x^2 - 1", "(x-1)(x+1)")

# Test format output
assert_format_matches("x^2", 'latex', "$x^{2}$")

# Generate test data
test_exprs = generate_test_expressions(category='algebra', count=100)
```