# Output Formats API

The formats module provides comprehensive support for rendering mathematical expressions in multiple output formats.

## MultiFormatRenderer Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.MultiFormatRenderer
   :members:
   :undoc-members:
   :show-inheritance:
```

### Constructor

```python
MultiFormatRenderer()
```

Creates a renderer with all default formatters registered.

**Example:**
```python
from mathml_parser.core.multi_format import MultiFormatRenderer

renderer = MultiFormatRenderer()
```

### Methods

#### render_expression()

```python
def render_expression(self, expression: str, format_type: str) -> str:
    """Render expression in specified format."""
```

**Parameters:**
- `expression` (str): Mathematical expression to render
- `format_type` (str): Target output format

**Returns:**
- `str`: Formatted expression

**Available Formats:**
- `'latex'`: LaTeX mathematical notation
- `'ascii'`: ASCII Math notation
- `'html'`: HTML with CSS styling
- `'svg'`: SVG graphics format
- `'text'`: Plain text format

**Example:**
```python
renderer = MultiFormatRenderer()

# LaTeX output
latex = renderer.render_expression("x^2 + 1", "latex")
print(latex)  # $x^{2} + 1$

# HTML output
html = renderer.render_expression("x^2 + 1", "html")
# Returns styled HTML markup
```

#### render_all_formats()

```python
def render_all_formats(self, expression: str) -> Dict[str, str]:
    """Render expression in all available formats."""
```

**Parameters:**
- `expression` (str): Mathematical expression to render

**Returns:**
- `Dict[str, str]`: Dictionary mapping format names to rendered expressions

**Example:**
```python
all_formats = renderer.render_all_formats("∫(x^2, 0, 1)")

for format_name, result in all_formats.items():
    print(f"{format_name:8}: {result}")
```

#### get_available_formats()

```python
def get_available_formats(self) -> List[str]:
    """Get list of available output formats."""
```

**Returns:**
- `List[str]`: List of format names

#### register_formatter()

```python
def register_formatter(self, formatter: OutputFormatter) -> None:
    """Register a new output formatter."""
```

**Parameters:**
- `formatter` (OutputFormatter): Formatter instance to register

**Example:**
```python
from mathml_parser.formatters.base import OutputFormatter

class CustomFormatter(OutputFormatter):
    def format_expression(self, expression: str) -> str:
        return f"Custom: {expression}"
    
    def get_format_name(self) -> str:
        return "custom"

renderer.register_formatter(CustomFormatter())
```

## OutputFormatter Base Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.OutputFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Abstract base class for all output formatters.

### Abstract Methods

#### format_expression()

```python
@abstractmethod
def format_expression(self, expression: str) -> str:
    """Format the mathematical expression to the target output format."""
```

#### get_format_name()

```python
@abstractmethod
def get_format_name(self) -> str:
    """Get the name of this output format."""
```

## LaTeXFormatter Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.LaTeXFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Converts mathematical expressions to LaTeX format.

### Features

- Comprehensive Greek letter support
- Proper fraction formatting with `\frac{}{}`
- Superscript and subscript handling
- Mathematical function formatting
- Matrix and vector notation
- Integral, summation, and limit notation
- Mathematical symbol conversion

### Methods

#### format_expression()

```python
def format_expression(self, expression: str) -> str:
    """Convert mathematical expression to LaTeX format."""
```

**Parameters:**
- `expression` (str): Mathematical expression in standard notation

**Returns:**
- `str`: LaTeX-formatted expression wrapped in math delimiters

**Examples:**
```python
formatter = LaTeXFormatter()

# Basic expressions
result = formatter.format_expression("x^2 + 1")
print(result)  # $x^{2} + 1$

# Complex fractions
result = formatter.format_expression("(a+b)/(c+d)")
print(result)  # $\frac{a+b}{c+d}$

# Greek letters and symbols
result = formatter.format_expression("α + β = π")
print(result)  # $\alpha + \beta = \pi$

# Integrals and limits
result = formatter.format_expression("∫(x^2, 0, 1)")
print(result)  # $\int_{0}^{1} x^{2}$

# Matrices
result = formatter.format_expression("[1,2;3,4]")
print(result)  # $\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}$
```

### Symbol Mappings

The formatter includes comprehensive symbol mappings:

```python
# Greek letters
'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma'

# Mathematical symbols  
'∞': r'\infty', '∫': r'\int', '∑': r'\sum'

# Operators
'≤': r'\leq', '≥': r'\geq', '≠': r'\neq'
```

## ASCIIMathFormatter Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.ASCIIMathFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Converts expressions to ASCII Math format using plain ASCII characters.

### Features

- Converts Unicode symbols to ASCII equivalents
- Uses simple notation for fractions, roots, and powers
- Maintains readability in plain text format
- Compatible with ASCII Math parsers and renderers

### Methods

#### format_expression()

```python
def format_expression(self, expression: str) -> str:
    """Convert mathematical expression to ASCII Math format."""
```

**Examples:**
```python
formatter = ASCIIMathFormatter()

# Greek letters
result = formatter.format_expression("α + β")
print(result)  # alpha + beta

# Mathematical symbols
result = formatter.format_expression("∞ ≤ x ≥ 0")
print(result)  # infinity <= x >= 0

# Functions
result = formatter.format_expression("arcsin(x)")
print(result)  # sin^-1(x)
```

### Symbol Mappings

```python
# Greek letters
'α': 'alpha', 'β': 'beta', 'γ': 'gamma'

# Mathematical symbols
'∞': 'infinity', '∫': 'int', '∑': 'sum'

# Operators
'≤': '<=', '≥': '>=', '≠': '!='
```

## HTMLFormatter Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.HTMLFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Generates HTML markup with CSS styling for web display.

### Features

- CSS-styled mathematical expressions
- Proper fraction layout with horizontal lines
- Superscript and subscript positioning
- Mathematical symbol rendering
- Responsive design considerations

### Methods

#### format_expression()

```python
def format_expression(self, expression: str) -> str:
    """Convert mathematical expression to HTML with CSS styling."""
```

**Examples:**
```python
formatter = HTMLFormatter()

# Basic expression
result = formatter.format_expression("x^2 + 1")
print(result)
# <div class="math-expression">x<span class="math-superscript">2</span> + 1</div>

# Fractions
result = formatter.format_expression("a/b")
# <div class="math-expression">
#   <div class="math-fraction">
#     <div class="numerator">a</div>
#     <div class="denominator">b</div>
#   </div>
# </div>
```

#### get_css_styles()

```python
def get_css_styles(self) -> str:
    """Return the CSS styles for mathematical formatting."""
```

**Returns:**
- `str`: Complete CSS stylesheet for mathematical formatting

**Example:**
```python
css = formatter.get_css_styles()
# Include this CSS in your HTML document for proper styling
```

### CSS Classes

The HTML formatter uses these CSS classes:

- `.math-expression`: Container for mathematical expressions
- `.math-fraction`: Fraction container
- `.math-superscript`: Superscript text
- `.math-subscript`: Subscript text
- `.math-function`: Mathematical function names
- `.math-operator`: Mathematical operators
- `.math-symbol`: Mathematical symbols

## SVGFormatter Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.SVGFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Generates SVG markup for mathematical expressions.

### Methods

#### format_expression()

```python
def format_expression(self, expression: str) -> str:
    """Convert mathematical expression to SVG format."""
```

**Example:**
```python
formatter = SVGFormatter()

result = formatter.format_expression("x^2 + 1")
# Returns SVG markup with mathematical expression
```

## PlainTextFormatter Class

```{eval-rst}
.. autoclass:: mathml_parser.core.multi_format.PlainTextFormatter
   :members:
   :undoc-members:
   :show-inheritance:
```

Converts expressions to readable plain text format.

### Methods

#### format_expression()

```python
def format_expression(self, expression: str) -> str:
    """Convert mathematical expression to plain text format."""
```

**Example:**
```python
formatter = PlainTextFormatter()

result = formatter.format_expression("α + π")
print(result)  # alpha + pi
```

## Convenience Functions

### render_expression()

```python
def render_expression(expression: str, format_type: str) -> str:
    """Render mathematical expression in specified format."""
```

**Parameters:**
- `expression` (str): Mathematical expression to render
- `format_type` (str): Target output format

**Returns:**
- `str`: Formatted expression

**Example:**
```python
from mathml_parser.core.multi_format import render_expression

latex = render_expression("x^2 + 1", 'latex')
html = render_expression("x^2 + 1", 'html')
ascii_math = render_expression("x^2 + 1", 'ascii')
```

### render_all_formats()

```python
def render_all_formats(expression: str) -> Dict[str, str]:
    """Render mathematical expression in all available formats."""
```

**Parameters:**
- `expression` (str): Mathematical expression to render

**Returns:**
- `Dict[str, str]`: Dictionary mapping format names to rendered expressions

**Example:**
```python
from mathml_parser.core.multi_format import render_all_formats

results = render_all_formats("∫(sin(x), 0, π)")
for format_name, output in results.items():
    print(f"{format_name}: {output}")
```

### get_available_formats()

```python
def get_available_formats() -> List[str]:
    """Get list of available output formats."""
```

**Returns:**
- `List[str]`: List of format names

**Example:**
```python
from mathml_parser.core.multi_format import get_available_formats

formats = get_available_formats()
print(f"Available formats: {', '.join(formats)}")
```

## Advanced Usage

### Custom Formatter Development

```python
from mathml_parser.core.multi_format import OutputFormatter
import re

class MarkdownFormatter(OutputFormatter):
    """Custom formatter for Markdown mathematical notation"""
    
    def format_expression(self, expression: str) -> str:
        result = expression
        
        # Convert superscripts to Markdown
        result = re.sub(r'([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)', 
                       r'\1<sup>\2</sup>', result)
        
        # Convert subscripts to Markdown  
        result = re.sub(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)',
                       r'\1<sub>\2</sub>', result)
        
        # Wrap in math delimiters
        return f"${result}$"
    
    def get_format_name(self) -> str:
        return "markdown"

# Register and use custom formatter
renderer = MultiFormatRenderer()
renderer.register_formatter(MarkdownFormatter())

result = renderer.render_expression("x^2 + y_1", "markdown")
print(result)  # $x<sup>2</sup> + y<sub>1</sub>$
```

### Batch Processing

```python
def batch_render(expressions: List[str], format_type: str) -> List[str]:
    """Render multiple expressions efficiently"""
    renderer = MultiFormatRenderer()
    return [renderer.render_expression(expr, format_type) 
            for expr in expressions]

# Process multiple expressions
expressions = ["x^2 + 1", "sin(π/2)", "∫(e^x, 0, 1)"]
latex_results = batch_render(expressions, 'latex')
```

### Format Conversion Pipeline

```python
class FormatConverter:
    """Pipeline for converting between multiple formats"""
    
    def __init__(self):
        self.renderer = MultiFormatRenderer()
    
    def convert(self, expression: str, 
                from_format: str, to_format: str) -> str:
        # For now, assume all input is in standard notation
        # Future versions could parse from specific formats
        return self.renderer.render_expression(expression, to_format)
    
    def convert_batch(self, expressions: List[str],
                     from_format: str, to_format: str) -> List[str]:
        return [self.convert(expr, from_format, to_format) 
                for expr in expressions]

# Usage
converter = FormatConverter()
latex_output = converter.convert("x^2 + 1", "standard", "latex")
```

### Error Handling

```python
from mathml_parser.core.multi_format import MultiFormatRenderer

renderer = MultiFormatRenderer()

def safe_render(expression: str, format_type: str) -> str:
    """Safely render expression with error handling"""
    try:
        return renderer.render_expression(expression, format_type)
    except ValueError as e:
        if "not supported" in str(e):
            available = renderer.get_available_formats()
            return f"Error: Format '{format_type}' not available. Use: {available}"
        else:
            return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# Safe usage
result = safe_render("x^2 + 1", "invalid_format")
print(result)  # Error message instead of exception
```

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache

class CachedRenderer:
    """Renderer with expression-level caching"""
    
    def __init__(self, cache_size: int = 1000):
        self.renderer = MultiFormatRenderer()
        self._render_cached = lru_cache(maxsize=cache_size)(
            self._render_uncached
        )
    
    def _render_uncached(self, expression: str, format_type: str) -> str:
        return self.renderer.render_expression(expression, format_type)
    
    def render_expression(self, expression: str, format_type: str) -> str:
        return self._render_cached(expression, format_type)
    
    def clear_cache(self):
        self._render_cached.cache_clear()

# Usage with caching
cached_renderer = CachedRenderer(cache_size=5000)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

def parallel_render(expressions: List[str], 
                   format_type: str,
                   max_workers: int = 4) -> List[str]:
    """Render expressions in parallel"""
    renderer = MultiFormatRenderer()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(renderer.render_expression, expr, format_type)
                  for expr in expressions]
        
        return [future.result() for future in futures]

# Parallel processing
expressions = ["x^2 + 1"] * 1000  # Large batch
results = parallel_render(expressions, 'latex', max_workers=8)
```