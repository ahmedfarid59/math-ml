# Core Module API

The core module provides the fundamental parsing and processing functionality for MathML Parser.

## MathMLParser Class

```{eval-rst}
.. autoclass:: mathml_parser.core.parser.MathMLParser
   :members:
   :undoc-members:
   :show-inheritance:
```

### Constructor

```python
MathMLParser(
    grammar_file: Optional[str] = None,
    strict_mode: bool = False,
    auto_simplify: bool = True,
    cache_size: int = 1000,
    timeout: Optional[int] = None
)
```

**Parameters:**
- `grammar_file`: Path to custom Lark grammar file
- `strict_mode`: Enable strict parsing (raises errors on ambiguity)
- `auto_simplify`: Automatically simplify parsed expressions
- `cache_size`: Maximum number of cached expressions (0 to disable)
- `timeout`: Parsing timeout in seconds

**Example:**
```python
from mathml_parser import MathMLParser

# Basic parser
parser = MathMLParser()

# Configured parser
parser = MathMLParser(
    strict_mode=True,
    cache_size=5000,
    timeout=30
)
```

### Methods

#### parse()

```python
def parse(self, expression: str) -> ExpressionTree:
    """Parse mathematical expression into an expression tree."""
```

**Parameters:**
- `expression` (str): Mathematical expression to parse

**Returns:**
- `ExpressionTree`: Parsed expression tree

**Raises:**
- `MathMLParseError`: If parsing fails
- `TimeoutError`: If parsing exceeds timeout

**Example:**
```python
parser = MathMLParser()
result = parser.parse("x^2 + 2*x + 1")
print(result)  # ExpressionTree representation
```

#### parse_batch()

```python
def parse_batch(
    self, 
    expressions: List[str],
    parallel: bool = True
) -> List[ExpressionTree]:
    """Parse multiple expressions efficiently."""
```

**Parameters:**
- `expressions` (List[str]): List of expressions to parse
- `parallel` (bool): Use parallel processing if available

**Returns:**
- `List[ExpressionTree]`: List of parsed expression trees

**Example:**
```python
expressions = ["x^2 + 1", "sin(x)", "∫(e^x, 0, 1)"]
results = parser.parse_batch(expressions)
```

#### simplify()

```python
def simplify(self, expression: Union[str, ExpressionTree]) -> ExpressionTree:
    """Simplify mathematical expression."""
```

**Parameters:**
- `expression`: Expression to simplify (string or tree)

**Returns:**
- `ExpressionTree`: Simplified expression tree

**Example:**
```python
simplified = parser.simplify("x + x + 1")  # Returns 2*x + 1
```

#### validate()

```python
def validate(self, expression: str) -> ValidationResult:
    """Validate mathematical expression without full parsing."""
```

**Parameters:**
- `expression` (str): Expression to validate

**Returns:**
- `ValidationResult`: Validation results with errors and warnings

**Example:**
```python
result = parser.validate("x^2 + ")  # Incomplete expression
print(result.is_valid)  # False
print(result.errors)    # List of error messages
```

#### configure()

```python
def configure(self, config: Dict[str, Any]) -> None:
    """Update parser configuration."""
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration parameters

**Available Configuration Options:**
- `angle_units`: 'radians' or 'degrees'
- `decimal_notation`: 'american' or 'european'
- `matrix_style`: 'brackets' or 'parentheses'
- `function_style`: 'upright' or 'italic'
- `implicit_multiplication`: bool

**Example:**
```python
parser.configure({
    'angle_units': 'degrees',
    'matrix_style': 'parentheses',
    'implicit_multiplication': True
})
```

#### get_statistics()

```python
def get_statistics(self) -> ParserStatistics:
    """Get parser performance statistics."""
```

**Returns:**
- `ParserStatistics`: Statistics about parser usage

**Example:**
```python
stats = parser.get_statistics()
print(f"Expressions parsed: {stats.total_parsed}")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
print(f"Average parse time: {stats.avg_parse_time:.3f}s")
```

## ExpressionTree Class

```{eval-rst}
.. autoclass:: mathml_parser.core.tree.ExpressionTree
   :members:
   :undoc-members:
   :show-inheritance:
```

### Properties

#### expression_type

```python
@property
def expression_type(self) -> ExpressionType:
    """Get the type of mathematical expression."""
```

**Returns:**
- `ExpressionType`: Type enum (ARITHMETIC, ALGEBRAIC, CALCULUS, etc.)

#### variables

```python
@property
def variables(self) -> Set[str]:
    """Get set of variables in the expression."""
```

**Returns:**
- `Set[str]`: Set of variable names

#### complexity

```python
@property
def complexity(self) -> int:
    """Get complexity score of the expression."""
```

**Returns:**
- `int`: Complexity score (higher = more complex)

### Methods

#### evaluate()

```python
def evaluate(
    self, 
    variables: Optional[Dict[str, float]] = None
) -> Union[float, complex]:
    """Evaluate expression numerically."""
```

**Parameters:**
- `variables` (Dict[str, float], optional): Variable substitutions

**Returns:**
- `Union[float, complex]`: Numerical result

**Example:**
```python
tree = parser.parse("x^2 + 2*x + 1")
result = tree.evaluate({'x': 3})  # Returns 16.0
```

#### substitute()

```python
def substitute(
    self, 
    substitutions: Dict[str, Union[str, ExpressionTree]]
) -> ExpressionTree:
    """Substitute variables with expressions."""
```

**Parameters:**
- `substitutions`: Variable to expression mappings

**Returns:**
- `ExpressionTree`: New tree with substitutions applied

**Example:**
```python
tree = parser.parse("f(x)")
new_tree = tree.substitute({'f': 'sin', 'x': 'π/2'})
```

#### differentiate()

```python
def differentiate(self, variable: str) -> ExpressionTree:
    """Compute symbolic derivative."""
```

**Parameters:**
- `variable` (str): Variable to differentiate with respect to

**Returns:**
- `ExpressionTree`: Derivative expression tree

**Example:**
```python
tree = parser.parse("x^3 + 2*x^2 + x")
derivative = tree.differentiate('x')  # 3*x^2 + 4*x + 1
```

#### integrate()

```python
def integrate(
    self, 
    variable: str,
    limits: Optional[Tuple[str, str]] = None
) -> ExpressionTree:
    """Compute symbolic integral."""
```

**Parameters:**
- `variable` (str): Variable to integrate
- `limits` (Tuple[str, str], optional): Integration limits

**Returns:**
- `ExpressionTree`: Integral expression tree

**Example:**
```python
tree = parser.parse("x^2")
integral = tree.integrate('x')  # x^3/3 + C
definite = tree.integrate('x', ('0', '1'))  # 1/3
```

#### to_string()

```python
def to_string(self, format_style: str = 'standard') -> str:
    """Convert tree back to string representation."""
```

**Parameters:**
- `format_style` (str): Output style ('standard', 'latex', 'ascii')

**Returns:**
- `str`: String representation

**Example:**
```python
tree = parser.parse("x^2 + 1")
standard = tree.to_string('standard')  # "x^2 + 1"
latex = tree.to_string('latex')        # "x^{2} + 1"
```

## Exception Classes

### MathMLParseError

```python
class MathMLParseError(Exception):
    """Base exception for parsing errors."""
    
    def __init__(
        self, 
        message: str,
        expression: str,
        location: Optional[int] = None,
        suggestions: Optional[List[str]] = None
    ):
        self.expression = expression
        self.location = location
        self.suggestions = suggestions or []
        super().__init__(message)
```

**Attributes:**
- `expression` (str): The expression that failed to parse
- `location` (int, optional): Character position of error
- `suggestions` (List[str]): Suggested corrections

**Example:**
```python
try:
    parser.parse("x^^ + 1")
except MathMLParseError as e:
    print(f"Error: {e}")
    print(f"Expression: {e.expression}")
    print(f"Location: {e.location}")
    print(f"Suggestions: {e.suggestions}")
```

### Specialized Parse Errors

#### SyntaxError

```python
class SyntaxError(MathMLParseError):
    """Invalid mathematical syntax."""
```

#### SemanticError

```python
class SemanticError(MathMLParseError):
    """Semantically incorrect expression."""
```

#### TimeoutError

```python
class TimeoutError(MathMLParseError):
    """Parsing timeout exceeded."""
```

## Utility Functions

### parse_expression()

```python
def parse_expression(
    expression: str,
    parser_config: Optional[Dict[str, Any]] = None
) -> ExpressionTree:
    """Convenience function for parsing expressions."""
```

**Parameters:**
- `expression` (str): Expression to parse
- `parser_config` (Dict, optional): Parser configuration

**Returns:**
- `ExpressionTree`: Parsed expression tree

**Example:**
```python
from mathml_parser.core import parse_expression

tree = parse_expression("x^2 + 1", {'strict_mode': True})
```

### simplify_expression()

```python
def simplify_expression(
    expression: Union[str, ExpressionTree]
) -> ExpressionTree:
    """Convenience function for simplifying expressions."""
```

**Parameters:**
- `expression`: Expression to simplify

**Returns:**
- `ExpressionTree`: Simplified expression tree

**Example:**
```python
from mathml_parser.core import simplify_expression

simplified = simplify_expression("x + x + 1")  # 2*x + 1
```

### validate_expression()

```python
def validate_expression(expression: str) -> ValidationResult:
    """Convenience function for validating expressions."""
```

**Parameters:**
- `expression` (str): Expression to validate

**Returns:**
- `ValidationResult`: Validation results

**Example:**
```python
from mathml_parser.core import validate_expression

result = validate_expression("sin(x")  # Missing closing parenthesis
print(result.is_valid)  # False
print(result.errors)    # ["Missing closing parenthesis"]
```

## Data Types

### ExpressionType Enum

```python
class ExpressionType(Enum):
    """Types of mathematical expressions."""
    
    ARITHMETIC = "arithmetic"
    ALGEBRAIC = "algebraic"
    TRANSCENDENTAL = "transcendental"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    COMPLEX = "complex"
```

### ValidationResult Class

```python
@dataclass
class ValidationResult:
    """Result of expression validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    expression_type: Optional[ExpressionType] = None
    complexity_score: Optional[int] = None
```

### ParserStatistics Class

```python
@dataclass
class ParserStatistics:
    """Parser performance statistics."""
    
    total_parsed: int
    total_errors: int
    cache_hits: int
    cache_misses: int
    avg_parse_time: float
    max_parse_time: float
    min_parse_time: float
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.total_parsed + self.total_errors
        return self.total_errors / total if total > 0 else 0.0
```

## Configuration

### Default Configuration

```python
DEFAULT_CONFIG = {
    'angle_units': 'radians',
    'decimal_notation': 'american',
    'matrix_style': 'brackets',
    'function_style': 'upright',
    'implicit_multiplication': True,
    'strict_parentheses': False,
    'allow_unicode': True,
    'normalize_whitespace': True,
    'case_sensitive': True
}
```

### Configuration Validation

The parser validates configuration parameters and provides helpful error messages for invalid values:

```python
try:
    parser.configure({'angle_units': 'invalid'})
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
    print(f"Valid options: {e.valid_options}")
```

## Thread Safety

The MathMLParser class is thread-safe for read operations. However, configuration changes should be synchronized:

```python
import threading
from mathml_parser import MathMLParser

parser = MathMLParser()
lock = threading.Lock()

def safe_configure(config):
    with lock:
        parser.configure(config)

# Safe to use parser.parse() from multiple threads
# Use safe_configure() for configuration changes
```

## Memory Management

### Cache Management

```python
# Configure cache size
parser = MathMLParser(cache_size=10000)

# Monitor cache usage
stats = parser.get_statistics()
if stats.cache_hit_rate < 0.5:
    parser.clear_cache()  # Clear if hit rate is low

# Disable caching for memory-constrained environments
parser = MathMLParser(cache_size=0)
```

### Large Expression Handling

For very large expressions, consider using streaming or chunking:

```python
# For expressions with many terms
long_expr = " + ".join([f"x^{i}" for i in range(1000)])

# Parse in chunks if needed
chunks = split_expression(long_expr, max_terms=100)
results = [parser.parse(chunk) for chunk in chunks]
combined = combine_trees(results)
```