# Contributing to MathML Parser

Thank you for your interest in contributing to the MathML Parser project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Guidelines](#documentation-guidelines)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them learn
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that people have different skill levels and backgrounds

### Unacceptable Behavior

- Harassment, discrimination, or inappropriate comments
- Spam, trolling, or deliberately disruptive behavior
- Publishing others' private information without permission
- Any conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of mathematical notation
- Familiarity with parsing concepts (helpful but not required)

### Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/mathml-parser.git
   cd mathml-parser
   ```

3. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   
   # Install development dependencies
   pip install -e ".[dev,docs,performance]"
   ```

4. **Verify installation**:
   ```bash
   python -m pytest mathml_parser/tests/
   python -m mathml_parser --help
   ```

## Development Setup

### Project Structure

```
mathml_parser/
├── __init__.py              # Main package interface
├── cli.py                   # Command-line interface
├── core/                    # Core parsing functionality
│   ├── grammar.py           # Mathematical grammar definition
│   ├── transformer.py       # MathML transformation logic
│   ├── parser.py            # Main parser class
│   ├── validator.py         # Input validation
│   ├── optimizer.py         # Expression optimization
│   ├── multi_format.py      # Multiple output formats
│   └── document_processor.py # Document processing
├── config/                  # Configuration management
├── plugins/                 # Plugin architecture
├── web/                     # Web API integration
├── education/               # Educational tools
├── visualization/           # Plotting and visualization
└── locale/                  # Internationalization
```

### Key Development Tools

- **Lark**: Parsing toolkit for grammar definition
- **pytest**: Testing framework
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting
- **Sphinx**: Documentation generation

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement features or fix bugs
4. **Documentation**: Improve docs, tutorials, or examples
5. **Testing**: Add test cases or improve test coverage
6. **Performance**: Optimize parsing speed or memory usage

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** to discuss major changes
3. **Review the roadmap** to understand project direction
4. **Read the architecture docs** to understand the codebase

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following coding standards

3. **Add tests** for new functionality

4. **Run the test suite**:
   ```bash
   python -m pytest mathml_parser/tests/ -v --cov=mathml_parser
   ```

5. **Check code quality**:
   ```bash
   # Format code
   black mathml_parser/
   isort mathml_parser/
   
   # Type checking
   mypy mathml_parser/
   
   # Linting
   flake8 mathml_parser/
   ```

6. **Update documentation** if needed

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add support for complex numbers"
   ```

8. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

### PR Requirements

- [ ] **Clear description** of changes and motivation
- [ ] **Tests included** for new functionality
- [ ] **Documentation updated** for API changes
- [ ] **Code follows** project style guidelines
- [ ] **All tests pass** and coverage is maintained
- [ ] **No breaking changes** without major version bump

### PR Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code is well-commented and documented
- [ ] Tests pass locally
- [ ] Documentation updated if needed
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** on multiple platforms if needed
4. **Documentation review** for user-facing changes
5. **Final approval** and merge by maintainer

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some project-specific conventions:

#### Code Formatting

```python
# Use Black for automatic formatting
# Line length: 88 characters
# Quote style: Double quotes for strings

def parse_expression(expression: str, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Parse mathematical expression and return MathML.
    
    Args:
        expression: Mathematical expression to parse
        options: Optional parsing configuration
        
    Returns:
        MathML representation of the expression
        
    Raises:
        MathParseError: If parsing fails
        
    Example:
        >>> parse_expression("x^2 + 2x + 1")
        '<math><mrow>...</mrow></math>'
    """
    if options is None:
        options = {}
    
    # Implementation here
    return result
```

#### Type Hints

- **Always use type hints** for function parameters and return values
- **Use generic types** from `typing` module when appropriate
- **Document complex types** with examples

```python
from typing import Dict, List, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class MathExpression(Generic[T]):
    def __init__(self, value: T, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.value = value
        self.metadata = metadata or {}
```

#### Documentation Strings

Use **Google-style docstrings** for all public functions and classes:

```python
def optimize_expression(
    expression: str, 
    rules: Optional[List[str]] = None,
    max_iterations: int = 10
) -> OptimizationResult:
    """
    Optimize mathematical expression using algebraic rules.
    
    This function applies various algebraic simplification rules to reduce
    the complexity of mathematical expressions while preserving their meaning.
    
    Args:
        expression: The mathematical expression to optimize
        rules: List of optimization rules to apply. If None, uses default rules
        max_iterations: Maximum number of optimization passes
        
    Returns:
        OptimizationResult containing the optimized expression and metadata
        
    Raises:
        MathParseError: If the expression cannot be parsed
        OptimizationError: If optimization fails
        
    Example:
        >>> result = optimize_expression("x + 0")
        >>> result.optimized_expression
        'x'
        >>> result.applied_rules
        ['addition_identity']
        
    Note:
        The optimization process is deterministic and will always produce
        the same result for the same input and rules.
    """
```

#### Error Handling

- **Use custom exceptions** for different error types
- **Provide helpful error messages** with context
- **Include suggestions** for fixing common errors

```python
class MathParseError(Exception):
    """Exception raised when mathematical expression parsing fails."""
    
    def __init__(
        self, 
        message: str, 
        expression: str,
        position: Optional[int] = None,
        suggestions: Optional[List[str]] = None
    ) -> None:
        super().__init__(message)
        self.expression = expression
        self.position = position
        self.suggestions = suggestions or []
```

### Architecture Principles

#### Single Responsibility Principle
Each class and function should have a single, well-defined purpose:

```python
# Good: Single responsibility
class ExpressionValidator:
    """Validates mathematical expressions for syntax and semantics."""
    
    def validate_syntax(self, expression: str) -> ValidationResult:
        """Check if expression has valid syntax."""
        pass
    
    def validate_semantics(self, expression: str) -> ValidationResult:
        """Check if expression is mathematically meaningful."""
        pass

# Avoid: Multiple responsibilities
class ExpressionProcessor:  # Too broad
    def validate(self, expression: str) -> bool: pass
    def parse(self, expression: str) -> str: pass
    def optimize(self, expression: str) -> str: pass
    def format(self, expression: str) -> str: pass
```

#### Dependency Injection
Use dependency injection for better testability and flexibility:

```python
class MathMLParser:
    """Main parser class with injected dependencies."""
    
    def __init__(
        self,
        grammar: MathematicalGrammar,
        transformer: MathMLTransformer,
        validator: Optional[InputValidator] = None
    ) -> None:
        self.grammar = grammar
        self.transformer = transformer
        self.validator = validator or InputValidator()
```

#### Plugin Architecture
Design for extensibility with clear interfaces:

```python
from abc import ABC, abstractmethod

class OutputFormatter(ABC):
    """Abstract base class for output formatters."""
    
    @abstractmethod
    def format_expression(self, expression: str) -> str:
        """Format the mathematical expression."""
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Get the name of this format."""
        pass

class LaTeXFormatter(OutputFormatter):
    """Concrete implementation for LaTeX output."""
    
    def format_expression(self, expression: str) -> str:
        # Implementation here
        return formatted_expression
    
    def get_format_name(self) -> str:
        return "latex"
```

## Testing Requirements

### Test Coverage

- **Minimum 90% code coverage** for new contributions
- **100% coverage** for critical parsing logic
- **Integration tests** for major features
- **Performance tests** for optimization changes

### Test Structure

```python
import pytest
from mathml_parser import MathMLParser, MathParseError
from mathml_parser.core.exceptions import ValidationError

class TestMathMLParser:
    """Test suite for MathMLParser class."""
    
    @pytest.fixture
    def parser(self) -> MathMLParser:
        """Create parser instance for testing."""
        return MathMLParser(enable_validation=True, enable_metrics=True)
    
    @pytest.fixture
    def sample_expressions(self) -> List[str]:
        """Sample mathematical expressions for testing."""
        return [
            "x^2 + 2x + 1",
            "sin(π/2)",
            "∫₀¹ x² dx",
            "[1, 2; 3, 4]"
        ]
    
    def test_basic_parsing(self, parser: MathMLParser) -> None:
        """Test basic expression parsing."""
        expression = "x + y"
        result = parser.parse_safe(expression)
        
        assert result.success
        assert "<math>" in result.mathml
        assert result.error is None
    
    @pytest.mark.parametrize("expression,expected_features", [
        ("x + y", ["arithmetic"]),
        ("sin(x)", ["functions"]),
        ("α + β", ["greek_letters"]),
    ])
    def test_feature_detection(
        self, 
        parser: MathMLParser, 
        expression: str, 
        expected_features: List[str]
    ) -> None:
        """Test mathematical feature detection."""
        result = parser.parse_safe(expression)
        
        assert result.success
        for feature in expected_features:
            assert feature in result.metrics.features_used
    
    def test_error_handling(self, parser: MathMLParser) -> None:
        """Test error handling for invalid expressions."""
        invalid_expression = "x + + y"
        result = parser.parse_safe(invalid_expression)
        
        assert not result.success
        assert result.error is not None
        assert len(result.error.suggestions) > 0
    
    @pytest.mark.performance
    def test_parsing_performance(self, parser: MathMLParser) -> None:
        """Test parsing performance for complex expressions."""
        complex_expression = "∑(n=1 to ∞) ((-1)^(n+1))/(n^2) = π²/12"
        
        import time
        start_time = time.time()
        result = parser.parse_safe(complex_expression)
        parse_time = time.time() - start_time
        
        assert result.success
        assert parse_time < 0.1  # Should parse in under 100ms
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test speed and memory usage
5. **Regression Tests**: Prevent previously fixed bugs

## Documentation Guidelines

### API Documentation

- **Document all public APIs** with comprehensive docstrings
- **Include examples** for complex functions
- **Document exceptions** that may be raised
- **Specify type information** clearly

### User Documentation

- **Getting Started Guide**: Quick introduction for new users
- **Tutorials**: Step-by-step learning materials
- **API Reference**: Complete function and class documentation
- **Examples**: Real-world usage scenarios
- **FAQ**: Common questions and solutions

### Code Comments

```python
def _optimize_algebraic_identities(self, expression: str) -> str:
    """
    Apply algebraic identity optimizations to expression.
    
    This is an internal method that applies common algebraic identities
    such as x + 0 = x, x * 1 = x, etc. to simplify expressions.
    """
    # Handle addition identity: x + 0 = x
    expression = re.sub(r'(.+?)\s*\+\s*0\b', r'\1', expression)
    
    # Handle multiplication identity: x * 1 = x  
    expression = re.sub(r'(.+?)\s*\*\s*1\b', r'\1', expression)
    
    # Handle power identity: x^1 = x
    expression = re.sub(r'(.+?)\^1\b', r'\1', expression)
    
    return expression
```

## Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes or major feature additions
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes and minor improvements

### Release Checklist

1. **Update version numbers** in all relevant files
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** on multiple Python versions
4. **Update documentation** for any API changes
5. **Create release branch** and test thoroughly
6. **Tag release** with proper version number
7. **Deploy to PyPI** via automated pipeline
8. **Create GitHub release** with release notes

### Backward Compatibility

- **Maintain compatibility** within major versions
- **Deprecate features** before removing them
- **Provide migration guides** for breaking changes
- **Support previous minor versions** for critical fixes

---

## Getting Help

### Community Resources

- **GitHub Discussions**: General questions and community chat
- **Issue Tracker**: Bug reports and feature requests
- **Stack Overflow**: Tag questions with `mathml-parser`
- **Documentation**: Comprehensive guides and API reference

### Maintainer Contact

For urgent issues or security concerns, contact the maintainers directly through GitHub.

---

Thank you for contributing to MathML Parser! Your efforts help make mathematical computing more accessible to everyone.