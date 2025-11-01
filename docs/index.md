# MathML Parser Documentation

Welcome to MathML Parser's comprehensive documentation!

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart
   examples
```

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/index
   user_guide/parsing
   user_guide/multi_format
   user_guide/optimization
   user_guide/cli
```

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index
   api/core
   api/parsers
   api/transformers
   api/formats
```

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   
   advanced/index
   advanced/expressions
   advanced/extensions
   advanced/performance
   advanced/troubleshooting
```

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Development
   
   development/index
   development/contributing
   development/architecture
   development/testing
   development/plugins
```

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Appendices
   
   appendices/changelog
   appendices/license
   appendices/bibliography
```

## Overview

MathML Parser is a powerful Python library for parsing, transforming, and rendering mathematical expressions. It provides:

- **Multi-format Support**: Convert between MathML, LaTeX, ASCII Math, HTML, SVG, and plain text
- **Advanced Parsing**: Robust parsing of complex mathematical notation using Lark grammar
- **Expression Optimization**: Algebraic simplification and optimization capabilities
- **Extensible Architecture**: Plugin system for custom formatters and transformations
- **Educational Tools**: Step-by-step solutions and learning modules
- **Web Integration**: Interactive web interface for real-time previews

## Key Features

### 🔧 Comprehensive Mathematical Support
- Basic arithmetic and algebraic expressions
- Advanced calculus notation (integrals, derivatives, limits)
- Linear algebra (matrices, vectors, determinants)
- Complex numbers and functions
- Set theory and logic notation
- Statistics and probability expressions

### 📐 Multiple Output Formats
- **LaTeX**: Publication-quality mathematical typesetting
- **MathML**: Standard XML-based mathematical markup
- **ASCII Math**: Plain text mathematical notation
- **HTML + CSS**: Web-friendly mathematical display
- **SVG**: Scalable vector graphics for diagrams
- **Plain Text**: Simple text representation

### ⚡ High Performance
- Efficient parsing with Lark parser
- Caching for repeated expressions
- Parallel processing support
- Memory-optimized operations

### 🎓 Educational Features
- Step-by-step solution generation
- Mathematical proof validation
- Interactive learning modules
- Expression simplification hints

## Quick Example

```python
from mathml_parser import MathMLParser, render_expression

# Create parser instance
parser = MathMLParser()

# Parse mathematical expression
expression = "x^2 + 2*x + 1"
parsed = parser.parse(expression)

# Convert to different formats
latex_output = render_expression(expression, 'latex')
print(latex_output)  # $x^{2} + 2 \cdot x + 1$

html_output = render_expression(expression, 'html')
ascii_output = render_expression(expression, 'ascii')

# Get all formats at once
all_formats = parser.render_all_formats(expression)
for format_name, result in all_formats.items():
    print(f"{format_name}: {result}")
```

## Installation

Install MathML Parser using pip:

```bash
pip install mathml-parser
```

For development installation:

```bash
git clone https://github.com/mathml-parser/mathml-parser.git
cd mathml-parser
pip install -e ".[dev]"
```

## Community and Support

- **GitHub**: [mathml-parser/mathml-parser](https://github.com/mathml-parser/mathml-parser)
- **Documentation**: [mathml-parser.readthedocs.io](https://mathml-parser.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/mathml-parser/mathml-parser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mathml-parser/mathml-parser/discussions)

## License

MathML Parser is released under the MIT License. See the [License](appendices/license.md) page for details.

## Acknowledgments

We thank the mathematical computing community and the developers of:
- [Lark](https://lark-parser.readthedocs.io/) for the excellent parsing framework
- [SymPy](https://www.sympy.org/) for mathematical computation inspiration
- [MathJax](https://www.mathjax.org/) for mathematical rendering insights

---

*Get started with the [Installation](installation.md) guide or dive into the [Quick Start](quickstart.md) tutorial!*