# MathML Parser Documentation

This directory contains the comprehensive documentation for MathML Parser built with Sphinx.

## Structure

```
docs/
├── _static/           # Static files (CSS, images, etc.)
├── _templates/        # Custom Sphinx templates
├── api/              # API reference documentation
├── user_guide/       # User guide and tutorials
├── advanced/         # Advanced topics
├── development/      # Development documentation
├── appendices/       # Additional resources
├── conf.py           # Sphinx configuration
├── index.md          # Main documentation index
├── installation.md   # Installation guide
├── quickstart.md     # Quick start tutorial
├── examples.md       # Examples and use cases
├── Makefile          # Unix build commands
├── make.bat          # Windows build commands
└── requirements.txt  # Documentation dependencies
```

## Building the Documentation

### Prerequisites

1. **Install Python 3.8+**
2. **Install documentation dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Build Commands

#### HTML Documentation (Recommended)

```bash
# Unix/Linux/macOS
make html

# Windows
make.bat html
```

The built documentation will be in `_build/html/index.html`.

#### Live Development Server

For development with automatic rebuilding:

```bash
# Unix/Linux/macOS
make livehtml

# Windows
make.bat livehtml
```

This starts a development server at `http://localhost:8000` with automatic rebuilding when files change.

#### Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Check for broken links
make linkcheck

# Clean build directory
make clean
```

### Viewing Locally

After building HTML documentation:

```bash
# Serve locally
make serve

# Or manually
cd _build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Documentation Structure

### Main Sections

1. **Getting Started**
   - Installation guide
   - Quick start tutorial
   - Basic examples

2. **User Guide**
   - Comprehensive usage documentation
   - Feature explanations
   - Best practices

3. **API Reference**
   - Complete API documentation
   - Class and function references
   - Type annotations

4. **Advanced Topics**
   - Performance optimization
   - Custom extensions
   - Troubleshooting

5. **Development**
   - Contributing guidelines
   - Architecture overview
   - Plugin development

### File Organization

- **Markdown files (.md)**: Main content using MyST parser
- **reStructuredText files (.rst)**: Legacy format, still supported
- **API docs**: Auto-generated from docstrings
- **Static assets**: CSS, images, JavaScript in `_static/`

## Writing Documentation

### Markdown with MyST

We use MyST (Markedly Structured Text) for enhanced Markdown:

```markdown
# Standard Markdown heading

```{note}
This is a note admonition.
```

```{code-block} python
:linenos:
:emphasize-lines: 2,3

def example_function():
    # This line is highlighted
    return "Hello, World!"
```

## Cross-references

Link to [other sections](installation.md) or {doc}`api/core`.

## Mathematical Notation

Inline math: $x^2 + y^2 = z^2$

Display math:
$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$
```

### Code Examples

Use consistent code examples:

```python
from mathml_parser import MathMLParser

# Create parser
parser = MathMLParser()

# Parse expression
result = parser.parse("x^2 + 1")
print(result)
```

### API Documentation

API docs are auto-generated from docstrings. Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> str:
    """
    Brief description of the function.
    
    Longer description with more details about what the function does,
    its purpose, and any important notes.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.
        
    Returns:
        Description of the return value.
        
    Raises:
        ValueError: Description of when this exception is raised.
        
    Example:
        >>> result = example_function("hello", 42)
        >>> print(result)
        'hello_42'
    """
    return f"{param1}_{param2}"
```

## Sphinx Configuration

Key configuration options in `conf.py`:

- **Extensions**: Sphinx extensions for enhanced functionality
- **Theme**: ReadTheDocs theme with customizations
- **Math support**: MathJax for mathematical notation
- **API generation**: Automatic API documentation
- **Cross-references**: Inter-document linking

## Custom Styling

Custom CSS is in `_static/custom.css`:

- Mathematical expression styling
- Code block enhancements
- API documentation formatting
- Responsive design improvements

## Contributing to Documentation

### Guidelines

1. **Use clear, concise language**
2. **Include practical examples**
3. **Test all code examples**
4. **Keep sections focused and well-organized**
5. **Use consistent formatting**

### Adding New Content

1. **Create new `.md` file** in appropriate directory
2. **Add to table of contents** in relevant `index.md`
3. **Include cross-references** to related sections
4. **Add examples and use cases**
5. **Test locally** before submitting

### Review Process

1. **Build documentation locally** to check for errors
2. **Review all links** and cross-references
3. **Test code examples** for accuracy
4. **Check spelling and grammar**
5. **Verify formatting** in browser

## Deployment

### GitHub Pages

Documentation is automatically deployed to GitHub Pages via GitHub Actions:

1. **Push to main branch** triggers build
2. **Sphinx builds** HTML documentation
3. **GitHub Pages** serves from `gh-pages` branch

### Read the Docs

Alternative deployment to Read the Docs:

1. **Connect repository** to Read the Docs
2. **Configure build** settings
3. **Auto-deploy** on commits

### Manual Deployment

For other hosting platforms:

1. **Build HTML documentation**:
   ```bash
   make html
   ```

2. **Upload `_build/html/` contents** to web server

3. **Ensure proper MIME types** for .html files

## Troubleshooting

### Common Issues

#### Build Errors

```bash
# Check for syntax errors
make clean
make html

# Check specific file
sphinx-build -b html . _build/html filename.md
```

#### Missing Dependencies

```bash
# Install missing packages
pip install -r requirements.txt

# Update packages
pip install -U -r requirements.txt
```

#### Link Errors

```bash
# Check for broken links
make linkcheck
```

#### Math Rendering Issues

- **Check MathJax configuration** in `conf.py`
- **Verify math syntax** (use LaTeX format)
- **Test in different browsers**

### Getting Help

1. **Check Sphinx documentation**: https://www.sphinx-doc.org/
2. **Review MyST parser docs**: https://myst-parser.readthedocs.io/
3. **Search GitHub issues** for similar problems
4. **Ask in project discussions** for help

## Performance

### Build Optimization

- **Use incremental builds** for development
- **Exclude unnecessary files** in `conf.py`
- **Optimize images** in `_static/`
- **Use caching** for API documentation

### Large Projects

For large documentation projects:

- **Split into multiple files**
- **Use external TOC** files
- **Implement lazy loading**
- **Consider build parallelization**

## Maintenance

### Regular Tasks

1. **Update dependencies** in `requirements.txt`
2. **Check for broken links** monthly
3. **Review and update** outdated content
4. **Test examples** with new releases
5. **Update version numbers** in `conf.py`

### Version Management

- **Tag documentation versions** with releases
- **Maintain compatibility** with supported Python versions
- **Archive old versions** if needed

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [ReadTheDocs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [MathJax Documentation](https://docs.mathjax.org/)