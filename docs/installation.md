# Installation Guide

This guide covers installation of MathML Parser for different use cases and environments.

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 512MB RAM (2GB+ recommended for large expressions)
- **Disk Space**: ~50MB for base installation

## Quick Installation

For most users, the simple pip installation is recommended:

```bash
pip install mathml-parser
```

This installs the core MathML Parser package with all basic dependencies.

## Installation Options

### Standard Installation

```bash
# Install from PyPI
pip install mathml-parser

# Verify installation
python -c "import mathml_parser; print('Installation successful!')"
```

### Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/mathml-parser/mathml-parser.git
cd mathml-parser

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Optional Dependencies

Install with specific feature sets:

```bash
# Web interface support
pip install "mathml-parser[web]"

# Advanced mathematical computation
pip install "mathml-parser[math]"

# Documentation building
pip install "mathml-parser[docs]"

# Testing and development
pip install "mathml-parser[dev]"

# Complete installation with all features
pip install "mathml-parser[all]"
```

## Platform-Specific Instructions

### Windows

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)
2. **Open Command Prompt** or PowerShell
3. **Install MathML Parser**:
   ```cmd
   pip install mathml-parser
   ```

#### Windows Subsystem for Linux (WSL)

If using WSL, follow the Linux instructions below.

### macOS

1. **Install Python 3.8+** using Homebrew:
   ```bash
   brew install python
   ```
2. **Install MathML Parser**:
   ```bash
   pip3 install mathml-parser
   ```

### Linux

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip

# Install MathML Parser
pip3 install mathml-parser
```

#### CentOS/RHEL/Fedora

```bash
# Install Python and pip
sudo dnf install python3 python3-pip

# Install MathML Parser
pip3 install mathml-parser
```

#### Arch Linux

```bash
# Install Python and pip
sudo pacman -S python python-pip

# Install MathML Parser
pip install mathml-parser
```

## Virtual Environment Setup

Using virtual environments is highly recommended:

### Using venv

```bash
# Create virtual environment
python -m venv mathml-env

# Activate virtual environment
# On Windows:
mathml-env\Scripts\activate
# On macOS/Linux:
source mathml-env/bin/activate

# Install MathML Parser
pip install mathml-parser
```

### Using conda

```bash
# Create conda environment
conda create -n mathml-env python=3.9

# Activate environment
conda activate mathml-env

# Install MathML Parser
pip install mathml-parser
```

### Using pipenv

```bash
# Install pipenv if not available
pip install pipenv

# Create Pipfile and install
pipenv install mathml-parser

# Activate shell
pipenv shell
```

## Docker Installation

Run MathML Parser in a Docker container:

```bash
# Pull the official image
docker pull mathmlparser/mathml-parser:latest

# Run interactive container
docker run -it mathmlparser/mathml-parser:latest python

# Mount local directory for file processing
docker run -v $(pwd):/workspace mathmlparser/mathml-parser:latest \
    python -c "import mathml_parser; print('Docker setup complete!')"
```

### Building from Dockerfile

```bash
# Clone repository
git clone https://github.com/mathml-parser/mathml-parser.git
cd mathml-parser

# Build Docker image
docker build -t mathml-parser .

# Run container
docker run -it mathml-parser
```

## Verification

### Basic Verification

```python
import mathml_parser

# Check version
print(f"MathML Parser version: {mathml_parser.__version__}")

# Test basic functionality
parser = mathml_parser.MathMLParser()
result = parser.parse("x^2 + 1")
print(f"Parsing test: {result}")
```

### Comprehensive Test

```python
from mathml_parser import MathMLParser, render_expression

# Test parsing
parser = MathMLParser()
expressions = [
    "x^2 + 2*x + 1",
    "sin(π/4)",
    "∫(x^2, 0, 1)",
    "[1,2;3,4]"
]

for expr in expressions:
    try:
        parsed = parser.parse(expr)
        print(f"✓ Parsed: {expr}")
    except Exception as e:
        print(f"✗ Failed: {expr} - {e}")

# Test multi-format rendering
test_expr = "x^2 + 1"
formats = ['latex', 'ascii', 'html']

for fmt in formats:
    try:
        result = render_expression(test_expr, fmt)
        print(f"✓ {fmt}: {result}")
    except Exception as e:
        print(f"✗ {fmt} failed: {e}")
```

## Troubleshooting

### Common Issues

#### Import Error

```
ImportError: No module named 'mathml_parser'
```

**Solutions**:
- Verify installation: `pip list | grep mathml-parser`
- Check Python path: `python -c "import sys; print(sys.path)"`
- Reinstall: `pip uninstall mathml-parser && pip install mathml-parser`

#### Permission Errors (Linux/macOS)

```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:
- Use user installation: `pip install --user mathml-parser`
- Use virtual environment (recommended)
- Use sudo (not recommended): `sudo pip install mathml-parser`

#### Dependency Conflicts

```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions**:
- Create fresh virtual environment
- Update pip: `pip install --upgrade pip`
- Use dependency resolver: `pip install --use-feature=2020-resolver mathml-parser`

#### Windows Path Issues

```
'python' is not recognized as an internal or external command
```

**Solutions**:
- Add Python to PATH during installation
- Use full path: `C:\Python39\python.exe -m pip install mathml-parser`
- Use Python Launcher: `py -m pip install mathml-parser`

### Getting Help

If you encounter issues not covered here:

1. **Check the [Troubleshooting Guide](../advanced/troubleshooting.md)**
2. **Search [GitHub Issues](https://github.com/mathml-parser/mathml-parser/issues)**
3. **Create a new issue** with:
   - Operating system and version
   - Python version
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. **[Quick Start Tutorial](quickstart.md)** - Learn basic usage
2. **[Examples](examples.md)** - See practical applications
3. **[User Guide](user_guide/index.md)** - Comprehensive documentation
4. **[API Reference](api/index.md)** - Detailed API documentation

## Performance Optimization

For optimal performance:

### System Dependencies

Install optimized numerical libraries:

```bash
# NumPy with optimized BLAS
pip install numpy[blas]

# SciPy for advanced mathematical functions
pip install scipy

# SymPy for symbolic computation
pip install sympy
```

### Memory Configuration

For processing large mathematical expressions:

```python
import mathml_parser

# Configure parser with optimizations
parser = mathml_parser.MathMLParser(
    cache_size=10000,
    parallel_processing=True,
    memory_limit="2GB"
)
```

### Benchmarking

Test performance on your system:

```python
from mathml_parser.benchmarks import run_performance_test

# Run standard benchmark suite
results = run_performance_test()
print(f"Parsing speed: {results['expressions_per_second']} expr/sec")
print(f"Memory usage: {results['peak_memory_mb']} MB")
```