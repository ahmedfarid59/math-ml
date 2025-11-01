"""
Setup configuration for the Mathematical Expression to MathML Parser package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A comprehensive mathematical expression to MathML parser"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['lark>=1.1.0']

setup(
    name="mathml-parser",
    version="2.0.0",
    description="A comprehensive mathematical expression to MathML parser with advanced features",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="MathML Parser Team",
    author_email="support@mathml-parser.com",
    url="https://github.com/mathml-parser/mathml-parser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Text Processing :: Markup :: XML",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "mathematics", "mathml", "parser", "expressions", "algebra",
        "calculus", "equations", "xml", "mathematical notation",
        "greek letters", "matrices", "functions", "trigonometry"
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "performance": [
            "cython>=0.29.0",
            "numpy>=1.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mathml-parse=mathml_parser.cli:main",
        ],
    },
    package_data={
        "mathml_parser": [
            "examples/*.py",
            "examples/*.md",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/mathml-parser/mathml-parser/issues",
        "Source": "https://github.com/mathml-parser/mathml-parser",
        "Documentation": "https://mathml-parser.readthedocs.io/",
        "Changelog": "https://github.com/mathml-parser/mathml-parser/blob/main/CHANGELOG.md",
    },
)