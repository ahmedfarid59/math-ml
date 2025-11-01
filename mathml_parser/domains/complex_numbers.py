"""
Complex Number Processing
========================

This module provides specialized parsing and formatting for complex numbers
and complex analysis notation.

Features:
- Complex number arithmetic and formatting
- Polar and rectangular form conversion
- Complex analysis functions (arg, abs, conjugate)
- Support for complex mathematical constants
- Integration with multi-format output
"""

import re
import math
import cmath
from typing import Union, Tuple, Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ComplexNumber:
    """
    Representation of a complex number with enhanced functionality.
    
    Attributes:
        real: Real part of the complex number
        imag: Imaginary part of the complex number
        form: Display form ('rectangular' or 'polar')
    """
    real: float
    imag: float
    form: str = 'rectangular'
    
    def __post_init__(self):
        """Validate form parameter."""
        if self.form not in ['rectangular', 'polar']:
            raise ValueError("Form must be 'rectangular' or 'polar'")
    
    @property
    def magnitude(self) -> float:
        """Calculate magnitude (modulus) of complex number."""
        return math.sqrt(self.real**2 + self.imag**2)
    
    @property
    def argument(self) -> float:
        """Calculate argument (phase) of complex number in radians."""
        return math.atan2(self.imag, self.real)
    
    @property
    def argument_degrees(self) -> float:
        """Calculate argument (phase) of complex number in degrees."""
        return math.degrees(self.argument)
    
    @property
    def conjugate(self) -> 'ComplexNumber':
        """Return complex conjugate."""
        return ComplexNumber(self.real, -self.imag, self.form)
    
    def to_polar(self) -> 'ComplexNumber':
        """Convert to polar form representation."""
        r = self.magnitude
        theta = self.argument
        return ComplexNumber(r, theta, 'polar')
    
    def to_rectangular(self) -> 'ComplexNumber':
        """Convert to rectangular form representation."""
        if self.form == 'polar':
            # self.real is r, self.imag is theta
            real = self.real * math.cos(self.imag)
            imag = self.real * math.sin(self.imag)
            return ComplexNumber(real, imag, 'rectangular')
        return self
    
    def __add__(self, other: Union['ComplexNumber', float, int]) -> 'ComplexNumber':
        """Addition with complex numbers or real numbers."""
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imag, self.form)
        
        # Convert both to rectangular for addition
        a = self.to_rectangular()
        b = other.to_rectangular()
        return ComplexNumber(a.real + b.real, a.imag + b.imag)
    
    def __mul__(self, other: Union['ComplexNumber', float, int]) -> 'ComplexNumber':
        """Multiplication with complex numbers or real numbers."""
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imag * other, self.form)
        
        # Convert to rectangular for multiplication
        a = self.to_rectangular()
        b = other.to_rectangular()
        
        real = a.real * b.real - a.imag * b.imag
        imag = a.real * b.imag + a.imag * b.real
        return ComplexNumber(real, imag)
    
    def __str__(self) -> str:
        """String representation based on form."""
        if self.form == 'polar':
            return f"{self.real:.3f}∠{math.degrees(self.imag):.1f}°"
        else:
            if self.imag >= 0:
                return f"{self.real:.3f} + {self.imag:.3f}i"
            else:
                return f"{self.real:.3f} - {abs(self.imag):.3f}i"


class ComplexNumberProcessor:
    """
    Processor for complex number expressions and operations.
    
    This class handles parsing, evaluation, and formatting of complex
    number expressions with support for various notations and operations.
    """
    
    # Complex number patterns for parsing
    COMPLEX_PATTERNS = {
        # Rectangular form: a + bi, a - bi
        'rectangular': [
            r'([+-]?\d*\.?\d+)\s*([+-])\s*(\d*\.?\d+)[ij]',
            r'([+-]?\d*\.?\d+)[ij]',  # Pure imaginary
            r'([+-]?\d*\.?\d+)',      # Pure real
        ],
        
        # Polar form: r∠θ, r∠θ°, r cis θ
        'polar': [
            r'(\d*\.?\d+)∠([+-]?\d*\.?\d+)°?',
            r'(\d*\.?\d+)\s*cis\s*([+-]?\d*\.?\d+)',
            r'(\d*\.?\d+)\s*∠\s*([+-]?\d*\.?\d+)',
        ],
        
        # Exponential form: re^(iθ)
        'exponential': [
            r'(\d*\.?\d+)\s*e\^?\(?i\*?([+-]?\d*\.?\d+)\)?',
            r'(\d*\.?\d+)\s*exp\(i\*?([+-]?\d*\.?\d+)\)',
        ]
    }
    
    # Complex mathematical constants
    COMPLEX_CONSTANTS = {
        'i': ComplexNumber(0, 1),
        'j': ComplexNumber(0, 1),  # Engineering notation
        '1+i': ComplexNumber(1, 1),
        '1-i': ComplexNumber(1, -1),
        '-i': ComplexNumber(0, -1),
        '-j': ComplexNumber(0, -1),
    }
    
    # Complex functions
    COMPLEX_FUNCTIONS = {
        'abs', 'arg', 'conj', 'real', 'imag', 'polar', 'rect',
        'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh'
    }
    
    def __init__(self):
        """Initialize the complex number processor."""
        self.default_precision = 6
        self.angle_unit = 'radians'  # or 'degrees'
        self.default_form = 'rectangular'
    
    def parse_complex(self, expression: str) -> Optional[ComplexNumber]:
        """
        Parse a complex number from string expression.
        
        Args:
            expression: String representation of complex number
            
        Returns:
            ComplexNumber object if parsing successful, None otherwise
            
        Example:
            >>> processor = ComplexNumberProcessor()
            >>> z = processor.parse_complex("3 + 4i")
            >>> print(z)
            3.000 + 4.000i
        """
        expression = expression.strip().replace(' ', '')
        
        # Try rectangular form
        for pattern in self.COMPLEX_PATTERNS['rectangular']:
            match = re.match(pattern, expression)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:  # a + bi form
                    real = float(groups[0])
                    sign = 1 if groups[1] == '+' else -1
                    imag = sign * float(groups[2] or '1')
                    return ComplexNumber(real, imag)
                
                elif len(groups) == 1:
                    if 'i' in expression or 'j' in expression:
                        # Pure imaginary
                        imag = float(groups[0].replace('i', '').replace('j', '') or '1')
                        return ComplexNumber(0, imag)
                    else:
                        # Pure real
                        real = float(groups[0])
                        return ComplexNumber(real, 0)
        
        # Try polar form
        for pattern in self.COMPLEX_PATTERNS['polar']:
            match = re.match(pattern, expression)
            if match:
                r = float(match.group(1))
                theta = float(match.group(2))
                
                # Convert degrees to radians if needed
                if '°' in expression:
                    theta = math.radians(theta)
                
                return ComplexNumber(r, theta, 'polar')
        
        # Try exponential form
        for pattern in self.COMPLEX_PATTERNS['exponential']:
            match = re.match(pattern, expression)
            if match:
                r = float(match.group(1))
                theta = float(match.group(2))
                return ComplexNumber(r, theta, 'polar')
        
        # Check for constants
        if expression.lower() in self.COMPLEX_CONSTANTS:
            return self.COMPLEX_CONSTANTS[expression.lower()]
        
        return None
    
    def evaluate_complex_expression(self, expression: str) -> Optional[ComplexNumber]:
        """
        Evaluate complex mathematical expressions.
        
        Args:
            expression: Mathematical expression involving complex numbers
            
        Returns:
            Result as ComplexNumber object
            
        Example:
            >>> result = processor.evaluate_complex_expression("(3+4i) * (1-2i)")
            >>> print(result)
            11.000 - 2.000i
        """
        # This is a simplified evaluator - real implementation would use
        # proper expression parsing and evaluation
        
        # Handle basic operations between two complex numbers
        operators = ['+', '-', '*', '/']
        for op in operators:
            if op in expression:
                parts = expression.split(op, 1)
                if len(parts) == 2:
                    left = self.parse_complex(parts[0].strip())
                    right = self.parse_complex(parts[1].strip())
                    
                    if left and right:
                        if op == '+':
                            return left + right
                        elif op == '-':
                            return left + (-1 * right)
                        elif op == '*':
                            return left * right
                        elif op == '/':
                            # Complex division: (a+bi)/(c+di) = ((a+bi)(c-di))/((c+di)(c-di))
                            conjugate = right.conjugate
                            numerator = left * conjugate
                            denominator = right * conjugate
                            return ComplexNumber(
                                numerator.real / denominator.real,
                                numerator.imag / denominator.real
                            )
        
        # Single complex number
        return self.parse_complex(expression)
    
    def format_complex(self, z: ComplexNumber, format_type: str = 'standard') -> str:
        """
        Format complex number for different output types.
        
        Args:
            z: ComplexNumber to format
            format_type: Output format ('standard', 'latex', 'html', 'ascii')
            
        Returns:
            Formatted string representation
        """
        if format_type == 'latex':
            return self._format_latex(z)
        elif format_type == 'html':
            return self._format_html(z)
        elif format_type == 'ascii':
            return self._format_ascii(z)
        else:
            return str(z)
    
    def _format_latex(self, z: ComplexNumber) -> str:
        """Format complex number for LaTeX output."""
        if z.form == 'polar':
            theta_deg = math.degrees(z.imag)
            return f"{z.real:.3f} \\angle {theta_deg:.1f}°"
        else:
            if abs(z.imag) < 1e-10:  # Essentially real
                return f"{z.real:.3f}"
            elif abs(z.real) < 1e-10:  # Essentially imaginary
                if abs(z.imag - 1) < 1e-10:
                    return "i"
                elif abs(z.imag + 1) < 1e-10:
                    return "-i"
                else:
                    return f"{z.imag:.3f}i"
            else:  # General complex number
                if z.imag >= 0:
                    return f"{z.real:.3f} + {z.imag:.3f}i"
                else:
                    return f"{z.real:.3f} - {abs(z.imag):.3f}i"
    
    def _format_html(self, z: ComplexNumber) -> str:
        """Format complex number for HTML output."""
        if z.form == 'polar':
            theta_deg = math.degrees(z.imag)
            return f'<span class="complex-polar">{z.real:.3f}∠{theta_deg:.1f}°</span>'
        else:
            if abs(z.imag) < 1e-10:
                return f'<span class="complex-real">{z.real:.3f}</span>'
            elif abs(z.real) < 1e-10:
                return f'<span class="complex-imag">{z.imag:.3f}<em>i</em></span>'
            else:
                sign = '+' if z.imag >= 0 else '-'
                return (f'<span class="complex-number">'
                       f'{z.real:.3f} {sign} {abs(z.imag):.3f}<em>i</em>'
                       f'</span>')
    
    def _format_ascii(self, z: ComplexNumber) -> str:
        """Format complex number for ASCII output."""
        if z.form == 'polar':
            theta_deg = math.degrees(z.imag)
            return f"{z.real:.3f} angle {theta_deg:.1f} deg"
        else:
            if abs(z.imag) < 1e-10:
                return f"{z.real:.3f}"
            elif abs(z.real) < 1e-10:
                return f"{z.imag:.3f}i"
            else:
                sign = '+' if z.imag >= 0 else '-'
                return f"{z.real:.3f} {sign} {abs(z.imag):.3f}i"
    
    def complex_functions(self, func_name: str, z: ComplexNumber) -> ComplexNumber:
        """
        Apply complex mathematical functions.
        
        Args:
            func_name: Name of the function to apply
            z: Complex number argument
            
        Returns:
            Result of applying function to z
        """
        z_rect = z.to_rectangular()
        c = complex(z_rect.real, z_rect.imag)
        
        if func_name == 'abs':
            return ComplexNumber(abs(c), 0)
        elif func_name == 'arg':
            return ComplexNumber(cmath.phase(c), 0)
        elif func_name == 'conj':
            return z.conjugate
        elif func_name == 'real':
            return ComplexNumber(z_rect.real, 0)
        elif func_name == 'imag':
            return ComplexNumber(z_rect.imag, 0)
        elif func_name == 'exp':
            result = cmath.exp(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'log':
            result = cmath.log(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'sqrt':
            result = cmath.sqrt(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'sin':
            result = cmath.sin(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'cos':
            result = cmath.cos(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'tan':
            result = cmath.tan(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'sinh':
            result = cmath.sinh(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'cosh':
            result = cmath.cosh(c)
            return ComplexNumber(result.real, result.imag)
        elif func_name == 'tanh':
            result = cmath.tanh(c)
            return ComplexNumber(result.real, result.imag)
        else:
            raise ValueError(f"Unknown complex function: {func_name}")
    
    def roots_of_unity(self, n: int) -> List[ComplexNumber]:
        """
        Calculate the nth roots of unity.
        
        Args:
            n: Order of roots (positive integer)
            
        Returns:
            List of n complex numbers representing nth roots of unity
        """
        roots = []
        for k in range(n):
            theta = 2 * math.pi * k / n
            z = ComplexNumber(math.cos(theta), math.sin(theta))
            roots.append(z)
        return roots
    
    def solve_quadratic_complex(self, a: float, b: float, c: float) -> Tuple[ComplexNumber, ComplexNumber]:
        """
        Solve quadratic equation ax² + bx + c = 0 allowing complex solutions.
        
        Args:
            a, b, c: Coefficients of quadratic equation
            
        Returns:
            Tuple of two complex solutions
        """
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            # Real solutions
            sqrt_disc = math.sqrt(discriminant)
            x1 = ComplexNumber((-b + sqrt_disc) / (2*a), 0)
            x2 = ComplexNumber((-b - sqrt_disc) / (2*a), 0)
        else:
            # Complex solutions
            sqrt_disc = math.sqrt(-discriminant)
            real_part = -b / (2*a)
            imag_part = sqrt_disc / (2*a)
            x1 = ComplexNumber(real_part, imag_part)
            x2 = ComplexNumber(real_part, -imag_part)
        
        return x1, x2
    
    def complex_plane_distance(self, z1: ComplexNumber, z2: ComplexNumber) -> float:
        """
        Calculate distance between two points in the complex plane.
        
        Args:
            z1, z2: Complex numbers representing points
            
        Returns:
            Euclidean distance between the points
        """
        z1_rect = z1.to_rectangular()
        z2_rect = z2.to_rectangular()
        
        return math.sqrt((z1_rect.real - z2_rect.real)**2 + 
                        (z1_rect.imag - z2_rect.imag)**2)
    
    def mandelbrot_iteration(self, c: ComplexNumber, max_iter: int = 100) -> int:
        """
        Perform Mandelbrot set iteration for a given complex number.
        
        Args:
            c: Complex number to test
            max_iter: Maximum number of iterations
            
        Returns:
            Number of iterations before divergence (or max_iter if bounded)
        """
        z = ComplexNumber(0, 0)
        
        for i in range(max_iter):
            if z.magnitude > 2:
                return i
            z = z * z + c
        
        return max_iter
    
    def configure(self, **kwargs):
        """
        Configure processor settings.
        
        Args:
            precision: Number of decimal places for display
            angle_unit: 'radians' or 'degrees' for angle measurements
            default_form: 'rectangular' or 'polar' for default display
        """
        if 'precision' in kwargs:
            self.default_precision = kwargs['precision']
        if 'angle_unit' in kwargs:
            if kwargs['angle_unit'] in ['radians', 'degrees']:
                self.angle_unit = kwargs['angle_unit']
        if 'default_form' in kwargs:
            if kwargs['default_form'] in ['rectangular', 'polar']:
                self.default_form = kwargs['default_form']


# Convenience functions for direct use
def parse_complex(expression: str) -> Optional[ComplexNumber]:
    """Parse complex number from string."""
    processor = ComplexNumberProcessor()
    return processor.parse_complex(expression)


def format_complex(z: ComplexNumber, format_type: str = 'standard') -> str:
    """Format complex number for output."""
    processor = ComplexNumberProcessor()
    return processor.format_complex(z, format_type)


def complex_add(z1: ComplexNumber, z2: ComplexNumber) -> ComplexNumber:
    """Add two complex numbers."""
    return z1 + z2


def complex_multiply(z1: ComplexNumber, z2: ComplexNumber) -> ComplexNumber:
    """Multiply two complex numbers."""
    return z1 * z2


# Integration with multi-format renderer
COMPLEX_FORMAT_EXTENSIONS = {
    'latex': lambda z: ComplexNumberProcessor()._format_latex(z),
    'html': lambda z: ComplexNumberProcessor()._format_html(z),
    'ascii': lambda z: ComplexNumberProcessor()._format_ascii(z)
}