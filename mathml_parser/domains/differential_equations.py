"""
Differential Equations Processing
================================

This module provides specialized parsing and formatting for differential
equations, boundary conditions, and solution methods.

Features:
- ODE and PDE notation parsing
- Boundary and initial condition handling
- Solution method representation
- LaPlace transform support
- Numerical method integration
- Multi-format output for equations
"""

import re
import math
import sympy as sp
from typing import Union, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class EquationType(Enum):
    """Types of differential equations."""
    ODE_FIRST_ORDER = "ode_first"
    ODE_SECOND_ORDER = "ode_second"
    ODE_HIGHER_ORDER = "ode_higher"
    PDE_PARABOLIC = "pde_parabolic"
    PDE_ELLIPTIC = "pde_elliptic"
    PDE_HYPERBOLIC = "pde_hyperbolic"
    SYSTEM_ODE = "system_ode"
    SYSTEM_PDE = "system_pde"


class BoundaryType(Enum):
    """Types of boundary conditions."""
    DIRICHLET = "dirichlet"  # u(x₀) = value
    NEUMANN = "neumann"      # u'(x₀) = value
    ROBIN = "robin"          # au(x₀) + bu'(x₀) = value
    PERIODIC = "periodic"    # u(a) = u(b), u'(a) = u'(b)


@dataclass
class BoundaryCondition:
    """
    Represents a boundary or initial condition.
    
    Attributes:
        condition_type: Type of boundary condition
        variable: Variable the condition applies to
        location: Location where condition applies (e.g., x=0, t=0)
        value: Value or expression for the condition
        coefficient_a: Coefficient for Robin conditions (au + bu' = c)
        coefficient_b: Coefficient for Robin conditions
    """
    condition_type: BoundaryType
    variable: str
    location: str
    value: Union[str, float]
    coefficient_a: float = 1.0
    coefficient_b: float = 0.0
    
    def __str__(self) -> str:
        """String representation of boundary condition."""
        if self.condition_type == BoundaryType.DIRICHLET:
            return f"{self.variable}({self.location}) = {self.value}"
        elif self.condition_type == BoundaryType.NEUMANN:
            return f"∂{self.variable}/∂x({self.location}) = {self.value}"
        elif self.condition_type == BoundaryType.ROBIN:
            return f"{self.coefficient_a}{self.variable}({self.location}) + {self.coefficient_b}∂{self.variable}/∂x({self.location}) = {self.value}"
        elif self.condition_type == BoundaryType.PERIODIC:
            return f"{self.variable} is periodic"
        return f"Condition: {self.condition_type}"


@dataclass
class DifferentialEquation:
    """
    Represents a differential equation with metadata.
    
    Attributes:
        equation: The differential equation as string or sympy expression
        equation_type: Type classification of the equation
        variables: List of dependent variables
        independent_vars: List of independent variables
        order: Order of the equation (highest derivative)
        linearity: Whether equation is linear or nonlinear
        homogeneous: Whether equation is homogeneous
        boundary_conditions: List of boundary/initial conditions
        analytical_solution: Known analytical solution if available
        numerical_methods: Applicable numerical solution methods
    """
    equation: Union[str, sp.Expr]
    equation_type: EquationType
    variables: List[str] = field(default_factory=list)
    independent_vars: List[str] = field(default_factory=list)
    order: int = 1
    linearity: str = "unknown"  # "linear", "nonlinear", "unknown"
    homogeneous: bool = False
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    analytical_solution: Optional[Union[str, sp.Expr]] = None
    numerical_methods: List[str] = field(default_factory=list)
    
    def add_boundary_condition(self, condition: BoundaryCondition):
        """Add a boundary condition to the equation."""
        self.boundary_conditions.append(condition)
    
    def is_well_posed(self) -> bool:
        """Check if equation has sufficient boundary conditions."""
        # Simplified check - real implementation would be more sophisticated
        if self.equation_type in [EquationType.ODE_FIRST_ORDER]:
            return len(self.boundary_conditions) >= 1
        elif self.equation_type in [EquationType.ODE_SECOND_ORDER]:
            return len(self.boundary_conditions) >= 2
        elif "pde" in self.equation_type.value:
            # PDEs need boundary conditions on all boundaries
            return len(self.boundary_conditions) >= 2
        return False


class DifferentialEquationProcessor:
    """
    Processor for differential equation expressions and analysis.
    
    This class handles parsing, classification, and formatting of differential
    equations with support for various notations and solution methods.
    """
    
    # Common ODE patterns for parsing
    ODE_PATTERNS = {
        # First order: dy/dx = f(x,y), y' = f(x,y)
        'first_order': [
            r"d([a-zA-Z])/d([a-zA-Z])\s*=\s*(.+)",
            r"([a-zA-Z])'\s*=\s*(.+)",
            r"([a-zA-Z])₁\s*=\s*(.+)",
        ],
        
        # Second order: d²y/dx² = f(x,y,y'), y'' = f(x,y,y')
        'second_order': [
            r"d²([a-zA-Z])/d([a-zA-Z])²\s*=\s*(.+)",
            r"([a-zA-Z])''\s*=\s*(.+)",
            r"([a-zA-Z])₂\s*=\s*(.+)",
        ],
        
        # Higher order: d³y/dx³, y''', etc.
        'higher_order': [
            r"d([0-9]+)([a-zA-Z])/d([a-zA-Z])\^?([0-9]+)\s*=\s*(.+)",
            r"([a-zA-Z])(\'+)\s*=\s*(.+)",
        ]
    }
    
    # PDE patterns
    PDE_PATTERNS = {
        # Heat equation: ∂u/∂t = α∂²u/∂x²
        'heat': [
            r"∂([a-zA-Z])/∂([a-zA-Z])\s*=\s*([a-zA-Z0-9]*)\s*∂²([a-zA-Z])/∂([a-zA-Z])²",
            r"([a-zA-Z])_([a-zA-Z])\s*=\s*([a-zA-Z0-9]*)\s*([a-zA-Z])_([a-zA-Z])([a-zA-Z])",
        ],
        
        # Wave equation: ∂²u/∂t² = c²∂²u/∂x²
        'wave': [
            r"∂²([a-zA-Z])/∂([a-zA-Z])²\s*=\s*([a-zA-Z0-9]*)\s*∂²([a-zA-Z])/∂([a-zA-Z])²",
            r"([a-zA-Z])_([a-zA-Z])([a-zA-Z])\s*=\s*([a-zA-Z0-9]*)\s*([a-zA-Z])_([a-zA-Z])([a-zA-Z])",
        ],
        
        # Laplace equation: ∂²u/∂x² + ∂²u/∂y² = 0
        'laplace': [
            r"∂²([a-zA-Z])/∂([a-zA-Z])²\s*\+\s*∂²([a-zA-Z])/∂([a-zA-Z])²\s*=\s*0",
            r"∇²([a-zA-Z])\s*=\s*0",
            r"Δ([a-zA-Z])\s*=\s*0",
        ]
    }
    
    # Boundary condition patterns
    BOUNDARY_PATTERNS = {
        'dirichlet': [
            r"([a-zA-Z])\(([^)]+)\)\s*=\s*(.+)",
            r"([a-zA-Z])\|_{([^}]+)}\s*=\s*(.+)",
        ],
        'neumann': [
            r"∂([a-zA-Z])/∂([a-zA-Z])\|_{([^}]+)}\s*=\s*(.+)",
            r"([a-zA-Z])'?\(([^)]+)\)\s*=\s*(.+)",
        ],
        'robin': [
            r"([0-9.]+)([a-zA-Z])\(([^)]+)\)\s*\+\s*([0-9.]+)∂([a-zA-Z])/∂([a-zA-Z])\|_{([^}]+)}\s*=\s*(.+)",
        ]
    }
    
    # Known solution methods
    SOLUTION_METHODS = {
        'analytical': [
            'separation_of_variables',
            'characteristic_equation',
            'laplace_transform',
            'fourier_transform',
            'green_function',
            'series_solution',
            'exact_equation',
            'integrating_factor',
            'substitution',
            'undetermined_coefficients',
            'variation_of_parameters'
        ],
        'numerical': [
            'euler_method',
            'runge_kutta',
            'finite_difference',
            'finite_element',
            'spectral_method',
            'monte_carlo',
            'shooting_method',
            'boundary_value_method'
        ]
    }
    
    def __init__(self):
        """Initialize the differential equation processor."""
        self.use_sympy = True
        self.numerical_precision = 6
        self.symbolic_variables = {}
        
        # Initialize sympy symbols for common variables
        if self.use_sympy:
            self.x, self.y, self.z = sp.symbols('x y z')
            self.t = sp.symbols('t')
            self.u, self.v, self.w = sp.symbols('u v w', cls=sp.Function)
    
    def parse_ode(self, equation_str: str) -> Optional[DifferentialEquation]:
        """
        Parse an ordinary differential equation.
        
        Args:
            equation_str: String representation of ODE
            
        Returns:
            DifferentialEquation object if parsing successful
            
        Example:
            >>> processor = DifferentialEquationProcessor()
            >>> eq = processor.parse_ode("dy/dx = x + y")
            >>> print(eq.equation_type)
            EquationType.ODE_FIRST_ORDER
        """
        equation_str = equation_str.strip()
        
        # Try first order patterns
        for pattern in self.ODE_PATTERNS['first_order']:
            match = re.match(pattern, equation_str)
            if match:
                if 'd' in pattern:  # dy/dx form
                    dep_var = match.group(1)
                    indep_var = match.group(2)
                    rhs = match.group(3)
                else:  # y' form
                    dep_var = match.group(1)
                    rhs = match.group(2)
                    indep_var = 'x'  # default
                
                equation = DifferentialEquation(
                    equation=equation_str,
                    equation_type=EquationType.ODE_FIRST_ORDER,
                    variables=[dep_var],
                    independent_vars=[indep_var],
                    order=1
                )
                
                # Analyze linearity and homogeneity
                equation.linearity = self._analyze_linearity(rhs, dep_var)
                equation.homogeneous = self._analyze_homogeneity(rhs, dep_var)
                
                return equation
        
        # Try second order patterns
        for pattern in self.ODE_PATTERNS['second_order']:
            match = re.match(pattern, equation_str)
            if match:
                if 'd²' in pattern:
                    dep_var = match.group(1)
                    indep_var = match.group(2)
                    rhs = match.group(3)
                else:
                    dep_var = match.group(1)
                    rhs = match.group(2)
                    indep_var = 'x'
                
                equation = DifferentialEquation(
                    equation=equation_str,
                    equation_type=EquationType.ODE_SECOND_ORDER,
                    variables=[dep_var],
                    independent_vars=[indep_var],
                    order=2
                )
                
                equation.linearity = self._analyze_linearity(rhs, dep_var)
                equation.homogeneous = self._analyze_homogeneity(rhs, dep_var)
                
                return equation
        
        return None
    
    def parse_pde(self, equation_str: str) -> Optional[DifferentialEquation]:
        """
        Parse a partial differential equation.
        
        Args:
            equation_str: String representation of PDE
            
        Returns:
            DifferentialEquation object if parsing successful
        """
        equation_str = equation_str.strip()
        
        # Try heat equation
        for pattern in self.PDE_PATTERNS['heat']:
            match = re.match(pattern, equation_str)
            if match:
                equation = DifferentialEquation(
                    equation=equation_str,
                    equation_type=EquationType.PDE_PARABOLIC,
                    variables=[match.group(1)],
                    independent_vars=['t', 'x'],
                    order=2,
                    linearity="linear"
                )
                equation.numerical_methods = ['finite_difference', 'finite_element']
                return equation
        
        # Try wave equation
        for pattern in self.PDE_PATTERNS['wave']:
            match = re.match(pattern, equation_str)
            if match:
                equation = DifferentialEquation(
                    equation=equation_str,
                    equation_type=EquationType.PDE_HYPERBOLIC,
                    variables=[match.group(1)],
                    independent_vars=['t', 'x'],
                    order=2,
                    linearity="linear"
                )
                equation.numerical_methods = ['finite_difference', 'characteristic_method']
                return equation
        
        # Try Laplace equation
        for pattern in self.PDE_PATTERNS['laplace']:
            match = re.match(pattern, equation_str)
            if match:
                equation = DifferentialEquation(
                    equation=equation_str,
                    equation_type=EquationType.PDE_ELLIPTIC,
                    variables=[match.group(1)],
                    independent_vars=['x', 'y'],
                    order=2,
                    linearity="linear",
                    homogeneous=True
                )
                equation.numerical_methods = ['finite_difference', 'finite_element']
                return equation
        
        return None
    
    def parse_boundary_condition(self, condition_str: str) -> Optional[BoundaryCondition]:
        """
        Parse boundary condition from string.
        
        Args:
            condition_str: String representation of boundary condition
            
        Returns:
            BoundaryCondition object if parsing successful
        """
        condition_str = condition_str.strip()
        
        # Try Dirichlet conditions
        for pattern in self.BOUNDARY_PATTERNS['dirichlet']:
            match = re.match(pattern, condition_str)
            if match:
                return BoundaryCondition(
                    condition_type=BoundaryType.DIRICHLET,
                    variable=match.group(1),
                    location=match.group(2),
                    value=match.group(3)
                )
        
        # Try Neumann conditions
        for pattern in self.BOUNDARY_PATTERNS['neumann']:
            match = re.match(pattern, condition_str)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    return BoundaryCondition(
                        condition_type=BoundaryType.NEUMANN,
                        variable=groups[0],
                        location=groups[-2],
                        value=groups[-1]
                    )
        
        # Try Robin conditions
        for pattern in self.BOUNDARY_PATTERNS['robin']:
            match = re.match(pattern, condition_str)
            if match:
                return BoundaryCondition(
                    condition_type=BoundaryType.ROBIN,
                    variable=match.group(2),
                    location=match.group(3),
                    value=match.group(8),
                    coefficient_a=float(match.group(1)),
                    coefficient_b=float(match.group(4))
                )
        
        return None
    
    def _analyze_linearity(self, expression: str, variable: str) -> str:
        """Analyze if expression is linear in the given variable."""
        # Simplified analysis - checks for nonlinear terms
        nonlinear_patterns = [
            f"{variable}\\^[2-9]",  # y^2, y^3, etc.
            f"{variable}\\*{variable}",  # y*y
            f"sin\\({variable}\\)",  # sin(y)
            f"cos\\({variable}\\)",  # cos(y)
            f"exp\\({variable}\\)",  # exp(y)
            f"log\\({variable}\\)",  # log(y)
        ]
        
        for pattern in nonlinear_patterns:
            if re.search(pattern, expression):
                return "nonlinear"
        
        return "linear"
    
    def _analyze_homogeneity(self, expression: str, variable: str) -> bool:
        """Analyze if expression represents homogeneous equation."""
        # Check if RHS contains only terms with the dependent variable
        # Simplified: if no constant terms or independent variable terms
        if re.search(r'\b[0-9]+\b', expression) and not re.search(variable, expression):
            return False  # Contains constant terms without dependent variable
        return True
    
    def suggest_solution_method(self, equation: DifferentialEquation) -> List[str]:
        """
        Suggest appropriate solution methods for the equation.
        
        Args:
            equation: DifferentialEquation to analyze
            
        Returns:
            List of suggested solution method names
        """
        methods = []
        
        if equation.equation_type == EquationType.ODE_FIRST_ORDER:
            if equation.linearity == "linear" and equation.homogeneous:
                methods.extend(['separation_of_variables', 'integrating_factor'])
            elif equation.linearity == "linear":
                methods.append('integrating_factor')
            else:
                methods.extend(['separation_of_variables', 'substitution'])
        
        elif equation.equation_type == EquationType.ODE_SECOND_ORDER:
            if equation.linearity == "linear" and equation.homogeneous:
                methods.append('characteristic_equation')
            elif equation.linearity == "linear":
                methods.extend(['undetermined_coefficients', 'variation_of_parameters'])
            else:
                methods.extend(['series_solution', 'numerical_methods'])
        
        elif "pde" in equation.equation_type.value:
            if equation.linearity == "linear":
                methods.extend(['separation_of_variables', 'fourier_transform', 'laplace_transform'])
            methods.extend(['finite_difference', 'finite_element'])
        
        # Always suggest numerical methods as backup
        methods.extend(['runge_kutta', 'finite_difference'])
        
        return list(set(methods))  # Remove duplicates
    
    def format_equation(self, equation: DifferentialEquation, format_type: str = 'standard') -> str:
        """
        Format differential equation for different output types.
        
        Args:
            equation: DifferentialEquation to format
            format_type: Output format ('standard', 'latex', 'html', 'ascii')
            
        Returns:
            Formatted string representation
        """
        if format_type == 'latex':
            return self._format_latex(equation)
        elif format_type == 'html':
            return self._format_html(equation)
        elif format_type == 'ascii':
            return self._format_ascii(equation)
        else:
            return str(equation.equation)
    
    def _format_latex(self, equation: DifferentialEquation) -> str:
        """Format equation for LaTeX output."""
        eq_str = str(equation.equation)
        
        # Convert common notation to LaTeX
        eq_str = re.sub(r"d([a-zA-Z])/d([a-zA-Z])", r"\\frac{d\1}{d\2}", eq_str)
        eq_str = re.sub(r"d²([a-zA-Z])/d([a-zA-Z])²", r"\\frac{d^2\1}{d\2^2}", eq_str)
        eq_str = re.sub(r"∂([a-zA-Z])/∂([a-zA-Z])", r"\\frac{\\partial \1}{\\partial \2}", eq_str)
        eq_str = re.sub(r"∂²([a-zA-Z])/∂([a-zA-Z])²", r"\\frac{\\partial^2 \1}{\\partial \2^2}", eq_str)
        eq_str = re.sub(r"([a-zA-Z])'", r"\1'", eq_str)
        eq_str = re.sub(r"([a-zA-Z])''", r"\1''", eq_str)
        
        return eq_str
    
    def _format_html(self, equation: DifferentialEquation) -> str:
        """Format equation for HTML output."""
        eq_str = str(equation.equation)
        
        # Convert to HTML with MathML or MathJax notation
        eq_str = re.sub(r"d([a-zA-Z])/d([a-zA-Z])", r"d\1/d\2", eq_str)
        eq_str = re.sub(r"([a-zA-Z])'", r"\1'", eq_str)
        eq_str = re.sub(r"∂", r"&part;", eq_str)
        eq_str = re.sub(r"²", r"&sup2;", eq_str)
        
        return f'<span class="differential-equation">{eq_str}</span>'
    
    def _format_ascii(self, equation: DifferentialEquation) -> str:
        """Format equation for ASCII output."""
        eq_str = str(equation.equation)
        
        # Convert to ASCII-friendly notation
        eq_str = re.sub(r"∂", "d", eq_str)
        eq_str = re.sub(r"²", "^2", eq_str)
        eq_str = re.sub(r"₁", "_1", eq_str)
        eq_str = re.sub(r"₂", "_2", eq_str)
        
        return eq_str
    
    def solve_first_order_linear(self, equation: DifferentialEquation) -> Optional[str]:
        """
        Solve first-order linear ODE using integrating factor method.
        
        Args:
            equation: First-order linear ODE
            
        Returns:
            General solution as string if solvable
        """
        if (equation.equation_type != EquationType.ODE_FIRST_ORDER or 
            equation.linearity != "linear"):
            return None
        
        # This is a simplified implementation
        # Real implementation would parse the equation more thoroughly
        eq_str = str(equation.equation)
        
        # Pattern: dy/dx + P(x)y = Q(x)
        pattern = r"d([a-zA-Z])/d([a-zA-Z])\s*\+\s*(.+)\*([a-zA-Z])\s*=\s*(.+)"
        match = re.match(pattern, eq_str)
        
        if match:
            y_var = match.group(1)
            x_var = match.group(2)
            P_func = match.group(2)
            Q_func = match.group(4)
            
            return (f"Solution using integrating factor: "
                   f"{y_var} = (1/μ(x)) ∫ μ(x) * ({Q_func}) dx + C, "
                   f"where μ(x) = exp(∫ {P_func} dx)")
        
        return None
    
    def check_exact_equation(self, equation: DifferentialEquation) -> bool:
        """
        Check if first-order ODE is exact.
        
        Args:
            equation: First-order ODE to check
            
        Returns:
            True if equation is exact
        """
        # Simplified check for exact equations
        # Pattern: M(x,y)dx + N(x,y)dy = 0
        # Exact if ∂M/∂y = ∂N/∂x
        
        if equation.equation_type != EquationType.ODE_FIRST_ORDER:
            return False
        
        # This would require symbolic computation for proper implementation
        # For now, return False as placeholder
        return False
    
    def laplace_transform_solution(self, equation: DifferentialEquation) -> Optional[str]:
        """
        Attempt solution using Laplace transforms.
        
        Args:
            equation: ODE with constant coefficients
            
        Returns:
            Solution approach description
        """
        if not equation.boundary_conditions:
            return None
        
        return ("Laplace transform method applicable. "
                "Transform equation, solve algebraically, "
                "then inverse transform.")
    
    def separation_of_variables(self, equation: DifferentialEquation) -> Optional[str]:
        """
        Attempt solution using separation of variables.
        
        Args:
            equation: ODE that may be separable
            
        Returns:
            Solution approach if applicable
        """
        if equation.equation_type == EquationType.ODE_FIRST_ORDER:
            return ("Try separation: rearrange as g(y)dy = h(x)dx, "
                   "then integrate both sides.")
        elif "pde" in equation.equation_type.value:
            return ("Assume solution of form u(x,t) = X(x)T(t), "
                   "substitute and separate variables.")
        
        return None
    
    def generate_numerical_solution_code(self, equation: DifferentialEquation, 
                                       method: str = 'runge_kutta') -> str:
        """
        Generate Python code for numerical solution.
        
        Args:
            equation: Differential equation to solve
            method: Numerical method to use
            
        Returns:
            Python code string for numerical solution
        """
        if method == 'runge_kutta' and equation.equation_type == EquationType.ODE_FIRST_ORDER:
            return '''
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Right-hand side of dy/dx = f(x, y)"""
    # TODO: Implement based on specific equation
    return x + y  # Example

def runge_kutta_4th(f, x0, y0, h, n):
    """4th order Runge-Kutta method"""
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0], y[0] = x0, y0
    
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i+1] = x[i] + h
    
    return x, y

# Solve the equation
x_vals, y_vals = runge_kutta_4th(f, 0, 1, 0.1, 100)
plt.plot(x_vals, y_vals)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution')
plt.show()
'''
        
        elif method == 'finite_difference' and "pde" in equation.equation_type.value:
            return '''
import numpy as np
import matplotlib.pyplot as plt

# Example: Heat equation ∂u/∂t = α∂²u/∂x²
def solve_heat_equation(nx, nt, dx, dt, alpha):
    """Finite difference solution for heat equation"""
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = np.sin(np.pi * np.linspace(0, 1, nx))
    
    # Boundary conditions
    u[:, 0] = 0  # u(0,t) = 0
    u[:, -1] = 0  # u(1,t) = 0
    
    # Time stepping
    r = alpha * dt / dx**2
    for n in range(nt-1):
        for i in range(1, nx-1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
    return u

# Solve
solution = solve_heat_equation(50, 100, 0.02, 0.0001, 1.0)
plt.imshow(solution, aspect='auto', origin='lower')
plt.colorbar()
plt.title('Heat Equation Solution')
plt.show()
'''
        
        return f"# Numerical solution code for {method} not implemented yet"


# Convenience functions for direct use
def parse_differential_equation(equation_str: str) -> Optional[DifferentialEquation]:
    """Parse differential equation from string."""
    processor = DifferentialEquationProcessor()
    
    # Try ODE first
    result = processor.parse_ode(equation_str)
    if result:
        return result
    
    # Try PDE
    return processor.parse_pde(equation_str)


def solve_ode_analytically(equation: DifferentialEquation) -> Optional[str]:
    """Attempt analytical solution of ODE."""
    processor = DifferentialEquationProcessor()
    
    if equation.linearity == "linear":
        return processor.solve_first_order_linear(equation)
    
    return processor.separation_of_variables(equation)


def get_numerical_solution(equation: DifferentialEquation, method: str = 'runge_kutta') -> str:
    """Generate numerical solution code."""
    processor = DifferentialEquationProcessor()
    return processor.generate_numerical_solution_code(equation, method)


# Integration with multi-format renderer
DE_FORMAT_EXTENSIONS = {
    'latex': lambda eq: DifferentialEquationProcessor()._format_latex(eq),
    'html': lambda eq: DifferentialEquationProcessor()._format_html(eq),
    'ascii': lambda eq: DifferentialEquationProcessor()._format_ascii(eq)
}