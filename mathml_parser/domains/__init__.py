"""
Advanced Mathematical Domains
============================

This package provides specialized processors for different mathematical domains
including complex numbers, differential equations, probability theory, and set theory.

Each domain processor extends the core MathML parsing capabilities with
domain-specific notation, operations, and formatting options.

Available processors:
- ComplexNumberProcessor: Complex arithmetic and analysis
- DifferentialEquationProcessor: ODE/PDE parsing and solution methods
- ProbabilityProcessor: Statistical distributions and Bayesian inference
- SetTheoryProcessor: Set operations and logical expressions
"""

# Import all domain processors and main classes
from .complex_numbers import (
    ComplexNumberProcessor, 
    ComplexNumber, 
    parse_complex, 
    format_complex,
    complex_add,
    complex_multiply
)

from .differential_equations import (
    DifferentialEquationProcessor, 
    DifferentialEquation, 
    BoundaryCondition,
    EquationType,
    BoundaryType,
    parse_differential_equation,
    solve_ode_analytically,
    get_numerical_solution
)

from .probability import (
    ProbabilityProcessor, 
    ProbabilityDistribution, 
    StatisticalTest, 
    BayesianInference,
    DistributionType,
    TestType,
    parse_probability_distribution,
    calculate_normal_probability,
    binomial_probability,
    poisson_probability
)

from .set_theory import (
    SetTheoryProcessor, 
    MathematicalSet, 
    LogicalExpression, 
    MathematicalProof,
    QuantifiedStatement,
    SetOperation,
    LogicalOperator,
    Quantifier,
    ProofType,
    parse_set_notation,
    create_set_from_list,
    set_union,
    set_intersection,
    parse_logic_expression
)

# Export all public classes and functions
__all__ = [
    # Complex Numbers
    'ComplexNumberProcessor',
    'ComplexNumber',
    'parse_complex',
    'format_complex',
    'complex_add',
    'complex_multiply',
    
    # Differential Equations
    'DifferentialEquationProcessor',
    'DifferentialEquation',
    'BoundaryCondition',
    'EquationType',
    'BoundaryType',
    'parse_differential_equation',
    'solve_ode_analytically',
    'get_numerical_solution',
    
    # Probability & Statistics
    'ProbabilityProcessor',
    'ProbabilityDistribution',
    'StatisticalTest',
    'BayesianInference',
    'DistributionType',
    'TestType',
    'parse_probability_distribution',
    'calculate_normal_probability',
    'binomial_probability',
    'poisson_probability',
    
    # Set Theory & Logic
    'SetTheoryProcessor',
    'MathematicalSet',
    'LogicalExpression',
    'MathematicalProof',
    'QuantifiedStatement',
    'SetOperation',
    'LogicalOperator',
    'Quantifier',
    'ProofType',
    'parse_set_notation',
    'create_set_from_list',
    'set_union',
    'set_intersection',
    'parse_logic_expression',
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "MathML Parser Team"
__description__ = "Advanced mathematical domain processors for specialized notation and operations"

# Domain processor registry for easy access
DOMAIN_PROCESSORS = {
    'complex': ComplexNumberProcessor,
    'differential': DifferentialEquationProcessor,
    'probability': ProbabilityProcessor,
    'sets': SetTheoryProcessor,
}

def get_processor(domain: str):
    """
    Get a domain processor by name.
    
    Args:
        domain: Name of domain ('complex', 'differential', 'probability', 'sets')
        
    Returns:
        Initialized processor instance
        
    Example:
        >>> complex_proc = get_processor('complex')
        >>> z = complex_proc.parse_complex("3 + 4i")
    """
    if domain not in DOMAIN_PROCESSORS:
        available = ', '.join(DOMAIN_PROCESSORS.keys())
        raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
    
    return DOMAIN_PROCESSORS[domain]()

def list_domains() -> list:
    """
    List all available mathematical domains.
    
    Returns:
        List of domain names
    """
    return list(DOMAIN_PROCESSORS.keys())

def demonstrate_domains():
    """
    Demonstrate capabilities of all domain processors.
    
    This function shows example usage of each mathematical domain processor
    with sample inputs and formatted outputs.
    """
    print("=== Mathematical Domains Demonstration ===\n")
    
    # Complex Numbers
    print("1. Complex Numbers:")
    complex_proc = ComplexNumberProcessor()
    z1 = complex_proc.parse_complex("3 + 4i")
    z2 = complex_proc.parse_complex("1 - 2i")
    if z1 and z2:
        result = z1 * z2
        print(f"   (3 + 4i) × (1 - 2i) = {result}")
        print(f"   LaTeX: {complex_proc.format_complex(result, 'latex')}")
        print(f"   Magnitude |z1| = {z1.magnitude:.3f}")
        print(f"   Argument arg(z1) = {z1.argument_degrees:.1f}°")
    print()
    
    # Differential Equations  
    print("2. Differential Equations:")
    de_proc = DifferentialEquationProcessor()
    ode = de_proc.parse_ode("dy/dx = x + y")
    if ode:
        print(f"   Equation: {ode.equation}")
        print(f"   Type: {ode.equation_type.value}")
        print(f"   Linearity: {ode.linearity}")
        methods = de_proc.suggest_solution_method(ode)
        print(f"   Suggested methods: {', '.join(methods[:3])}")
    print()
    
    # Probability & Statistics
    print("3. Probability & Statistics:")
    prob_proc = ProbabilityProcessor()
    normal_dist = prob_proc.parse_distribution("X ~ N(0, 1)")
    if normal_dist:
        print(f"   Distribution: {prob_proc.format_distribution(normal_dist)}")
        print(f"   Mean: {normal_dist.properties.get('mean', 'N/A')}")
        print(f"   Variance: {normal_dist.properties.get('variance', 'N/A')}")
        print(f"   PDF formula: {normal_dist.pdf_formula()}")
    
    # Hypothesis test example
    t_test = prob_proc.create_hypothesis_test(TestType.T_TEST_ONE_SAMPLE, 0.0, "two-sided")
    print(f"   Hypothesis test: {t_test.null_hypothesis}")
    print(f"                   {t_test.alternative_hypothesis}")
    print()
    
    # Set Theory & Logic
    print("4. Set Theory & Logic:")
    set_proc = SetTheoryProcessor()
    set_a = set_proc.parse_set("{1, 2, 3, 4}")
    set_b = set_proc.parse_set("{3, 4, 5, 6}")
    if set_a and set_b:
        union_set = set_a.union(set_b)
        intersection_set = set_a.intersection(set_b)
        print(f"   A = {set_proc.format_set(set_a)}")
        print(f"   B = {set_proc.format_set(set_b)}")
        print(f"   A ∪ B = {set_proc.format_set(union_set)}")
        print(f"   A ∩ B = {set_proc.format_set(intersection_set)}")
    
    # Logical expression
    logic_expr = set_proc.parse_logical_expression("p ∧ q → r")
    if logic_expr:
        print(f"   Logic: {logic_expr.expression}")
        print(f"   Variables: {logic_expr.variables}")
        if logic_expr.truth_table:
            print(f"   Tautology: {logic_expr.is_tautology}")
    print()
    
    print("All domain processors loaded successfully!")
    print(f"Available domains: {', '.join(list_domains())}")

# Integration hooks for multi-format renderer
def get_format_extensions():
    """
    Get format extensions from all domain processors.
    
    Returns:
        Dictionary of format extensions for multi-format rendering
    """
    extensions = {}
    
    # Import format extensions from each domain
    try:
        from .complex_numbers import COMPLEX_FORMAT_EXTENSIONS
        extensions.update(COMPLEX_FORMAT_EXTENSIONS)
    except ImportError:
        pass
    
    try:
        from .differential_equations import DE_FORMAT_EXTENSIONS
        extensions.update(DE_FORMAT_EXTENSIONS)
    except ImportError:
        pass
    
    try:
        from .probability import PROBABILITY_FORMAT_EXTENSIONS
        extensions.update(PROBABILITY_FORMAT_EXTENSIONS)
    except ImportError:
        pass
    
    try:
        from .set_theory import SET_THEORY_FORMAT_EXTENSIONS
        extensions.update(SET_THEORY_FORMAT_EXTENSIONS)
    except ImportError:
        pass
    
    return extensions