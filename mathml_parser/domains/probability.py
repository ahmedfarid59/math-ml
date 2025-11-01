"""
Probability and Statistics Processing
====================================

This module provides specialized parsing and formatting for probability
distributions, statistical notation, and Bayesian inference expressions.

Features:
- Probability distribution notation (normal, binomial, Poisson, etc.)
- Statistical measure calculations (mean, variance, etc.)
- Bayesian inference notation
- Hypothesis testing representation
- Confidence interval formatting
- Multi-format output for statistical expressions
"""

import math
import re
from typing import Union, List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class DistributionType(Enum):
    """Types of probability distributions."""
    NORMAL = "normal"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    BETA = "beta"
    GAMMA = "gamma"
    CHI_SQUARE = "chi_square"
    T_DISTRIBUTION = "t_distribution"
    F_DISTRIBUTION = "f_distribution"
    BERNOULLI = "bernoulli"
    GEOMETRIC = "geometric"
    HYPERGEOMETRIC = "hypergeometric"
    MULTINOMIAL = "multinomial"


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST_ONE_SAMPLE = "t_test_one"
    T_TEST_TWO_SAMPLE = "t_test_two"
    CHI_SQUARE_GOODNESS = "chi_square_goodness"
    CHI_SQUARE_INDEPENDENCE = "chi_square_independence"
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    Z_TEST = "z_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"


@dataclass
class ProbabilityDistribution:
    """
    Represents a probability distribution with parameters.
    
    Attributes:
        distribution_type: Type of distribution
        parameters: Dictionary of distribution parameters
        variable: Random variable name
        support: Support of the distribution (domain)
        properties: Calculated properties (mean, variance, etc.)
    """
    distribution_type: DistributionType
    parameters: Dict[str, Union[float, int]] = field(default_factory=dict)
    variable: str = "X"
    support: Optional[Tuple[Union[float, int], Union[float, int]]] = None
    properties: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate distribution properties after initialization."""
        self._calculate_properties()
    
    def _calculate_properties(self):
        """Calculate mean, variance, and other properties."""
        if self.distribution_type == DistributionType.NORMAL:
            if 'mu' in self.parameters and 'sigma' in self.parameters:
                self.properties['mean'] = self.parameters['mu']
                self.properties['variance'] = self.parameters['sigma']**2
                self.properties['std'] = self.parameters['sigma']
                self.support = (-float('inf'), float('inf'))
        
        elif self.distribution_type == DistributionType.BINOMIAL:
            if 'n' in self.parameters and 'p' in self.parameters:
                n, p = self.parameters['n'], self.parameters['p']
                self.properties['mean'] = n * p
                self.properties['variance'] = n * p * (1 - p)
                self.properties['std'] = math.sqrt(self.properties['variance'])
                self.support = (0, int(n))
        
        elif self.distribution_type == DistributionType.POISSON:
            if 'lambda' in self.parameters:
                lam = self.parameters['lambda']
                self.properties['mean'] = lam
                self.properties['variance'] = lam
                self.properties['std'] = math.sqrt(lam)
                self.support = (0, float('inf'))
        
        elif self.distribution_type == DistributionType.EXPONENTIAL:
            if 'lambda' in self.parameters:
                lam = self.parameters['lambda']
                self.properties['mean'] = 1 / lam
                self.properties['variance'] = 1 / (lam**2)
                self.properties['std'] = 1 / lam
                self.support = (0, float('inf'))
        
        elif self.distribution_type == DistributionType.UNIFORM:
            if 'a' in self.parameters and 'b' in self.parameters:
                a, b = self.parameters['a'], self.parameters['b']
                self.properties['mean'] = (a + b) / 2
                self.properties['variance'] = (b - a)**2 / 12
                self.properties['std'] = math.sqrt(self.properties['variance'])
                self.support = (a, b)
    
    def pdf_formula(self) -> str:
        """Return the probability density function formula."""
        if self.distribution_type == DistributionType.NORMAL:
            return f"f(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)"
        elif self.distribution_type == DistributionType.BINOMIAL:
            return f"P(X=k) = C(n,k) * p^k * (1-p)^(n-k)"
        elif self.distribution_type == DistributionType.POISSON:
            return f"P(X=k) = (λ^k * e^(-λ)) / k!"
        elif self.distribution_type == DistributionType.EXPONENTIAL:
            return f"f(x) = λ * e^(-λx)"
        elif self.distribution_type == DistributionType.UNIFORM:
            return f"f(x) = 1/(b-a) for a ≤ x ≤ b"
        return "PDF formula not available"
    
    def cdf_formula(self) -> str:
        """Return the cumulative distribution function formula."""
        if self.distribution_type == DistributionType.NORMAL:
            return f"F(x) = Φ((x-μ)/σ)"
        elif self.distribution_type == DistributionType.EXPONENTIAL:
            return f"F(x) = 1 - e^(-λx)"
        elif self.distribution_type == DistributionType.UNIFORM:
            return f"F(x) = (x-a)/(b-a) for a ≤ x ≤ b"
        return "CDF formula not available"


@dataclass
class StatisticalTest:
    """
    Represents a statistical hypothesis test.
    
    Attributes:
        test_type: Type of statistical test
        null_hypothesis: Null hypothesis statement
        alternative_hypothesis: Alternative hypothesis statement
        significance_level: Alpha level (e.g., 0.05)
        test_statistic: Calculated test statistic value
        p_value: Calculated p-value
        critical_value: Critical value for decision
        conclusion: Test conclusion
    """
    test_type: TestType
    null_hypothesis: str
    alternative_hypothesis: str
    significance_level: float = 0.05
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    critical_value: Optional[float] = None
    conclusion: Optional[str] = None
    
    def make_decision(self) -> str:
        """Make statistical decision based on test results."""
        if self.p_value is not None:
            if self.p_value < self.significance_level:
                return f"Reject H₀ (p-value = {self.p_value:.4f} < α = {self.significance_level})"
            else:
                return f"Fail to reject H₀ (p-value = {self.p_value:.4f} ≥ α = {self.significance_level})"
        
        if self.test_statistic is not None and self.critical_value is not None:
            if abs(self.test_statistic) > abs(self.critical_value):
                return f"Reject H₀ (|test statistic| = {abs(self.test_statistic):.4f} > critical value = {abs(self.critical_value):.4f})"
            else:
                return f"Fail to reject H₀ (|test statistic| = {abs(self.test_statistic):.4f} ≤ critical value = {abs(self.critical_value):.4f})"
        
        return "Insufficient information for decision"


@dataclass
class BayesianInference:
    """
    Represents Bayesian inference components.
    
    Attributes:
        prior: Prior distribution
        likelihood: Likelihood function description
        posterior: Posterior distribution
        evidence: Evidence (marginal likelihood)
        prior_parameters: Parameters of prior distribution
        posterior_parameters: Parameters of posterior distribution
    """
    prior: str
    likelihood: str
    posterior: Optional[str] = None
    evidence: Optional[str] = None
    prior_parameters: Dict[str, float] = field(default_factory=dict)
    posterior_parameters: Dict[str, float] = field(default_factory=dict)
    
    def bayes_theorem(self) -> str:
        """Return Bayes' theorem formula."""
        return "P(θ|data) = P(data|θ) × P(θ) / P(data)"


class ProbabilityProcessor:
    """
    Processor for probability and statistics expressions.
    
    This class handles parsing, calculation, and formatting of probability
    distributions, statistical tests, and Bayesian inference notation.
    """
    
    # Distribution notation patterns
    DISTRIBUTION_PATTERNS = {
        'normal': [
            r"([A-Z])\s*~\s*N\(([^,]+),\s*([^)]+)\)",
            r"([A-Z])\s*~\s*Normal\(([^,]+),\s*([^)]+)\)",
            r"([A-Z])\s*follows\s*normal\s*distribution\s*with\s*μ\s*=\s*([^,]+),\s*σ²?\s*=\s*(.+)",
        ],
        'binomial': [
            r"([A-Z])\s*~\s*Bin\(([^,]+),\s*([^)]+)\)",
            r"([A-Z])\s*~\s*Binomial\(([^,]+),\s*([^)]+)\)",
            r"([A-Z])\s*~\s*B\(([^,]+),\s*([^)]+)\)",
        ],
        'poisson': [
            r"([A-Z])\s*~\s*Poisson\(([^)]+)\)",
            r"([A-Z])\s*~\s*Pois\(([^)]+)\)",
        ],
        'exponential': [
            r"([A-Z])\s*~\s*Exp\(([^)]+)\)",
            r"([A-Z])\s*~\s*Exponential\(([^)]+)\)",
        ],
        'uniform': [
            r"([A-Z])\s*~\s*U\(([^,]+),\s*([^)]+)\)",
            r"([A-Z])\s*~\s*Uniform\(([^,]+),\s*([^)]+)\)",
        ]
    }
    
    # Probability notation patterns
    PROBABILITY_PATTERNS = {
        'basic': [
            r"P\(([^)]+)\)",
            r"Pr\(([^)]+)\)",
            r"prob\(([^)]+)\)",
        ],
        'conditional': [
            r"P\(([^|]+)\|([^)]+)\)",
            r"Pr\(([^|]+)\|([^)]+)\)",
        ],
        'joint': [
            r"P\(([^,]+),\s*([^)]+)\)",
            r"P\(([^∩]+)∩([^)]+)\)",
        ]
    }
    
    # Statistical measure patterns
    STATISTICAL_PATTERNS = {
        'expectation': [
            r"E\[([^]]+)\]",
            r"𝔼\[([^]]+)\]",
            r"expected\s*value\s*of\s*([A-Za-z]+)",
        ],
        'variance': [
            r"Var\(([^)]+)\)",
            r"V\(([^)]+)\)",
            r"variance\s*of\s*([A-Za-z]+)",
        ],
        'covariance': [
            r"Cov\(([^,]+),\s*([^)]+)\)",
            r"covariance\s*of\s*([A-Za-z]+)\s*and\s*([A-Za-z]+)",
        ],
        'correlation': [
            r"Corr\(([^,]+),\s*([^)]+)\)",
            r"ρ\(([^,]+),\s*([^)]+)\)",
        ]
    }
    
    # Hypothesis test patterns
    HYPOTHESIS_PATTERNS = {
        't_test': [
            r"H₀:\s*μ\s*=\s*([0-9.]+)",
            r"H₁:\s*μ\s*≠\s*([0-9.]+)",
            r"H₁:\s*μ\s*>\s*([0-9.]+)",
            r"H₁:\s*μ\s*<\s*([0-9.]+)",
        ],
        'proportion': [
            r"H₀:\s*p\s*=\s*([0-9.]+)",
            r"H₁:\s*p\s*≠\s*([0-9.]+)",
        ]
    }
    
    def __init__(self):
        """Initialize the probability processor."""
        self.precision = 4
        self.confidence_level = 0.95
        self.use_unicode = True
        
        # Statistical constants
        self.z_critical_values = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        self.t_critical_values = {
            # Simplified - real implementation would have full t-table
            (10, 0.95): 2.228,
            (20, 0.95): 2.086,
            (30, 0.95): 2.042,
        }
    
    def parse_distribution(self, distribution_str: str) -> Optional[ProbabilityDistribution]:
        """
        Parse probability distribution from string notation.
        
        Args:
            distribution_str: String representation of distribution
            
        Returns:
            ProbabilityDistribution object if parsing successful
            
        Example:
            >>> processor = ProbabilityProcessor()
            >>> dist = processor.parse_distribution("X ~ N(0, 1)")
            >>> print(dist.distribution_type)
            DistributionType.NORMAL
        """
        distribution_str = distribution_str.strip()
        
        # Try normal distribution
        for pattern in self.DISTRIBUTION_PATTERNS['normal']:
            match = re.match(pattern, distribution_str)
            if match:
                variable = match.group(1)
                mu = float(match.group(2))
                sigma = float(match.group(3))
                
                return ProbabilityDistribution(
                    distribution_type=DistributionType.NORMAL,
                    parameters={'mu': mu, 'sigma': sigma},
                    variable=variable
                )
        
        # Try binomial distribution
        for pattern in self.DISTRIBUTION_PATTERNS['binomial']:
            match = re.match(pattern, distribution_str)
            if match:
                variable = match.group(1)
                n = int(match.group(2))
                p = float(match.group(3))
                
                return ProbabilityDistribution(
                    distribution_type=DistributionType.BINOMIAL,
                    parameters={'n': n, 'p': p},
                    variable=variable
                )
        
        # Try Poisson distribution
        for pattern in self.DISTRIBUTION_PATTERNS['poisson']:
            match = re.match(pattern, distribution_str)
            if match:
                variable = match.group(1)
                lam = float(match.group(2))
                
                return ProbabilityDistribution(
                    distribution_type=DistributionType.POISSON,
                    parameters={'lambda': lam},
                    variable=variable
                )
        
        # Try exponential distribution
        for pattern in self.DISTRIBUTION_PATTERNS['exponential']:
            match = re.match(pattern, distribution_str)
            if match:
                variable = match.group(1)
                lam = float(match.group(2))
                
                return ProbabilityDistribution(
                    distribution_type=DistributionType.EXPONENTIAL,
                    parameters={'lambda': lam},
                    variable=variable
                )
        
        # Try uniform distribution
        for pattern in self.DISTRIBUTION_PATTERNS['uniform']:
            match = re.match(pattern, distribution_str)
            if match:
                variable = match.group(1)
                a = float(match.group(2))
                b = float(match.group(3))
                
                return ProbabilityDistribution(
                    distribution_type=DistributionType.UNIFORM,
                    parameters={'a': a, 'b': b},
                    variable=variable
                )
        
        return None
    
    def parse_probability_expression(self, expression: str) -> Dict[str, Any]:
        """
        Parse probability expressions like P(A), P(A|B), P(A∩B).
        
        Args:
            expression: Probability expression string
            
        Returns:
            Dictionary with parsed components
        """
        expression = expression.strip()
        result = {'type': 'unknown', 'events': [], 'condition': None}
        
        # Try conditional probability
        for pattern in self.PROBABILITY_PATTERNS['conditional']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'conditional'
                result['events'] = [match.group(1).strip()]
                result['condition'] = match.group(2).strip()
                return result
        
        # Try joint probability
        for pattern in self.PROBABILITY_PATTERNS['joint']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'joint'
                result['events'] = [match.group(1).strip(), match.group(2).strip()]
                return result
        
        # Try basic probability
        for pattern in self.PROBABILITY_PATTERNS['basic']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'basic'
                result['events'] = [match.group(1).strip()]
                return result
        
        return result
    
    def parse_statistical_measure(self, expression: str) -> Dict[str, Any]:
        """
        Parse statistical measures like E[X], Var(X), Cov(X,Y).
        
        Args:
            expression: Statistical measure expression
            
        Returns:
            Dictionary with measure type and variables
        """
        expression = expression.strip()
        result = {'type': 'unknown', 'variables': []}
        
        # Try expectation
        for pattern in self.STATISTICAL_PATTERNS['expectation']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'expectation'
                result['variables'] = [match.group(1).strip()]
                return result
        
        # Try variance
        for pattern in self.STATISTICAL_PATTERNS['variance']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'variance'
                result['variables'] = [match.group(1).strip()]
                return result
        
        # Try covariance
        for pattern in self.STATISTICAL_PATTERNS['covariance']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'covariance'
                result['variables'] = [match.group(1).strip(), match.group(2).strip()]
                return result
        
        # Try correlation
        for pattern in self.STATISTICAL_PATTERNS['correlation']:
            match = re.match(pattern, expression)
            if match:
                result['type'] = 'correlation'
                result['variables'] = [match.group(1).strip(), match.group(2).strip()]
                return result
        
        return result
    
    def create_hypothesis_test(self, test_type: TestType, null_value: float,
                             alternative: str = "two-sided") -> StatisticalTest:
        """
        Create a statistical hypothesis test.
        
        Args:
            test_type: Type of test to create
            null_value: Value under null hypothesis
            alternative: Type of alternative ("two-sided", "greater", "less")
            
        Returns:
            StatisticalTest object
        """
        if test_type == TestType.T_TEST_ONE_SAMPLE:
            if alternative == "two-sided":
                h0 = f"H₀: μ = {null_value}"
                h1 = f"H₁: μ ≠ {null_value}"
            elif alternative == "greater":
                h0 = f"H₀: μ ≤ {null_value}"
                h1 = f"H₁: μ > {null_value}"
            else:  # less
                h0 = f"H₀: μ ≥ {null_value}"
                h1 = f"H₁: μ < {null_value}"
        
        elif test_type == TestType.Z_TEST:
            if alternative == "two-sided":
                h0 = f"H₀: p = {null_value}"
                h1 = f"H₁: p ≠ {null_value}"
            elif alternative == "greater":
                h0 = f"H₀: p ≤ {null_value}"
                h1 = f"H₁: p > {null_value}"
            else:  # less
                h0 = f"H₀: p ≥ {null_value}"
                h1 = f"H₁: p < {null_value}"
        
        else:
            h0 = f"H₀: parameter = {null_value}"
            h1 = f"H₁: parameter ≠ {null_value}"
        
        return StatisticalTest(
            test_type=test_type,
            null_hypothesis=h0,
            alternative_hypothesis=h1
        )
    
    def calculate_t_statistic(self, sample_mean: float, null_mean: float,
                            sample_std: float, n: int) -> float:
        """Calculate t-statistic for one-sample t-test."""
        return (sample_mean - null_mean) / (sample_std / math.sqrt(n))
    
    def calculate_z_statistic(self, sample_prop: float, null_prop: float,
                            n: int) -> float:
        """Calculate z-statistic for proportion test."""
        se = math.sqrt(null_prop * (1 - null_prop) / n)
        return (sample_prop - null_prop) / se
    
    def calculate_confidence_interval(self, sample_mean: float, sample_std: float,
                                    n: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean.
        
        Args:
            sample_mean: Sample mean
            sample_std: Sample standard deviation
            n: Sample size
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Use t-distribution for small samples, z for large
        if n < 30:
            # Simplified - would need proper t-table lookup
            critical_value = 2.0  # Approximation
        else:
            critical_value = self.z_critical_values.get(confidence_level, 1.96)
        
        margin_of_error = critical_value * (sample_std / math.sqrt(n))
        
        return (sample_mean - margin_of_error, sample_mean + margin_of_error)
    
    def bayes_update_beta_binomial(self, prior_alpha: float, prior_beta: float,
                                  successes: int, trials: int) -> Tuple[float, float]:
        """
        Bayesian update for Beta-Binomial conjugate model.
        
        Args:
            prior_alpha: Alpha parameter of Beta prior
            prior_beta: Beta parameter of Beta prior
            successes: Number of observed successes
            trials: Number of trials
            
        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + trials - successes
        
        return (posterior_alpha, posterior_beta)
    
    def format_distribution(self, distribution: ProbabilityDistribution,
                          format_type: str = 'standard') -> str:
        """
        Format probability distribution for different output types.
        
        Args:
            distribution: ProbabilityDistribution to format
            format_type: Output format ('standard', 'latex', 'html', 'ascii')
            
        Returns:
            Formatted string representation
        """
        if format_type == 'latex':
            return self._format_distribution_latex(distribution)
        elif format_type == 'html':
            return self._format_distribution_html(distribution)
        elif format_type == 'ascii':
            return self._format_distribution_ascii(distribution)
        else:
            return self._format_distribution_standard(distribution)
    
    def _format_distribution_standard(self, dist: ProbabilityDistribution) -> str:
        """Format distribution in standard notation."""
        if dist.distribution_type == DistributionType.NORMAL:
            mu = dist.parameters['mu']
            sigma = dist.parameters['sigma']
            return f"{dist.variable} ~ N({mu}, {sigma}²)"
        
        elif dist.distribution_type == DistributionType.BINOMIAL:
            n = dist.parameters['n']
            p = dist.parameters['p']
            return f"{dist.variable} ~ Bin({n}, {p})"
        
        elif dist.distribution_type == DistributionType.POISSON:
            lam = dist.parameters['lambda']
            return f"{dist.variable} ~ Poisson({lam})"
        
        elif dist.distribution_type == DistributionType.EXPONENTIAL:
            lam = dist.parameters['lambda']
            return f"{dist.variable} ~ Exp({lam})"
        
        elif dist.distribution_type == DistributionType.UNIFORM:
            a = dist.parameters['a']
            b = dist.parameters['b']
            return f"{dist.variable} ~ U({a}, {b})"
        
        return f"{dist.variable} ~ {dist.distribution_type.value}"
    
    def _format_distribution_latex(self, dist: ProbabilityDistribution) -> str:
        """Format distribution for LaTeX output."""
        if dist.distribution_type == DistributionType.NORMAL:
            mu = dist.parameters['mu']
            sigma = dist.parameters['sigma']
            return f"{dist.variable} \\sim \\mathcal{{N}}({mu}, {sigma}^2)"
        
        elif dist.distribution_type == DistributionType.BINOMIAL:
            n = dist.parameters['n']
            p = dist.parameters['p']
            return f"{dist.variable} \\sim \\text{{Bin}}({n}, {p})"
        
        elif dist.distribution_type == DistributionType.POISSON:
            lam = dist.parameters['lambda']
            return f"{dist.variable} \\sim \\text{{Poisson}}({lam})"
        
        return f"{dist.variable} \\sim \\text{{{dist.distribution_type.value}}}"
    
    def _format_distribution_html(self, dist: ProbabilityDistribution) -> str:
        """Format distribution for HTML output."""
        if dist.distribution_type == DistributionType.NORMAL:
            mu = dist.parameters['mu']
            sigma = dist.parameters['sigma']
            return f'<span class="distribution">{dist.variable} ~ 𝒩({mu}, {sigma}²)</span>'
        
        elif dist.distribution_type == DistributionType.BINOMIAL:
            n = dist.parameters['n']
            p = dist.parameters['p']
            return f'<span class="distribution">{dist.variable} ~ Bin({n}, {p})</span>'
        
        return f'<span class="distribution">{dist.variable} ~ {dist.distribution_type.value}</span>'
    
    def _format_distribution_ascii(self, dist: ProbabilityDistribution) -> str:
        """Format distribution for ASCII output."""
        if dist.distribution_type == DistributionType.NORMAL:
            mu = dist.parameters['mu']
            sigma = dist.parameters['sigma']
            return f"{dist.variable} ~ N({mu}, {sigma}^2)"
        
        elif dist.distribution_type == DistributionType.BINOMIAL:
            n = dist.parameters['n']
            p = dist.parameters['p']
            return f"{dist.variable} ~ Bin({n}, {p})"
        
        return f"{dist.variable} ~ {dist.distribution_type.value}"
    
    def format_hypothesis_test(self, test: StatisticalTest,
                             format_type: str = 'standard') -> str:
        """Format hypothesis test for different output types."""
        if format_type == 'latex':
            h0 = test.null_hypothesis.replace('₀', '_0').replace('₁', '_1')
            h1 = test.alternative_hypothesis.replace('₀', '_0').replace('₁', '_1')
            return f"$H_0: {h0.split(':', 1)[1].strip()}$\n$H_1: {h1.split(':', 1)[1].strip()}$"
        
        elif format_type == 'html':
            return f'<div class="hypothesis-test">{test.null_hypothesis}<br>{test.alternative_hypothesis}</div>'
        
        else:
            return f"{test.null_hypothesis}\n{test.alternative_hypothesis}"
    
    def generate_probability_table(self, distribution: ProbabilityDistribution,
                                 values: List[Union[int, float]]) -> Dict[Union[int, float], float]:
        """
        Generate probability table for discrete distributions.
        
        Args:
            distribution: Discrete probability distribution
            values: List of values to calculate probabilities for
            
        Returns:
            Dictionary mapping values to probabilities
        """
        table = {}
        
        if distribution.distribution_type == DistributionType.BINOMIAL:
            n = int(distribution.parameters['n'])
            p = distribution.parameters['p']
            
            for k in values:
                if 0 <= k <= n:
                    # Binomial probability: C(n,k) * p^k * (1-p)^(n-k)
                    comb = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
                    prob = comb * (p ** k) * ((1 - p) ** (n - k))
                    table[k] = prob
        
        elif distribution.distribution_type == DistributionType.POISSON:
            lam = distribution.parameters['lambda']
            
            for k in values:
                if k >= 0:
                    # Poisson probability: (λ^k * e^(-λ)) / k!
                    prob = (lam ** k) * math.exp(-lam) / math.factorial(k)
                    table[k] = prob
        
        return table
    
    def central_limit_theorem_approximation(self, population_mean: float,
                                          population_std: float,
                                          sample_size: int) -> ProbabilityDistribution:
        """
        Apply Central Limit Theorem to get sampling distribution.
        
        Args:
            population_mean: Population mean
            population_std: Population standard deviation
            sample_size: Size of each sample
            
        Returns:
            Normal distribution of sample means
        """
        sampling_mean = population_mean
        sampling_std = population_std / math.sqrt(sample_size)
        
        return ProbabilityDistribution(
            distribution_type=DistributionType.NORMAL,
            parameters={'mu': sampling_mean, 'sigma': sampling_std},
            variable="X̄"
        )


# Convenience functions for direct use
def parse_probability_distribution(distribution_str: str) -> Optional[ProbabilityDistribution]:
    """Parse probability distribution from string."""
    processor = ProbabilityProcessor()
    return processor.parse_distribution(distribution_str)


def calculate_normal_probability(x: float, mu: float, sigma: float) -> float:
    """Calculate probability density for normal distribution."""
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * math.exp(exponent)


def binomial_probability(k: int, n: int, p: float) -> float:
    """Calculate binomial probability P(X = k)."""
    if not (0 <= k <= n):
        return 0.0
    
    comb = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    return comb * (p ** k) * ((1 - p) ** (n - k))


def poisson_probability(k: int, lam: float) -> float:
    """Calculate Poisson probability P(X = k)."""
    if k < 0:
        return 0.0
    
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


# Integration with multi-format renderer
PROBABILITY_FORMAT_EXTENSIONS = {
    'latex': lambda obj: ProbabilityProcessor().format_distribution(obj, 'latex') if hasattr(obj, 'distribution_type') else str(obj),
    'html': lambda obj: ProbabilityProcessor().format_distribution(obj, 'html') if hasattr(obj, 'distribution_type') else str(obj),
    'ascii': lambda obj: ProbabilityProcessor().format_distribution(obj, 'ascii') if hasattr(obj, 'distribution_type') else str(obj)
}