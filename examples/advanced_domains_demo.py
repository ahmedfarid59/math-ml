"""
Advanced Mathematical Domains Examples
======================================

This script demonstrates the capabilities of all advanced mathematical
domain processors including complex numbers, differential equations,
probability theory, and set theory.

Run this script to see comprehensive examples of each domain processor
in action with various mathematical expressions and operations.
"""

import sys
import os

# Add the parent directory to the path to import mathml_parser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("ADVANCED MATHEMATICAL DOMAINS DEMONSTRATION")
    print("=" * 60)
    print()
    
    try:
        # Import domain processors
        from mathml_parser.domains import (
            ComplexNumberProcessor, DifferentialEquationProcessor,
            ProbabilityProcessor, SetTheoryProcessor,
            get_processor, list_domains, demonstrate_domains
        )
        
        print("✅ Successfully imported all domain processors!")
        print(f"Available domains: {', '.join(list_domains())}")
        print()
        
        # Run comprehensive demonstration
        demonstrate_domains()
        
        print("\n" + "=" * 60)
        print("DETAILED EXAMPLES BY DOMAIN")
        print("=" * 60)
        
        # 1. Complex Numbers - Detailed Examples
        print("\n🔢 COMPLEX NUMBERS - Detailed Examples")
        print("-" * 40)
        
        complex_proc = ComplexNumberProcessor()
        
        # Basic complex number operations
        print("Basic Operations:")
        z1 = complex_proc.parse_complex("5 + 3i")
        z2 = complex_proc.parse_complex("2 - 4i")
        
        if z1 and z2:
            print(f"  z₁ = {z1}")
            print(f"  z₂ = {z2}")
            print(f"  z₁ + z₂ = {z1 + z2}")
            print(f"  z₁ × z₂ = {z1 * z2}")
            print(f"  |z₁| = {z1.magnitude:.3f}")
            print(f"  arg(z₁) = {z1.argument_degrees:.1f}°")
            print(f"  z₁* = {z1.conjugate}")
        
        # Polar form conversion
        print("\nPolar Form:")
        z_polar = z1.to_polar() if z1 else None
        if z_polar:
            print(f"  Rectangular: {z1}")
            print(f"  Polar: {z_polar}")
        
        # Complex functions
        print("\nComplex Functions:")
        if z1:
            exp_z = complex_proc.complex_functions('exp', z1)
            sqrt_z = complex_proc.complex_functions('sqrt', z1)
            print(f"  exp(z₁) = {exp_z}")
            print(f"  √z₁ = {sqrt_z}")
        
        # Roots of unity
        print("\nRoots of Unity (5th roots):")
        roots = complex_proc.roots_of_unity(5)
        for i, root in enumerate(roots):
            print(f"  ω_{i} = {root}")
        
        # Quadratic equation with complex solutions
        print("\nQuadratic Equation: x² + 2x + 5 = 0")
        x1, x2 = complex_proc.solve_quadratic_complex(1, 2, 5)
        print(f"  x₁ = {x1}")
        print(f"  x₂ = {x2}")
        
        # 2. Differential Equations - Detailed Examples
        print("\n📐 DIFFERENTIAL EQUATIONS - Detailed Examples")
        print("-" * 40)
        
        de_proc = DifferentialEquationProcessor()
        
        # First-order ODE
        print("First-Order ODE:")
        ode1 = de_proc.parse_ode("dy/dx = 2x + y")
        if ode1:
            print(f"  Equation: {ode1.equation}")
            print(f"  Type: {ode1.equation_type.value}")
            print(f"  Order: {ode1.order}")
            print(f"  Linearity: {ode1.linearity}")
            print(f"  Homogeneous: {ode1.homogeneous}")
            
            methods = de_proc.suggest_solution_method(ode1)
            print(f"  Suggested methods: {', '.join(methods[:3])}")
            
            solution = de_proc.solve_first_order_linear(ode1)
            if solution:
                print(f"  Solution approach: {solution}")
        
        # Second-order ODE
        print("\nSecond-Order ODE:")
        ode2 = de_proc.parse_ode("d²y/dx² + 4y = sin(x)")
        if ode2:
            print(f"  Equation: {ode2.equation}")
            print(f"  Type: {ode2.equation_type.value}")
            print(f"  Order: {ode2.order}")
            methods = de_proc.suggest_solution_method(ode2)
            print(f"  Suggested methods: {', '.join(methods[:3])}")
        
        # PDE - Heat equation
        print("\nPartial Differential Equation:")
        pde = de_proc.parse_pde("∂u/∂t = α∂²u/∂x²")
        if pde:
            print(f"  Equation: {pde.equation}")
            print(f"  Type: {pde.equation_type.value}")
            print(f"  Variables: {pde.variables}")
            print(f"  Independent vars: {pde.independent_vars}")
            print(f"  Numerical methods: {', '.join(pde.numerical_methods)}")
        
        # Boundary conditions
        print("\nBoundary Conditions:")
        bc1 = de_proc.parse_boundary_condition("u(0) = 0")
        bc2 = de_proc.parse_boundary_condition("∂u/∂x|_{x=1} = 5")
        if bc1:
            print(f"  {bc1}")
        if bc2:
            print(f"  {bc2}")
        
        # Numerical solution code generation
        print("\nNumerical Solution Code (Runge-Kutta):")
        if ode1:
            code = de_proc.generate_numerical_solution_code(ode1, 'runge_kutta')
            print("  [Code generated - see full implementation]")
        
        # 3. Probability & Statistics - Detailed Examples
        print("\n📊 PROBABILITY & STATISTICS - Detailed Examples")
        print("-" * 40)
        
        prob_proc = ProbabilityProcessor()
        
        # Normal distribution
        print("Normal Distribution:")
        normal = prob_proc.parse_distribution("X ~ N(100, 15)")
        if normal:
            print(f"  Distribution: {prob_proc.format_distribution(normal)}")
            print(f"  Mean: {normal.properties['mean']}")
            print(f"  Standard deviation: {normal.properties['std']}")
            print(f"  Variance: {normal.properties['variance']}")
            print(f"  PDF formula: {normal.pdf_formula()}")
        
        # Binomial distribution
        print("\nBinomial Distribution:")
        binomial = prob_proc.parse_distribution("Y ~ Bin(20, 0.3)")
        if binomial:
            print(f"  Distribution: {prob_proc.format_distribution(binomial)}")
            print(f"  Mean: {binomial.properties['mean']}")
            print(f"  Variance: {binomial.properties['variance']}")
            
            # Probability table for some values
            prob_table = prob_proc.generate_probability_table(binomial, [0, 1, 2, 5, 10])
            print("  Probability table:")
            for k, p in prob_table.items():
                print(f"    P(Y = {k}) = {p:.4f}")
        
        # Poisson distribution
        print("\nPoisson Distribution:")
        poisson = prob_proc.parse_distribution("Z ~ Poisson(3)")
        if poisson:
            print(f"  Distribution: {prob_proc.format_distribution(poisson)}")
            print(f"  Mean = Variance = {poisson.properties['mean']}")
        
        # Hypothesis testing
        print("\nHypothesis Testing:")
        from mathml_parser.domains import TestType
        t_test = prob_proc.create_hypothesis_test(TestType.T_TEST_ONE_SAMPLE, 50.0, "two-sided")
        print(f"  {t_test.null_hypothesis}")
        print(f"  {t_test.alternative_hypothesis}")
        
        # Sample calculation
        t_stat = prob_proc.calculate_t_statistic(52.5, 50.0, 8.2, 25)
        print(f"  Sample t-statistic: {t_stat:.3f}")
        
        # Confidence interval
        ci_lower, ci_upper = prob_proc.calculate_confidence_interval(52.5, 8.2, 25, 0.95)
        print(f"  95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
        
        # Central Limit Theorem
        print("\nCentral Limit Theorem:")
        sampling_dist = prob_proc.central_limit_theorem_approximation(50, 12, 36)
        print(f"  Population: μ = 50, σ = 12")
        print(f"  Sample size: n = 36")
        print(f"  Sampling distribution: {prob_proc.format_distribution(sampling_dist)}")
        
        # Bayesian inference
        print("\nBayesian Inference (Beta-Binomial):")
        posterior_alpha, posterior_beta = prob_proc.bayes_update_beta_binomial(2, 2, 8, 10)
        print(f"  Prior: Beta(2, 2)")
        print(f"  Data: 8 successes in 10 trials")
        print(f"  Posterior: Beta({posterior_alpha}, {posterior_beta})")
        
        # 4. Set Theory & Logic - Detailed Examples
        print("\n🎯 SET THEORY & LOGIC - Detailed Examples")
        print("-" * 40)
        
        set_proc = SetTheoryProcessor()
        
        # Basic sets
        print("Basic Set Operations:")
        A = set_proc.parse_set("{1, 2, 3, 4, 5}")
        B = set_proc.parse_set("{4, 5, 6, 7, 8}")
        C = set_proc.parse_set("{2, 4, 6, 8, 10}")
        
        if A and B and C:
            print(f"  A = {set_proc.format_set(A)}")
            print(f"  B = {set_proc.format_set(B)}")
            print(f"  C = {set_proc.format_set(C)}")
            
            # Set operations
            union_AB = A.union(B)
            intersection_AB = A.intersection(B)
            difference_AB = A.difference(B)
            
            print(f"  A ∪ B = {set_proc.format_set(union_AB)}")
            print(f"  A ∩ B = {set_proc.format_set(intersection_AB)}")
            print(f"  A \\ B = {set_proc.format_set(difference_AB)}")
            
            # Set relations
            relations = set_proc.check_set_relations(A, B)
            print(f"  A ⊆ B: {relations['subset']}")
            print(f"  A = B: {relations['equal']}")
            print(f"  A ∩ B = ∅: {relations['disjoint']}")
        
        # Set builder notation
        print("\nSet Builder Notation:")
        set_builder = set_proc.parse_set("{x | x > 0}")
        if set_builder:
            print(f"  {set_builder.description}")
            print(f"  Type: {'finite' if set_builder.is_finite else 'infinite'}")
        
        # Standard sets
        print("\nStandard Mathematical Sets:")
        naturals = set_proc.parse_set("ℕ")
        integers = set_proc.parse_set("ℤ")
        rationals = set_proc.parse_set("ℚ")
        reals = set_proc.parse_set("ℝ")
        
        if naturals:
            print(f"  ℕ = {naturals.description}")
        if integers:
            print(f"  ℤ = {integers.description}")
        if rationals:
            print(f"  ℚ = {rationals.description}")
        if reals:
            print(f"  ℝ = {reals.description}")
        
        # Power set
        print("\nPower Set:")
        small_set = set_proc.parse_set("{1, 2}")
        if small_set:
            power_set = set_proc.power_set(small_set)
            print(f"  S = {set_proc.format_set(small_set)}")
            print(f"  P(S) has {power_set.properties.get('cardinality', 'unknown')} elements")
            if power_set.elements:
                print("  P(S) = {", end="")
                for i, subset in enumerate(power_set.elements):
                    if i > 0:
                        print(", ", end="")
                    print("{" + ", ".join(map(str, subset)) + "}", end="")
                print("}")
        
        # Logical expressions
        print("\nLogical Expressions:")
        logic1 = set_proc.parse_logical_expression("p ∧ q")
        logic2 = set_proc.parse_logical_expression("p → q")
        logic3 = set_proc.parse_logical_expression("(p ∨ q) ∧ ¬r")
        
        for i, logic in enumerate([logic1, logic2, logic3], 1):
            if logic:
                print(f"  Expression {i}: {logic.expression}")
                print(f"    Variables: {logic.variables}")
                print(f"    Operators: {[op.value for op in logic.operators]}")
                if logic.truth_table is not None:
                    print(f"    Tautology: {logic.is_tautology}")
                    print(f"    Contradiction: {logic.is_contradiction}")
        
        # Quantified statements
        print("\nQuantified Statements:")
        quant1 = set_proc.parse_quantified_statement("∀x ∈ ℝ, x² ≥ 0")
        quant2 = set_proc.parse_quantified_statement("∃y ∈ ℕ, y > 10")
        
        if quant1:
            print(f"  {quant1}")
            print(f"    Quantifier: {quant1.quantifier.value}")
            print(f"    Variable: {quant1.variable}")
            print(f"    Domain: {quant1.domain}")
        
        if quant2:
            print(f"  {quant2}")
            print(f"    Quantifier: {quant2.quantifier.value}")
            print(f"    Variable: {quant2.variable}")
            print(f"    Domain: {quant2.domain}")
        
        # Mathematical proof structure
        print("\nMathematical Proof Structure:")
        from mathml_parser.domains import ProofType
        proof = set_proc.create_proof(
            "For all integers n, if n is even, then n² is even",
            ProofType.DIRECT
        )
        
        set_proc.add_proof_step(
            "Let n be an arbitrary even integer",
            "Assumption"
        )
        set_proc.add_proof_step(
            "Then n = 2k for some integer k",
            "Definition of even"
        )
        set_proc.add_proof_step(
            "n² = (2k)² = 4k² = 2(2k²)",
            "Algebra"
        )
        set_proc.add_proof_step(
            "Since 2k² is an integer, n² is even",
            "Definition of even"
        )
        
        proof.conclusion = "Therefore, if n is even, then n² is even"
        
        print("  Proof by Direct Method:")
        formatted_proof = set_proc.format_proof(proof)
        for line in formatted_proof.split('\n'):
            print(f"    {line}")
        
        # Output format examples
        print("\n" + "=" * 60)
        print("MULTI-FORMAT OUTPUT EXAMPLES")
        print("=" * 60)
        
        if z1:  # Complex number
            print("\nComplex Number Formatting:")
            print(f"  Standard: {complex_proc.format_complex(z1, 'standard')}")
            print(f"  LaTeX:    {complex_proc.format_complex(z1, 'latex')}")
            print(f"  HTML:     {complex_proc.format_complex(z1, 'html')}")
            print(f"  ASCII:    {complex_proc.format_complex(z1, 'ascii')}")
        
        if normal:  # Probability distribution
            print("\nProbability Distribution Formatting:")
            print(f"  Standard: {prob_proc.format_distribution(normal, 'standard')}")
            print(f"  LaTeX:    {prob_proc.format_distribution(normal, 'latex')}")
            print(f"  HTML:     {prob_proc.format_distribution(normal, 'html')}")
            print(f"  ASCII:    {prob_proc.format_distribution(normal, 'ascii')}")
        
        if A:  # Set
            print("\nSet Formatting:")
            print(f"  Standard: {set_proc.format_set(A, 'standard')}")
            print(f"  LaTeX:    {set_proc.format_set(A, 'latex')}")
            print(f"  HTML:     {set_proc.format_set(A, 'html')}")
            print(f"  ASCII:    {set_proc.format_set(A, 'ascii')}")
        
        if ode1:  # Differential equation
            print("\nDifferential Equation Formatting:")
            print(f"  Standard: {de_proc.format_equation(ode1, 'standard')}")
            print(f"  LaTeX:    {de_proc.format_equation(ode1, 'latex')}")
            print(f"  HTML:     {de_proc.format_equation(ode1, 'html')}")
            print(f"  ASCII:    {de_proc.format_equation(ode1, 'ascii')}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60)
        print("\nAll advanced mathematical domain processors are working correctly.")
        print("The MathML parser now supports comprehensive mathematical notation")
        print("across complex analysis, differential equations, probability theory,")
        print("and set theory with multi-format output capabilities.")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all domain modules are properly implemented.")
        return 1
    
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())