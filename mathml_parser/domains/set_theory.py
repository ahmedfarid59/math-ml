"""
Set Theory and Logic Processing
==============================

This module provides specialized parsing and formatting for set theory
operations, logical expressions, and mathematical proof structures.

Features:
- Set operations (union, intersection, complement, etc.)
- Logical operators (and, or, not, implies, etc.)
- Proof structure representation
- Quantifier handling (forall, exists)
- Set builder notation
- Multi-format output for logical expressions
"""

import re
from typing import Union, List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class SetOperation(Enum):
    """Types of set operations."""
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    SYMMETRIC_DIFFERENCE = "symmetric_difference"
    COMPLEMENT = "complement"
    CARTESIAN_PRODUCT = "cartesian_product"
    POWER_SET = "power_set"


class LogicalOperator(Enum):
    """Types of logical operators."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"
    NAND = "nand"
    NOR = "nor"


class Quantifier(Enum):
    """Types of quantifiers."""
    FORALL = "forall"  # ∀
    EXISTS = "exists"  # ∃
    EXISTS_UNIQUE = "exists_unique"  # ∃!
    NOT_EXISTS = "not_exists"  # ∄


class ProofType(Enum):
    """Types of mathematical proofs."""
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    CONTRAPOSITIVE = "contrapositive"
    INDUCTION = "induction"
    STRONG_INDUCTION = "strong_induction"
    CONSTRUCTION = "construction"
    EXISTENCE = "existence"
    UNIQUENESS = "uniqueness"


@dataclass
class MathematicalSet:
    """
    Represents a mathematical set with elements and properties.
    
    Attributes:
        name: Name of the set (e.g., 'A', 'S')
        elements: List of elements in the set (for finite sets)
        description: Set builder notation or description
        is_finite: Whether the set is finite
        cardinality: Number of elements (for finite sets)
        properties: Additional properties of the set
    """
    name: str
    elements: Optional[List[Any]] = None
    description: Optional[str] = None
    is_finite: bool = True
    cardinality: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate cardinality for finite sets."""
        if self.is_finite and self.elements is not None:
            self.cardinality = len(set(self.elements))  # Remove duplicates
    
    def add_element(self, element: Any):
        """Add an element to the set."""
        if self.elements is None:
            self.elements = []
        if element not in self.elements:
            self.elements.append(element)
            if self.is_finite:
                self.cardinality = len(self.elements)
    
    def contains(self, element: Any) -> bool:
        """Check if element is in the set."""
        if self.elements is not None:
            return element in self.elements
        return False  # For infinite sets, would need special handling
    
    def is_subset_of(self, other: 'MathematicalSet') -> bool:
        """Check if this set is a subset of another."""
        if self.elements is None or other.elements is None:
            return False  # Simplified for finite sets
        
        return all(elem in other.elements for elem in self.elements)
    
    def union(self, other: 'MathematicalSet') -> 'MathematicalSet':
        """Return union of this set with another."""
        if self.elements is None or other.elements is None:
            return MathematicalSet(
                name=f"{self.name} ∪ {other.name}",
                description=f"Union of {self.name} and {other.name}"
            )
        
        union_elements = list(set(self.elements + other.elements))
        return MathematicalSet(
            name=f"{self.name} ∪ {other.name}",
            elements=union_elements
        )
    
    def intersection(self, other: 'MathematicalSet') -> 'MathematicalSet':
        """Return intersection of this set with another."""
        if self.elements is None or other.elements is None:
            return MathematicalSet(
                name=f"{self.name} ∩ {other.name}",
                description=f"Intersection of {self.name} and {other.name}"
            )
        
        intersection_elements = [elem for elem in self.elements if elem in other.elements]
        return MathematicalSet(
            name=f"{self.name} ∩ {other.name}",
            elements=intersection_elements
        )
    
    def difference(self, other: 'MathematicalSet') -> 'MathematicalSet':
        """Return difference of this set with another."""
        if self.elements is None or other.elements is None:
            return MathematicalSet(
                name=f"{self.name} \\ {other.name}",
                description=f"Difference of {self.name} and {other.name}"
            )
        
        difference_elements = [elem for elem in self.elements if elem not in other.elements]
        return MathematicalSet(
            name=f"{self.name} \\ {other.name}",
            elements=difference_elements
        )


@dataclass
class LogicalExpression:
    """
    Represents a logical expression or statement.
    
    Attributes:
        expression: The logical expression as string
        variables: List of propositional variables
        operators: List of logical operators used
        truth_table: Truth table for the expression
        is_tautology: Whether expression is always true
        is_contradiction: Whether expression is always false
        normal_forms: CNF and DNF representations
    """
    expression: str
    variables: List[str] = field(default_factory=list)
    operators: List[LogicalOperator] = field(default_factory=list)
    truth_table: Optional[Dict] = None
    is_tautology: Optional[bool] = None
    is_contradiction: Optional[bool] = None
    normal_forms: Dict[str, str] = field(default_factory=dict)


@dataclass
class QuantifiedStatement:
    """
    Represents a quantified logical statement.
    
    Attributes:
        quantifier: Type of quantifier (∀, ∃, etc.)
        variable: Quantified variable
        domain: Domain of quantification
        predicate: Predicate expression
        nested_quantifiers: List of nested quantifiers
    """
    quantifier: Quantifier
    variable: str
    domain: Optional[str] = None
    predicate: Optional[str] = None
    nested_quantifiers: List['QuantifiedStatement'] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation of quantified statement."""
        quant_symbol = {
            Quantifier.FORALL: "∀",
            Quantifier.EXISTS: "∃",
            Quantifier.EXISTS_UNIQUE: "∃!",
            Quantifier.NOT_EXISTS: "∄"
        }.get(self.quantifier, "?")
        
        domain_part = f" ∈ {self.domain}" if self.domain else ""
        predicate_part = f", {self.predicate}" if self.predicate else ""
        
        return f"{quant_symbol}{self.variable}{domain_part}{predicate_part}"


@dataclass
class MathematicalProof:
    """
    Represents a mathematical proof structure.
    
    Attributes:
        theorem: Statement to be proved
        proof_type: Type of proof strategy
        assumptions: List of assumptions/hypotheses
        steps: List of proof steps
        conclusion: Final conclusion
        references: References to lemmas, theorems, etc.
    """
    theorem: str
    proof_type: ProofType
    assumptions: List[str] = field(default_factory=list)
    steps: List[Dict[str, str]] = field(default_factory=list)
    conclusion: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    def add_step(self, step_number: int, statement: str, justification: str):
        """Add a step to the proof."""
        self.steps.append({
            'number': step_number,
            'statement': statement,
            'justification': justification
        })


class SetTheoryProcessor:
    """
    Processor for set theory and logical expressions.
    
    This class handles parsing, manipulation, and formatting of sets,
    logical expressions, and proof structures.
    """
    
    # Set notation patterns
    SET_PATTERNS = {
        'roster': [
            r"\{([^}]+)\}",  # {1, 2, 3, 4}
            r"\{(.+)\}",     # General form
        ],
        'builder': [
            r"\{([^|]+)\|([^}]+)\}",  # {x | x > 0}
            r"\{([^:]+):([^}]+)\}",   # {x : x > 0}
        ],
        'interval': [
            r"\[([^,]+),\s*([^\]]+)\]",  # [a, b]
            r"\(([^,]+),\s*([^\)]+)\)",  # (a, b)
            r"\[([^,]+),\s*([^\)]+)\)",  # [a, b)
            r"\(([^,]+),\s*([^\]]+)\]",  # (a, b]
        ]
    }
    
    # Set operation patterns
    SET_OPERATION_PATTERNS = {
        'union': [r"([A-Z])\s*∪\s*([A-Z])", r"([A-Z])\s*\\\cup\s*([A-Z])"],
        'intersection': [r"([A-Z])\s*∩\s*([A-Z])", r"([A-Z])\s*\\\cap\s*([A-Z])"],
        'difference': [r"([A-Z])\s*\\\s*([A-Z])", r"([A-Z])\s*-\s*([A-Z])"],
        'complement': [r"([A-Z])^c", r"([A-Z])'", r"\\\overline\{([A-Z])\}"],
        'cartesian': [r"([A-Z])\s*×\s*([A-Z])", r"([A-Z])\s*\\\times\s*([A-Z])"],
    }
    
    # Logical operator patterns
    LOGICAL_PATTERNS = {
        'and': [r"([^∧]+)\s*∧\s*([^∧]+)", r"([^&]+)\s*&\s*([^&]+)", r"([^a]+)\s*and\s*([^a]+)"],
        'or': [r"([^∨]+)\s*∨\s*([^∨]+)", r"([^|]+)\s*\|\s*([^|]+)", r"([^o]+)\s*or\s*([^o]+)"],
        'not': [r"¬([^¬]+)", r"~([^~]+)", r"not\s*([^n]+)"],
        'implies': [r"([^→]+)\s*→\s*([^→]+)", r"([^i]+)\s*implies\s*([^i]+)"],
        'iff': [r"([^↔]+)\s*↔\s*([^↔]+)", r"([^i]+)\s*iff\s*([^i]+)"],
    }
    
    # Quantifier patterns
    QUANTIFIER_PATTERNS = {
        'forall': [r"∀([a-z])\s*([^,]*),?\s*(.+)", r"for\s*all\s*([a-z])\s*(.+)"],
        'exists': [r"∃([a-z])\s*([^,]*),?\s*(.+)", r"there\s*exists\s*([a-z])\s*(.+)"],
        'exists_unique': [r"∃!([a-z])\s*([^,]*),?\s*(.+)"],
    }
    
    # Well-known sets
    STANDARD_SETS = {
        'ℕ': 'natural numbers',
        'ℤ': 'integers',
        'ℚ': 'rational numbers',
        'ℝ': 'real numbers',
        'ℂ': 'complex numbers',
        '∅': 'empty set',
        'ℙ': 'prime numbers',
    }
    
    def __init__(self):
        """Initialize the set theory processor."""
        self.use_unicode = True
        self.proof_steps_counter = 0
        self.current_proof = None
        
        # Truth table generation settings
        self.max_variables = 8  # Limit for truth table generation
        
    def parse_set(self, set_str: str) -> Optional[MathematicalSet]:
        """
        Parse set notation from string.
        
        Args:
            set_str: String representation of set
            
        Returns:
            MathematicalSet object if parsing successful
            
        Example:
            >>> processor = SetTheoryProcessor()
            >>> s = processor.parse_set("{1, 2, 3, 4}")
            >>> print(s.cardinality)
            4
        """
        set_str = set_str.strip()
        
        # Check for standard sets
        if set_str in self.STANDARD_SETS:
            return MathematicalSet(
                name=set_str,
                description=self.STANDARD_SETS[set_str],
                is_finite=False
            )
        
        # Try roster notation
        for pattern in self.SET_PATTERNS['roster']:
            match = re.match(pattern, set_str)
            if match:
                elements_str = match.group(1)
                # Parse elements (simplified)
                elements = [elem.strip() for elem in elements_str.split(',')]
                
                # Try to convert to numbers if possible
                parsed_elements = []
                for elem in elements:
                    try:
                        # Try integer first
                        parsed_elements.append(int(elem))
                    except ValueError:
                        try:
                            # Try float
                            parsed_elements.append(float(elem))
                        except ValueError:
                            # Keep as string
                            parsed_elements.append(elem)
                
                return MathematicalSet(
                    name="S",  # Default name
                    elements=parsed_elements
                )
        
        # Try set builder notation
        for pattern in self.SET_PATTERNS['builder']:
            match = re.match(pattern, set_str)
            if match:
                variable = match.group(1).strip()
                condition = match.group(2).strip()
                
                return MathematicalSet(
                    name="S",
                    description=f"{{{variable} | {condition}}}",
                    is_finite=False  # Generally infinite for builder notation
                )
        
        # Try interval notation
        for pattern in self.SET_PATTERNS['interval']:
            match = re.match(pattern, set_str)
            if match:
                lower = match.group(1).strip()
                upper = match.group(2).strip()
                
                # Determine interval type from pattern
                if set_str.startswith('[') and set_str.endswith(']'):
                    interval_type = "closed"
                elif set_str.startswith('(') and set_str.endswith(')'):
                    interval_type = "open"
                elif set_str.startswith('[') and set_str.endswith(')'):
                    interval_type = "half_open_right"
                else:  # ( ... ]
                    interval_type = "half_open_left"
                
                return MathematicalSet(
                    name="I",
                    description=f"Interval {set_str}",
                    is_finite=False,
                    properties={'type': 'interval', 'interval_type': interval_type,
                              'lower': lower, 'upper': upper}
                )
        
        return None
    
    def parse_set_operation(self, operation_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse set operation expressions.
        
        Args:
            operation_str: String with set operation
            
        Returns:
            Dictionary with operation details
        """
        operation_str = operation_str.strip()
        
        # Try union
        for pattern in self.SET_OPERATION_PATTERNS['union']:
            match = re.match(pattern, operation_str)
            if match:
                return {
                    'operation': SetOperation.UNION,
                    'operands': [match.group(1), match.group(2)],
                    'expression': operation_str
                }
        
        # Try intersection
        for pattern in self.SET_OPERATION_PATTERNS['intersection']:
            match = re.match(pattern, operation_str)
            if match:
                return {
                    'operation': SetOperation.INTERSECTION,
                    'operands': [match.group(1), match.group(2)],
                    'expression': operation_str
                }
        
        # Try difference
        for pattern in self.SET_OPERATION_PATTERNS['difference']:
            match = re.match(pattern, operation_str)
            if match:
                return {
                    'operation': SetOperation.DIFFERENCE,
                    'operands': [match.group(1), match.group(2)],
                    'expression': operation_str
                }
        
        # Try complement
        for pattern in self.SET_OPERATION_PATTERNS['complement']:
            match = re.match(pattern, operation_str)
            if match:
                return {
                    'operation': SetOperation.COMPLEMENT,
                    'operands': [match.group(1)],
                    'expression': operation_str
                }
        
        # Try cartesian product
        for pattern in self.SET_OPERATION_PATTERNS['cartesian']:
            match = re.match(pattern, operation_str)
            if match:
                return {
                    'operation': SetOperation.CARTESIAN_PRODUCT,
                    'operands': [match.group(1), match.group(2)],
                    'expression': operation_str
                }
        
        return None
    
    def parse_logical_expression(self, expression_str: str) -> Optional[LogicalExpression]:
        """
        Parse logical expressions with operators.
        
        Args:
            expression_str: String with logical expression
            
        Returns:
            LogicalExpression object if parsing successful
        """
        expression_str = expression_str.strip()
        
        # Extract propositional variables (single letters)
        variables = list(set(re.findall(r'\b[p-z]\b', expression_str)))
        
        # Identify operators
        operators = []
        
        # Check for each operator type
        for op_type, patterns in self.LOGICAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, expression_str):
                    operators.append(LogicalOperator(op_type))
                    break
        
        expression = LogicalExpression(
            expression=expression_str,
            variables=sorted(variables),
            operators=list(set(operators))  # Remove duplicates
        )
        
        # Generate truth table if not too many variables
        if len(variables) <= self.max_variables:
            expression.truth_table = self._generate_truth_table(expression)
            expression.is_tautology = self._check_tautology(expression.truth_table)
            expression.is_contradiction = self._check_contradiction(expression.truth_table)
        
        return expression
    
    def parse_quantified_statement(self, statement_str: str) -> Optional[QuantifiedStatement]:
        """
        Parse quantified logical statements.
        
        Args:
            statement_str: String with quantified statement
            
        Returns:
            QuantifiedStatement object if parsing successful
        """
        statement_str = statement_str.strip()
        
        # Try forall quantifier
        for pattern in self.QUANTIFIER_PATTERNS['forall']:
            match = re.match(pattern, statement_str)
            if match:
                variable = match.group(1)
                domain = match.group(2).strip() if len(match.groups()) > 2 and match.group(2) else None
                predicate = match.group(3) if len(match.groups()) > 2 else match.group(2)
                
                # Clean up domain
                if domain and domain.startswith('∈'):
                    domain = domain[1:].strip()
                
                return QuantifiedStatement(
                    quantifier=Quantifier.FORALL,
                    variable=variable,
                    domain=domain,
                    predicate=predicate
                )
        
        # Try exists quantifier
        for pattern in self.QUANTIFIER_PATTERNS['exists']:
            match = re.match(pattern, statement_str)
            if match:
                variable = match.group(1)
                domain = match.group(2).strip() if len(match.groups()) > 2 and match.group(2) else None
                predicate = match.group(3) if len(match.groups()) > 2 else match.group(2)
                
                if domain and domain.startswith('∈'):
                    domain = domain[1:].strip()
                
                return QuantifiedStatement(
                    quantifier=Quantifier.EXISTS,
                    variable=variable,
                    domain=domain,
                    predicate=predicate
                )
        
        # Try exists unique quantifier
        for pattern in self.QUANTIFIER_PATTERNS['exists_unique']:
            match = re.match(pattern, statement_str)
            if match:
                variable = match.group(1)
                domain = match.group(2).strip() if len(match.groups()) > 2 and match.group(2) else None
                predicate = match.group(3) if len(match.groups()) > 2 else match.group(2)
                
                if domain and domain.startswith('∈'):
                    domain = domain[1:].strip()
                
                return QuantifiedStatement(
                    quantifier=Quantifier.EXISTS_UNIQUE,
                    variable=variable,
                    domain=domain,
                    predicate=predicate
                )
        
        return None
    
    def _generate_truth_table(self, expression: LogicalExpression) -> Dict:
        """Generate truth table for logical expression."""
        variables = expression.variables
        n_vars = len(variables)
        
        # Generate all possible truth value combinations
        truth_combinations = []
        for i in range(2**n_vars):
            combination = {}
            for j, var in enumerate(variables):
                # Extract bit j from i
                combination[var] = bool((i >> j) & 1)
            truth_combinations.append(combination)
        
        # Evaluate expression for each combination
        # This is a simplified evaluator - real implementation would parse the expression
        truth_table = {
            'variables': variables,
            'combinations': truth_combinations,
            'results': []
        }
        
        for combo in truth_combinations:
            # Simplified evaluation - would need proper expression parser
            result = self._evaluate_expression_simple(expression.expression, combo)
            truth_table['results'].append(result)
        
        return truth_table
    
    def _evaluate_expression_simple(self, expression: str, values: Dict[str, bool]) -> bool:
        """
        Simplified expression evaluator for truth tables.
        Real implementation would use proper parsing.
        """
        # Replace variables with their truth values
        expr = expression
        for var, val in values.items():
            expr = expr.replace(var, str(val))
        
        # Replace logical operators with Python equivalents
        expr = expr.replace('∧', ' and ')
        expr = expr.replace('∨', ' or ')
        expr = expr.replace('¬', ' not ')
        expr = expr.replace('→', ' <= ')  # Simplified implication
        expr = expr.replace('↔', ' == ')  # Biconditional
        
        try:
            # Evaluate the expression (unsafe - for demo only)
            return eval(expr)
        except:
            return False  # Default for unparseable expressions
    
    def _check_tautology(self, truth_table: Dict) -> bool:
        """Check if all results in truth table are True."""
        return all(truth_table['results'])
    
    def _check_contradiction(self, truth_table: Dict) -> bool:
        """Check if all results in truth table are False."""
        return not any(truth_table['results'])
    
    def create_proof(self, theorem: str, proof_type: ProofType) -> MathematicalProof:
        """
        Create a new proof structure.
        
        Args:
            theorem: Statement to prove
            proof_type: Type of proof strategy
            
        Returns:
            MathematicalProof object
        """
        proof = MathematicalProof(
            theorem=theorem,
            proof_type=proof_type
        )
        self.current_proof = proof
        self.proof_steps_counter = 0
        return proof
    
    def add_proof_step(self, statement: str, justification: str) -> int:
        """Add a step to the current proof."""
        if self.current_proof is None:
            raise ValueError("No current proof. Create a proof first.")
        
        self.proof_steps_counter += 1
        self.current_proof.add_step(self.proof_steps_counter, statement, justification)
        return self.proof_steps_counter
    
    def prove_by_contradiction(self, theorem: str, assumption: str) -> MathematicalProof:
        """
        Set up a proof by contradiction structure.
        
        Args:
            theorem: Statement to prove
            assumption: Negation of theorem (assumption for contradiction)
            
        Returns:
            Proof structure set up for contradiction
        """
        proof = self.create_proof(theorem, ProofType.CONTRADICTION)
        proof.assumptions.append(f"Assume {assumption} (for contradiction)")
        return proof
    
    def prove_by_induction(self, theorem: str, base_case: str) -> MathematicalProof:
        """
        Set up a proof by mathematical induction.
        
        Args:
            theorem: Statement to prove (usually involving n)
            base_case: Base case (usually n=1 or n=0)
            
        Returns:
            Proof structure set up for induction
        """
        proof = self.create_proof(theorem, ProofType.INDUCTION)
        proof.assumptions.append(f"Base case: {base_case}")
        proof.assumptions.append("Inductive hypothesis: Assume P(k) holds for some k ≥ base")
        proof.assumptions.append("Inductive step: Show P(k+1) holds")
        return proof
    
    def format_set(self, math_set: MathematicalSet, format_type: str = 'standard') -> str:
        """
        Format mathematical set for different output types.
        
        Args:
            math_set: MathematicalSet to format
            format_type: Output format ('standard', 'latex', 'html', 'ascii')
            
        Returns:
            Formatted string representation
        """
        if format_type == 'latex':
            return self._format_set_latex(math_set)
        elif format_type == 'html':
            return self._format_set_html(math_set)
        elif format_type == 'ascii':
            return self._format_set_ascii(math_set)
        else:
            return self._format_set_standard(math_set)
    
    def _format_set_standard(self, math_set: MathematicalSet) -> str:
        """Format set in standard notation."""
        if math_set.elements is not None:
            elements_str = ', '.join(str(elem) for elem in math_set.elements)
            return f"{{{elements_str}}}"
        elif math_set.description:
            return math_set.description
        else:
            return math_set.name
    
    def _format_set_latex(self, math_set: MathematicalSet) -> str:
        """Format set for LaTeX output."""
        if math_set.elements is not None:
            elements_str = ', '.join(str(elem) for elem in math_set.elements)
            return f"\\{{{elements_str}\\}}"
        elif math_set.description:
            # Convert set builder notation
            desc = math_set.description.replace('{', '\\{').replace('}', '\\}')
            desc = desc.replace('|', '\\mid')
            return desc
        else:
            return f"\\mathbb{{{math_set.name}}}" if math_set.name in self.STANDARD_SETS else math_set.name
    
    def _format_set_html(self, math_set: MathematicalSet) -> str:
        """Format set for HTML output."""
        if math_set.elements is not None:
            elements_str = ', '.join(str(elem) for elem in math_set.elements)
            return f'<span class="set">{{{elements_str}}}</span>'
        elif math_set.description:
            return f'<span class="set">{math_set.description}</span>'
        else:
            return f'<span class="set-name">{math_set.name}</span>'
    
    def _format_set_ascii(self, math_set: MathematicalSet) -> str:
        """Format set for ASCII output."""
        if math_set.elements is not None:
            elements_str = ', '.join(str(elem) for elem in math_set.elements)
            return f"{{{elements_str}}}"
        elif math_set.description:
            # Convert Unicode to ASCII
            desc = math_set.description.replace('∈', ' in ')
            desc = desc.replace('∀', 'for all ')
            desc = desc.replace('∃', 'exists ')
            return desc
        else:
            # Convert standard set names to ASCII
            ascii_names = {
                'ℕ': 'N', 'ℤ': 'Z', 'ℚ': 'Q', 'ℝ': 'R', 'ℂ': 'C', '∅': 'empty'
            }
            return ascii_names.get(math_set.name, math_set.name)
    
    def format_logical_expression(self, expression: LogicalExpression,
                                format_type: str = 'standard') -> str:
        """Format logical expression for different output types."""
        if format_type == 'latex':
            expr = expression.expression
            expr = expr.replace('∧', '\\land')
            expr = expr.replace('∨', '\\lor')
            expr = expr.replace('¬', '\\neg')
            expr = expr.replace('→', '\\rightarrow')
            expr = expr.replace('↔', '\\leftrightarrow')
            return expr
        
        elif format_type == 'html':
            expr = expression.expression
            return f'<span class="logical-expression">{expr}</span>'
        
        elif format_type == 'ascii':
            expr = expression.expression
            expr = expr.replace('∧', ' AND ')
            expr = expr.replace('∨', ' OR ')
            expr = expr.replace('¬', ' NOT ')
            expr = expr.replace('→', ' IMPLIES ')
            expr = expr.replace('↔', ' IFF ')
            return expr
        
        else:
            return expression.expression
    
    def format_proof(self, proof: MathematicalProof, format_type: str = 'standard') -> str:
        """Format mathematical proof for different output types."""
        lines = []
        
        lines.append(f"Theorem: {proof.theorem}")
        lines.append(f"Proof ({proof.proof_type.value}):")
        
        for assumption in proof.assumptions:
            lines.append(f"  {assumption}")
        
        for step in proof.steps:
            lines.append(f"  {step['number']}. {step['statement']} ({step['justification']})")
        
        if proof.conclusion:
            lines.append(f"  Therefore, {proof.conclusion}")
        
        lines.append("  Q.E.D.")
        
        if format_type == 'latex':
            latex_lines = []
            latex_lines.append("\\begin{proof}")
            for line in lines[1:]:  # Skip theorem line
                latex_lines.append(line.replace('  ', ''))
            latex_lines.append("\\end{proof}")
            return '\n'.join(latex_lines)
        
        elif format_type == 'html':
            html_lines = ['<div class="proof">']
            for line in lines:
                html_lines.append(f'<p>{line}</p>')
            html_lines.append('</div>')
            return '\n'.join(html_lines)
        
        else:
            return '\n'.join(lines)
    
    def check_set_relations(self, set1: MathematicalSet, set2: MathematicalSet) -> Dict[str, bool]:
        """
        Check various relations between two sets.
        
        Args:
            set1, set2: Sets to compare
            
        Returns:
            Dictionary of relation results
        """
        relations = {}
        
        if set1.elements is not None and set2.elements is not None:
            relations['subset'] = set1.is_subset_of(set2)
            relations['superset'] = set2.is_subset_of(set1)
            relations['equal'] = (set(set1.elements) == set(set2.elements))
            relations['disjoint'] = len(set(set1.elements) & set(set2.elements)) == 0
            relations['proper_subset'] = (relations['subset'] and not relations['equal'])
        else:
            # For infinite sets, these would require special handling
            relations = {k: None for k in ['subset', 'superset', 'equal', 'disjoint', 'proper_subset']}
        
        return relations
    
    def power_set(self, math_set: MathematicalSet) -> MathematicalSet:
        """
        Generate power set (set of all subsets).
        
        Args:
            math_set: Set to generate power set for
            
        Returns:
            Power set as MathematicalSet
        """
        if math_set.elements is None or not math_set.is_finite:
            return MathematicalSet(
                name=f"P({math_set.name})",
                description=f"Power set of {math_set.name}",
                is_finite=False
            )
        
        from itertools import combinations
        
        elements = math_set.elements
        power_set_elements = []
        
        # Generate all subsets (combinations of all lengths)
        for r in range(len(elements) + 1):
            for combo in combinations(elements, r):
                power_set_elements.append(list(combo))
        
        return MathematicalSet(
            name=f"P({math_set.name})",
            elements=power_set_elements,
            properties={'cardinality': 2**len(elements)}
        )


# Convenience functions for direct use
def parse_set_notation(set_str: str) -> Optional[MathematicalSet]:
    """Parse set from string notation."""
    processor = SetTheoryProcessor()
    return processor.parse_set(set_str)


def create_set_from_list(elements: List[Any], name: str = "S") -> MathematicalSet:
    """Create set from list of elements."""
    return MathematicalSet(name=name, elements=elements)


def set_union(set1: MathematicalSet, set2: MathematicalSet) -> MathematicalSet:
    """Compute union of two sets."""
    return set1.union(set2)


def set_intersection(set1: MathematicalSet, set2: MathematicalSet) -> MathematicalSet:
    """Compute intersection of two sets."""
    return set1.intersection(set2)


def parse_logic_expression(expression_str: str) -> Optional[LogicalExpression]:
    """Parse logical expression from string."""
    processor = SetTheoryProcessor()
    return processor.parse_logical_expression(expression_str)


# Integration with multi-format renderer
SET_THEORY_FORMAT_EXTENSIONS = {
    'latex': lambda obj: SetTheoryProcessor().format_set(obj, 'latex') if hasattr(obj, 'elements') else str(obj),
    'html': lambda obj: SetTheoryProcessor().format_set(obj, 'html') if hasattr(obj, 'elements') else str(obj),
    'ascii': lambda obj: SetTheoryProcessor().format_set(obj, 'ascii') if hasattr(obj, 'elements') else str(obj)
}