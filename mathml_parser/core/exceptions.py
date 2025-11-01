"""
Exception classes and result types for MathML Parser
====================================================

This module defines custom exceptions and result objects for the MathML parser
to provide comprehensive error handling and structured result reporting.
"""

from typing import Optional, Dict, Any


class MathParseError(Exception):
    """
    Custom exception for mathematical parsing errors.
    
    Provides detailed error information including position, suggestions,
    and context for debugging and user feedback.
    """
    
    def __init__(self, message: str, position: Optional[int] = None, 
                 suggestion: Optional[str] = None, expression: Optional[str] = None,
                 error_type: Optional[str] = None):
        """
        Initialize a mathematical parsing error.
        
        Args:
            message (str): Error description
            position (int, optional): Character position where error occurred
            suggestion (str, optional): Helpful suggestion for fixing the error
            expression (str, optional): Original expression that caused the error
            error_type (str, optional): Type/category of the error
        """
        self.message = message
        self.position = position
        self.suggestion = suggestion
        self.expression = expression
        self.error_type = error_type or "UnknownError"
        super().__init__(message)
    
    def __str__(self):
        """Return a formatted error message."""
        result = self.message
        if self.position is not None:
            result += f" (at position {self.position})"
        if self.suggestion:
            result += f"\nSuggestion: {self.suggestion}"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary format.
        
        Returns:
            dict: Error information as dictionary
        """
        return {
            'error': self.message,
            'position': self.position,
            'suggestion': self.suggestion,
            'expression': self.expression
        }


class MathParseResult:
    """
    Result object for mathematical expression parsing.
    
    Provides structured results for both successful and failed parsing attempts
    with comprehensive information for error handling and debugging.
    """
    
    def __init__(self, success: bool, expression: str, mathml: Optional[str] = None,
                 error: Optional[str] = None, position: Optional[int] = None,
                 suggestion: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Initialize a parsing result.
        
        Args:
            success (bool): Whether parsing was successful
            expression (str): Original mathematical expression
            mathml (str, optional): Generated MathML (if successful)
            error (str, optional): Error message (if failed)
            position (int, optional): Error position (if failed)
            suggestion (str, optional): Helpful suggestion (if failed)
            metadata (dict, optional): Additional metadata
        """
        self.success = success
        self.expression = expression
        self.mathml = mathml
        self.error = error
        self.position = position
        self.suggestion = suggestion
        self.metadata = metadata or {}
    
    def __str__(self):
        """Return a string representation of the result."""
        if self.success:
            return f"Success: '{self.expression}' -> MathML generated"
        else:
            result = f"Failed: '{self.expression}' -> {self.error}"
            if self.suggestion:
                result += f" (Suggestion: {self.suggestion})"
            return result
    
    def __bool__(self):
        """Return the success status when used in boolean context."""
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns:
            dict: Complete result information as dictionary
        """
        result = {
            'success': self.success,
            'expression': self.expression
        }
        
        if self.success:
            result['mathml'] = self.mathml
        else:
            result['error'] = self.error
            if self.position is not None:
                result['position'] = self.position
            if self.suggestion:
                result['suggestion'] = self.suggestion
        
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result
    
    @classmethod
    def success_result(cls, expression: str, mathml: str, metadata: Optional[Dict] = None):
        """
        Create a successful parsing result.
        
        Args:
            expression (str): Original expression
            mathml (str): Generated MathML
            metadata (dict, optional): Additional metadata
            
        Returns:
            MathParseResult: Success result object
        """
        return cls(success=True, expression=expression, mathml=mathml, metadata=metadata)
    
    @classmethod
    def error_result(cls, expression: str, error: str, position: Optional[int] = None,
                     suggestion: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Create a failed parsing result.
        
        Args:
            expression (str): Original expression
            error (str): Error message
            position (int, optional): Error position
            suggestion (str, optional): Helpful suggestion
            metadata (dict, optional): Additional metadata
            
        Returns:
            MathParseResult: Error result object
        """
        return cls(success=False, expression=expression, error=error,
                  position=position, suggestion=suggestion, metadata=metadata)
    
    @classmethod
    def from_exception(cls, expression: str, exception: Exception):
        """
        Create a result from an exception.
        
        Args:
            expression (str): Original expression
            exception (Exception): The exception that occurred
            
        Returns:
            MathParseResult: Error result object
        """
        if isinstance(exception, MathParseError):
            return cls.error_result(
                expression=expression,
                error=exception.message,
                position=exception.position,
                suggestion=exception.suggestion
            )
        else:
            return cls.error_result(
                expression=expression,
                error=str(exception)
            )


class ValidationError(MathParseError):
    """Exception raised during input validation."""
    pass


class GrammarError(MathParseError):
    """Exception raised due to grammar-related issues."""
    pass


class TransformationError(MathParseError):
    """Exception raised during MathML transformation."""
    pass


# Convenience functions for creating results
def success(expression: str, mathml: str, **metadata) -> MathParseResult:
    """Create a success result."""
    return MathParseResult.success_result(expression, mathml, metadata)


def error(expression: str, message: str, position: int = None, 
          suggestion: str = None, **metadata) -> MathParseResult:
    """Create an error result."""
    return MathParseResult.error_result(expression, message, position, suggestion, metadata)