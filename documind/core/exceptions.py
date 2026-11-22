"""
Core custom exceptions for DocuMind.
"""


class DocuMindException(Exception):
    """Base exception for all DocuMind domain errors."""


class DocumentParsingError(DocuMindException):
    """Raised when parsing or extracting text from a document fails."""


class IndexingError(DocuMindException):
    """Raised when indexing documents into vector or lexical stores fails."""


class RetrievalError(DocuMindException):
    """Raised during query retrieval or re-ranking."""


class CitationVerificationError(DocuMindException):
    """Raised when checking or verifying source-page citations fails."""
