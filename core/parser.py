from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from .models import CodeRepository, Module, CodeElement

class LanguageParser(ABC):
    """Base class for language-specific parsers"""
    
    def __init__(self):
        self.repository: Optional[CodeRepository] = None
        
    @abstractmethod
    def setup(self):
        """Setup any necessary tools or dependencies"""
        pass
        
    @abstractmethod
    def parse_directory(self, directory: str) -> CodeRepository:
        """Parse all code files in a directory"""
        pass
        
    @abstractmethod
    def parse_file(self, file_path: str) -> Module:
        """Parse a single code file"""
        pass
        
    @abstractmethod
    def get_dependencies(self, element: CodeElement) -> Dict[str, List[CodeElement]]:
        """Get all dependencies of a code element"""
        pass
        
    @abstractmethod
    def get_references(self, element: CodeElement) -> Dict[str, List[CodeElement]]:
        """Get all references to a code element"""
        pass
        
    @abstractmethod
    def get_call_graph(self, module: Module) -> Dict[str, List[str]]:
        """Get the function call graph for a module"""
        pass
        
    @abstractmethod
    def get_inheritance_graph(self, module: Module) -> Dict[str, List[str]]:
        """Get the class/interface inheritance graph for a module"""
        pass
        
    @abstractmethod
    def get_symbol_table(self, module: Module) -> Dict[str, CodeElement]:
        """Get a symbol table for a module"""
        pass

class ParserRegistry:
    """Registry for language parsers"""
    
    _parsers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, language: str, parser_class: type):
        """Register a parser for a language"""
        if not issubclass(parser_class, LanguageParser):
            raise ValueError(f"Parser class must inherit from LanguageParser")
        cls._parsers[language.lower()] = parser_class
        
    @classmethod
    def get_parser(cls, language: str) -> Optional[LanguageParser]:
        """Get a parser instance for a language"""
        parser_class = cls._parsers.get(language.lower())
        if parser_class:
            return parser_class()
        return None
        
    @classmethod
    def supported_languages(cls) -> List[str]:
        """Get list of supported languages"""
        return list(cls._parsers.keys())

class ParserPlugin:
    """Decorator for registering parser plugins"""
    
    def __init__(self, language: str):
        self.language = language
        
    def __call__(self, parser_class: type):
        ParserRegistry.register(self.language, parser_class)
        return parser_class 