from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

class CodeElement(ABC):
    """Base class for all code elements"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 docstring: str = "", attributes: Dict[str, Any] = None, context: str = ""):
        self.name = name
        self.element_type = element_type
        self.language = language
        self.file = file
        self.line = line
        self.docstring = docstring
        self.attributes = attributes or {}
        self.context = context  # Source code context of the element

class Function(CodeElement):
    """Represents a function or method"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 docstring: str = "", attributes: Dict[str, Any] = None,
                 parameters: List[Dict[str, str]] = None, 
                 return_types: List[str] = None,
                 calls: List[str] = None, 
                 parent_class: Optional[str] = None,
                 context: str = ""):
        super().__init__(name, element_type, language, file, line, docstring, attributes, context)
        self.parameters = parameters or []
        self.return_types = return_types or []
        self.calls = calls or []
        self.parent_class = parent_class

class Class(CodeElement):
    """Represents a class or struct"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 docstring: str = "", attributes: Dict[str, Any] = None,
                 fields: List[Dict[str, str]] = None, 
                 methods: List[Function] = None,
                 parent_classes: List[str] = None, 
                 implements: List[str] = None,
                 context: str = ""):
        super().__init__(name, element_type, language, file, line, docstring, attributes, context)
        self.fields = fields or []
        self.methods = methods or []
        self.parent_classes = parent_classes or []
        self.implements = implements or []

class Interface(CodeElement):
    """Represents an interface"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 docstring: str = "", attributes: Dict[str, Any] = None,
                 methods: List[Function] = None, 
                 extends: List[str] = None,
                 context: str = ""):
        super().__init__(name, element_type, language, file, line, docstring, attributes, context)
        self.methods = methods or []
        self.extends = extends or []

class Module(CodeElement):
    """Represents a module/package/namespace"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 docstring: str = "", attributes: Dict[str, Any] = None,
                 imports: List[str] = None, 
                 functions: List[Function] = None,
                 classes: List[Class] = None, 
                 interfaces: List[Interface] = None,
                 submodules: List[str] = None, 
                 files: List[str] = None,
                 context: str = ""):
        super().__init__(name, element_type, language, file, line, docstring, attributes, context)
        self.imports = imports or []
        self.functions = functions or []
        self.classes = classes or []
        self.interfaces = interfaces or []
        self.submodules = submodules or []
        self.files = files or []

class Variable(CodeElement):
    """Represents a variable or constant"""
    def __init__(self, name: str, element_type: str, language: str, file: str, line: int, 
                 var_type: str, docstring: str = "", attributes: Dict[str, Any] = None, 
                 is_constant: bool = False, value: Optional[str] = None,
                 context: str = ""):
        super().__init__(name, element_type, language, file, line, docstring, attributes, context)
        self.var_type = var_type
        self.is_constant = is_constant
        self.value = value

@dataclass
class Annotation(CodeElement):
    """Represents an annotation/decorator"""
    target_type: str  # function, class, field, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeRepository:
    """Represents an entire code repository"""
    root_path: str
    language: str
    modules: Dict[str, Module] = field(default_factory=dict)
    global_symbols: Dict[str, CodeElement] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) 