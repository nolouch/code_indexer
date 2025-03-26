import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from ...core.parser import LanguageParser, ParserPlugin
from ...core.models import (
    CodeRepository, Module, Function, Class,
    Interface, CodeElement
)

logger = logging.getLogger(__name__)

@ParserPlugin(language="go")
class GoParser(LanguageParser):
    """Go language parser implementation"""
    
    def __init__(self):
        super().__init__()
        self.ast_tool_path: Optional[Path] = None
        
    def setup(self):
        """Setup the Go AST analyzer tool"""
        tools_dir = Path(__file__).parent.parent.parent / "tools" / "go" / "analyzer"
        self.ast_tool_path = tools_dir / "ast_analyzer"
        
        try:
            subprocess.run(
                ["go", "build", "-o", str(self.ast_tool_path)],
                cwd=tools_dir.parent,  # Use go directory for build context
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building AST analyzer: {e.stderr.decode()}")
            raise
            
    def parse_directory(self, directory: str) -> CodeRepository:
        """Parse all Go packages in a directory"""
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory {directory} does not exist")
            
        self.repository = CodeRepository(
            root_path=str(path),
            language="go"
        )
        
        # Get all Go packages in the directory
        packages = self._get_packages(str(path))
        
        # Parse each package
        for pkg_path in packages:
            try:
                module = self._parse_package(pkg_path)
                self.repository.modules[pkg_path] = module
            except Exception as e:
                logger.error(f"Error parsing package {pkg_path}: {e}")
                
        return self.repository
        
    def parse_file(self, file_path: str) -> Module:
        """Parse a single Go file"""
        # For Go, we always parse at package level
        pkg_dir = Path(file_path).parent
        return self._parse_package(str(pkg_dir))
        
    def _get_packages(self, directory: str) -> List[str]:
        """Get all Go packages in the directory"""
        try:
            result = subprocess.run(
                ['go', 'list', './...'],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True
            )
            return [pkg for pkg in result.stdout.strip().split('\n') if pkg]
        except subprocess.CalledProcessError as e:
            logger.error(f"Error listing Go packages: {e}")
            return []
            
    def _parse_package(self, package_path: str) -> Module:
        """Parse a Go package using the AST analyzer"""
        if not self.ast_tool_path:
            self.setup()
            
        try:
            # Run AST analyzer
            result = subprocess.run(
                [str(self.ast_tool_path), "-pkg", package_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse JSON output
            ast_data = json.loads(result.stdout)
            
            # Convert AST data to our model
            for pkg_path, pkg_info in ast_data.items():
                module = Module(
                    name=pkg_info["name"],
                    element_type="package",
                    language="go",
                    file=pkg_info["files"][0] if pkg_info["files"] else "",
                    line=1,  # Package declaration is usually at line 1
                    imports=pkg_info["imports"],
                    files=pkg_info["files"]
                )
                
                # Add functions
                for func_info in pkg_info["functions"]:
                    func = Function(
                        name=func_info["name"],
                        element_type="function",
                        language="go",
                        file=func_info["file"],
                        line=func_info["line"],
                        docstring=func_info.get("docstring", ""),
                        calls=func_info.get("calls", []),
                        parent_class=func_info.get("receiver_type")
                    )
                    module.functions.append(func)
                    
                # Add structs
                for struct_info in pkg_info["structs"]:
                    struct = Class(
                        name=struct_info["name"],
                        element_type="struct",
                        language="go",
                        file=struct_info["file"],
                        line=struct_info["line"],
                        docstring=struct_info.get("docstring", ""),
                        fields=struct_info["fields"]
                    )
                    module.classes.append(struct)
                    
                # Add interfaces
                for iface_info in pkg_info["interfaces"]:
                    iface = Interface(
                        name=iface_info["name"],
                        element_type="interface",
                        language="go",
                        file=iface_info["file"],
                        line=iface_info["line"],
                        docstring=iface_info.get("docstring", ""),
                        methods=[
                            Function(
                                name=m,
                                element_type="function",
                                language="go",
                                file=iface_info["file"],
                                line=iface_info["line"]
                            )
                            for m in iface_info["methods"]
                        ]
                    )
                    module.interfaces.append(iface)
                    
                return module
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running AST analyzer: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing AST analyzer output: {e}")
            raise
            
    def get_dependencies(self, element: CodeElement) -> Dict[str, List[CodeElement]]:
        """Get all dependencies of a code element"""
        deps = {
            "imports": [],
            "calls": [],
            "implements": [],
            "embeds": []
        }
        
        if isinstance(element, Module):
            deps["imports"] = [
                CodeElement(
                    name=imp,
                    element_type="package",
                    language="go",
                    file="",
                    line=0
                )
                for imp in element.imports
            ]
            
        elif isinstance(element, Function):
            deps["calls"] = [
                CodeElement(
                    name=call,
                    element_type="function",
                    language="go",
                    file="",
                    line=0
                )
                for call in element.calls
            ]
            
        elif isinstance(element, Class):
            # Find interface implementations
            for module in self.repository.modules.values():
                for interface in module.interfaces:
                    if self._struct_implements_interface(element, interface):
                        deps["implements"].append(interface)
                        
        return deps
        
    def get_references(self, element: CodeElement) -> Dict[str, List[CodeElement]]:
        """Get all references to a code element"""
        refs = {
            "called_by": [],
            "implemented_by": [],
            "embedded_by": []
        }
        
        if isinstance(element, Function):
            # Find function calls
            for module in self.repository.modules.values():
                for func in module.functions:
                    if element.name in func.calls:
                        refs["called_by"].append(func)
                        
        elif isinstance(element, Interface):
            # Find implementations
            for module in self.repository.modules.values():
                for struct in module.classes:
                    if self._struct_implements_interface(struct, element):
                        refs["implemented_by"].append(struct)
                        
        return refs
        
    def get_call_graph(self, module: Module) -> Dict[str, List[str]]:
        """Get the function call graph for a module"""
        call_graph = {}
        
        # Add all functions to the graph
        for func in module.functions:
            call_graph[func.name] = func.calls
            
        # Add method calls
        for cls in module.classes:
            for method in cls.methods:
                call_graph[f"{cls.name}.{method.name}"] = method.calls
                
        return call_graph
        
    def get_inheritance_graph(self, module: Module) -> Dict[str, List[str]]:
        """Get the interface implementation graph for a module"""
        inheritance_graph = {}
        
        # Add interface implementations
        for interface in module.interfaces:
            implementations = []
            for cls in module.classes:
                if self._struct_implements_interface(cls, interface):
                    implementations.append(cls.name)
            inheritance_graph[interface.name] = implementations
            
        return inheritance_graph
        
    def get_symbol_table(self, module: Module) -> Dict[str, CodeElement]:
        """Get a symbol table for a module"""
        symbols = {}
        
        # Add functions
        for func in module.functions:
            symbols[func.name] = func
            
        # Add structs and their methods
        for struct in module.classes:
            symbols[struct.name] = struct
            for method in struct.methods:
                symbols[f"{struct.name}.{method.name}"] = method
                
        # Add interfaces
        for interface in module.interfaces:
            symbols[interface.name] = interface
            
        return symbols
        
    def _struct_implements_interface(self, struct: Class, interface: Interface) -> bool:
        """Check if a struct implements an interface"""
        struct_methods = {m.name for m in struct.methods}
        interface_methods = {m.name for m in interface.methods}
        return interface_methods.issubset(struct_methods) 