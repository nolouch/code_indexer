import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from core.parser import LanguageParser, ParserPlugin
from core.models import (
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
        self.repository = None
        
    def setup(self):
        """Setup the Go AST analyzer tool"""
        tools_dir = Path(__file__).parent.parent.parent / "tools" / "go" / "analyzer"
        self.ast_tool_path = tools_dir / "ast_analyzer"
        
        # Check if ast_analyzer already exists
        if self.ast_tool_path.exists():
            logger.info(f"Found existing AST analyzer at {self.ast_tool_path}")
            self.initialized = True
            return
            
        logger.info(f"Building AST analyzer in {tools_dir}")
        
        try:
            # Ensure working directory is analyzer directory
            result = subprocess.run(
                ["go", "build", "-o", str(self.ast_tool_path), "ast_analyzer.go"],
                cwd=tools_dir,  # Use analyzer directory for build context
                check=True,
                capture_output=True
            )
            
            if self.ast_tool_path.exists():
                logger.info(f"Successfully built AST analyzer at {self.ast_tool_path}")
                self.initialized = True
            else:
                logger.error(f"AST analyzer was not built at {self.ast_tool_path}")
                self.initialized = False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building AST analyzer: {e.stderr.decode() if hasattr(e.stderr, 'decode') else e.stderr}")
            logger.error(f"Command output: {e.stdout.decode() if hasattr(e.stdout, 'decode') else e.stdout}")
            self.initialized = False
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
        
        # If no packages found, add a default placeholder module
        if not packages:
            logger.warning(f"No Go packages found in directory: {directory}")
            default_module = self._create_default_module(directory)
            self.repository.modules[directory] = default_module
            return self.repository
            
        # Parse each package
        modules_parsed = 0
        for pkg_path in packages:
            try:
                module = self._parse_package(pkg_path)
                if module:
                    self.repository.modules[pkg_path] = module
                    modules_parsed += 1
                else:
                    # Create a default module if parsing failed
                    logger.warning(f"Failed to parse package {pkg_path}, creating a default module")
                    default_module = self._create_default_module(pkg_path)
                    self.repository.modules[pkg_path] = default_module
            except Exception as e:
                logger.error(f"Error parsing package {pkg_path}: {e}")
                # Create a default module on exception
                default_module = self._create_default_module(pkg_path)
                self.repository.modules[pkg_path] = default_module
        
        # If no modules were successfully parsed, add a default placeholder
        if modules_parsed == 0 and not self.repository.modules:
            logger.warning(f"No modules parsed successfully in directory: {directory}")
            default_module = self._create_default_module(directory)
            self.repository.modules[directory] = default_module
                
        return self.repository
        
    def _create_default_module(self, path: str) -> Module:
        """Create a default module for cases where parsing fails"""
        logger.info(f"Creating default module for path: {path}")
        module_name = os.path.basename(path)
        if not module_name or module_name == '.':
            module_name = "default"
            
        return Module(
            name=module_name,
            element_type="package",
            language="go",
            file=path,
            line=1,
            imports=[],
            files=[path]
        )
        
    def parse_file(self, file_path: str) -> Module:
        """Parse a single Go file"""
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist")
            return None
        
        # For Go, we always parse at package level
        pkg_dir = Path(file_path).parent
        
        # Find the go.mod directory to properly resolve the module
        go_mod_dir = self._find_go_mod_dir(pkg_dir)
        
        # Create a repository if it doesn't exist
        if not self.repository:
            root_path = str(go_mod_dir) if go_mod_dir else str(pkg_dir.parent)
            self.repository = CodeRepository(
                root_path=root_path,
                language="go"
            )
            
        # Use the directory containing the file for parsing
        module = self._parse_package(str(pkg_dir))
        
        # Add module to repository for dependency analysis
        if module and module.name:
            self.repository.modules[module.name] = module
            
        return module
        
    def _find_go_mod_dir(self, directory: Path) -> Optional[Path]:
        """Find the directory containing go.mod by walking up the directory tree"""
        current_dir = directory
        # Check up to 10 parent directories
        for _ in range(10):
            go_mod_path = current_dir / "go.mod"
            if go_mod_path.exists():
                return current_dir
            
            # Move up one directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
            
        return None
        
    def _get_packages(self, directory: str) -> List[str]:
        """Get a list of Go packages"""
        # First check if the directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return []
            
        # Check if it's a valid Go module directory
        if os.path.exists(os.path.join(directory, "go.mod")):
            logger.info(f"Found go.mod in {directory}, trying to use 'go list'")
            try:
                # Try to use the go list command to get package list
                env = os.environ.copy()
                env["GO111MODULE"] = "on"
                cmd = ["go", "list", "-f", "{{.ImportPath}}:{{.Dir}}", "./..."]
                result = subprocess.run(
                    cmd,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    package_paths = {}
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if ':' in line:
                            import_path, dir_path = line.split(':', 1)
                            # Store both import path and directory path
                            package_paths[import_path] = dir_path
                            
                    logger.info(f"Found {len(package_paths)} Go packages using 'go list'")
                    
                    # We'll use the directory paths for analysis since our AST tool works better with paths
                    return list(package_paths.values())
                else:
                    logger.warning(f"'go list' failed or returned empty result: {result.stderr}")
                    logger.info("Falling back to filesystem scan")
            except Exception as e:
                logger.warning(f"Error running 'go list': {e}")
                logger.info("Falling back to filesystem scan")
        else:
            logger.info(f"No go.mod found in {directory}, using filesystem scan")
            
        # If go list fails or go.mod is not found, use filesystem scanning
        return self._get_packages_by_filesystem(directory)
            
    def _get_packages_by_filesystem(self, repo_path: str) -> List[str]:
        """Scan filesystem for Go files and return a list of packages"""
        # Ensure path is absolute
        repo_path = os.path.abspath(repo_path)
        logger.info(f"Scanning for Go files in {repo_path}")
        
        packages = set()
        go_files = []
        
        # Walk through directories looking for Go files
        for root, _, files in os.walk(repo_path):
            # Exclude vendor and hidden directories
            if 'vendor' in root.split(os.path.sep) or any(p.startswith('.') for p in root.split(os.path.sep)):
                continue
                
            # Check if there are Go files
            has_go_files = False
            dir_go_files = []
            for file in files:
                if file.endswith('.go') and not file.endswith('_test.go'):
                    has_go_files = True
                    dir_go_files.append(os.path.join(root, file))
                    
            if has_go_files:
                # Add this directory as a package
                rel_path = os.path.relpath(root, repo_path)
                if rel_path == '.':
                    packages.add(repo_path)
                else:
                    packages.add(os.path.join(repo_path, rel_path))
                go_files.extend(dir_go_files)
                
        logger.info(f"Found {len(packages)} Go packages by filesystem scan")
        logger.debug(f"Found {len(go_files)} Go files")
        
        return list(packages)
            
    def _parse_package(self, package_path: str) -> Module:
        """Parse a Go package using the AST analyzer"""
        if not package_path:
            logger.error("Empty package path provided")
            return self._create_default_module("unknown")
            
        if not self.ast_tool_path:
            self.setup()
            
        if not self.initialized:
            logger.error("AST analyzer not initialized. Cannot parse package.")
            return self._create_default_module(package_path)
            
        # Check if package_path is a local filesystem path
        is_filesystem_path = os.path.exists(package_path)
        abs_path = os.path.abspath(package_path) if is_filesystem_path else ""
        
        logger.info(f"Parsing {'directory' if is_filesystem_path else 'package'}: {package_path}")
        
        try:
            # Prepare command line arguments
            cmd = [str(self.ast_tool_path)]
            
            if is_filesystem_path:
                # For filesystem paths, use the -dir parameter
                cmd.extend(["-dir", abs_path])
                logger.debug(f"Using directory mode with path: {abs_path}")
            else:
                # For Go package paths, we need to use directory mode with GOMOD to resolve properly
                # Try to get the directory containing the go.mod file
                if self.repository and hasattr(self.repository, "root_path"):
                    root_path = self.repository.root_path
                    # Need to use directory mode with proper Go environment
                    cmd.extend(["-dir", root_path])
                    logger.debug(f"Using directory mode with root path for package: {package_path}")
                else:
                    # Use the pkg parameter as a fallback, though it may not work correctly
                    cmd.extend(["-pkg", package_path])
                    logger.debug(f"Using package mode with path: {package_path}")
                
            # Run the AST analyzer with proper Go environment
            env = os.environ.copy()
            # Set GO111MODULE to "on" to ensure module-aware mode
            env["GO111MODULE"] = "on"
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exceptions, we'll handle errors manually
                env=env
            )
            
            # Check if the command was successful
            if result.returncode != 0:
                logger.warning(f"AST analyzer failed with code {result.returncode}: {result.stderr}")
                logger.debug(f"Command: {' '.join(cmd)}")
                return self._create_default_module(package_path)
                
            # If output is empty, package may not have been found
            if not result.stdout or not result.stdout.strip():
                logger.warning(f"No output from AST analyzer for {package_path}")
                return self._create_default_module(package_path)
                
            # Parse JSON output
            try:
                logger.debug("Parsing AST analyzer JSON output")
                ast_data = json.loads(result.stdout)
                logger.debug(f"JSON data type: {type(ast_data)}")
                
                # Safety check: if AST data is None or empty, return early
                if not ast_data:
                    logger.warning(f"AST data is empty for {package_path}")
                    return self._create_default_module(package_path)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from AST analyzer: {e}")
                logger.debug(f"JSON snippet: {result.stdout[:200]}...")
                return self._create_default_module(package_path)
                
            # Check if there is package data
            if not ast_data:
                logger.warning(f"No packages found in {package_path}")
                return self._create_default_module(package_path)
                
            # Get the first package information
            try:
                logger.debug(f"AST data keys: {list(ast_data.keys()) if isinstance(ast_data, dict) else 'Not a dict'}")
                
                # Ensure ast_data is a dictionary
                if not isinstance(ast_data, dict) or not ast_data:
                    logger.warning(f"AST data for {package_path} is not a valid dictionary or is empty")
                    return self._create_default_module(package_path)
                
                # For a package path, we need to look for the specific package
                if not is_filesystem_path and "." in package_path:
                    # See if we can find the package by name
                    pkg_path = None
                    for key in ast_data.keys():
                        if key.endswith(package_path) or package_path.endswith(key):
                            pkg_path = key
                            break
                    
                    if not pkg_path:
                        logger.warning(f"Package {package_path} not found in AST data")
                        # Still try with first package as fallback
                        if ast_data:
                            pkg_path = next(iter(ast_data))
                        else:
                            logger.warning("AST data is empty, cannot extract package information")
                            return self._create_default_module(package_path)
                else:
                    # Get the first package key
                    if ast_data:
                        pkg_path = next(iter(ast_data))
                    else:
                        logger.warning("AST data is empty, cannot extract package information")
                        return self._create_default_module(package_path)
                
                # Ensure pkg_info is a dictionary
                pkg_info = ast_data.get(pkg_path)
                if not isinstance(pkg_info, dict) or not pkg_info:
                    logger.warning(f"Package info for {pkg_path} is not a valid dictionary or is empty")
                    return self._create_default_module(package_path)
                
                logger.debug(f"Package info type: {type(pkg_info)}")
                logger.info(f"Found package {pkg_info.get('name', 'unknown')} at {pkg_path}")
                
                # Create module object even if package has no files
                # Instead of returning None, create a minimal module
                module_name = pkg_info.get("name", os.path.basename(package_path))
                if not module_name:
                    module_name = os.path.basename(package_path)
                
                # Ensure "files" is a list
                files_list = pkg_info.get("files", [])
                if not isinstance(files_list, list):
                    logger.warning(f"Files for package {pkg_path} is not a list")
                    files_list = []
                    
                # Normalize file paths
                files = []
                for file_path in files_list:
                    if not file_path:  # Skip empty file paths
                        continue
                        
                    # Ensure file path is absolute
                    if os.path.isabs(file_path):
                        files.append(file_path)
                    elif is_filesystem_path:
                        # If it's a relative path, convert to absolute
                        files.append(os.path.join(abs_path, file_path))
                    else:
                        files.append(file_path)
                
                # Create module object with what we have, even if files list is empty
                module = Module(
                    name=module_name,
                    element_type="package",
                    language="go",
                    file=files[0] if files else package_path,  # Use package_path as fallback
                    line=1,  # Package declaration is usually at line 1
                    imports=pkg_info.get("imports", []),
                    files=files if files else [package_path]  # Use package_path as fallback
                )
                
                # Ensure functions, structs, and interfaces are lists before processing
                funcs_list = pkg_info.get("functions", [])
                if not isinstance(funcs_list, list):
                    logger.warning(f"Functions for package {pkg_path} is not a list")
                    funcs_list = []
                
                # Add functions
                for func_info in funcs_list:
                    # Skip if function info is None or not a dictionary
                    if not func_info or not isinstance(func_info, dict):
                        continue
                        
                    # Normalize file path
                    func_file = func_info.get("file", "")
                    if func_file and not os.path.isabs(func_file) and is_filesystem_path:
                        func_file = os.path.join(abs_path, func_file)
                        
                    func = Function(
                        name=func_info.get("name", "unknown"),
                        element_type="function",
                        language="go",
                        file=func_file,
                        line=func_info.get("line", 1),
                        docstring=func_info.get("docstring", ""),
                        calls=func_info.get("calls", []),
                        parent_class=func_info.get("receiver_type", ""),
                        context=func_info.get("context", "")
                    )
                    module.functions.append(func)
                
                structs_list = pkg_info.get("structs", [])
                if not isinstance(structs_list, list):
                    logger.warning(f"Structs for package {pkg_path} is not a list")
                    structs_list = []
                    
                # Add structs
                for struct_info in structs_list:
                    # Skip if struct info is None or not a dictionary
                    if not struct_info or not isinstance(struct_info, dict):
                        continue
                        
                    # Normalize file path
                    struct_file = struct_info.get("file", "")
                    if struct_file and not os.path.isabs(struct_file) and is_filesystem_path:
                        struct_file = os.path.join(abs_path, struct_file)
                        
                    # Ensure fields is a list
                    fields = struct_info.get("fields", [])
                    if not isinstance(fields, list):
                        fields = []
                        
                    # Ensure methods is a list
                    methods_list = struct_info.get("methods", [])
                    if not isinstance(methods_list, list):
                        logger.warning(f"Methods for struct {struct_info.get('name', 'unknown')} is not a list")
                        methods_list = []
                    
                    # Convert method names to Function objects
                    parsed_methods = []
                    for method_item in methods_list:
                        # Check what type of method data we have
                        if isinstance(method_item, str):
                            # Just a method name, create a minimal Function
                            logger.info(f"Got method name as string: {method_item}")
                            method_func = Function(
                                name=method_item,
                                element_type="method",
                                language="go",
                                file=struct_file,
                                line=1,  # Unknown line number
                                docstring="",
                                context=f"func ({struct_info.get('name', 'unknown')}) {method_item}()"
                            )
                            parsed_methods.append(method_func)
                        elif isinstance(method_item, dict):
                            # Method info as dictionary
                            method_name = method_item.get("name", "unknown")
                            method_file = method_item.get("file", struct_file)
                            if method_file and not os.path.isabs(method_file) and is_filesystem_path:
                                method_file = os.path.join(abs_path, method_file)
                                
                            method_func = Function(
                                name=method_name,
                                element_type="method",
                                language="go",
                                file=method_file,
                                line=method_item.get("line", 1),
                                docstring=method_item.get("docstring", ""),
                                calls=method_item.get("calls", []),
                                parent_class=struct_info.get("name", ""),
                                context=method_item.get("context", "")
                            )
                            parsed_methods.append(method_func)
                        else:
                            logger.warning(f"Unknown method format for struct {struct_info.get('name', 'unknown')}: {type(method_item)}")
                        
                    struct = Class(
                        name=struct_info.get("name", "unknown"),
                        element_type="struct",
                        language="go",
                        file=struct_file,
                        line=struct_info.get("line", 1),
                        docstring=struct_info.get("docstring", ""),
                        fields=[{
                            "name": f["name"],
                            "type": f["type"],
                            "tag": f.get("tag", "")
                        } for f in fields],
                        methods=parsed_methods,
                        context=struct_info.get("context", "")
                    )
                    module.classes.append(struct)
                
                ifaces_list = pkg_info.get("interfaces", [])
                if not isinstance(ifaces_list, list):
                    logger.warning(f"Interfaces for package {pkg_path} is not a list")
                    ifaces_list = []
                    
                # Add interfaces
                for iface_info in ifaces_list:
                    # Skip if interface info is None or not a dictionary
                    if not iface_info or not isinstance(iface_info, dict):
                        continue
                        
                    # Normalize file path
                    iface_file = iface_info.get("file", "")
                    if iface_file and not os.path.isabs(iface_file) and is_filesystem_path:
                        iface_file = os.path.join(abs_path, iface_file)
                        
                    # Ensure methods is a list
                    methods_list = iface_info.get("methods", [])
                    if not isinstance(methods_list, list):
                        methods_list = []
                        
                    # Create method objects
                    methods = []
                    for method_name in methods_list:
                        if method_name and isinstance(method_name, str):  # Check if method name is a valid string
                            methods.append(
                                Function(
                                    name=method_name,
                                    element_type="function",
                                    language="go",
                                    file=iface_file,
                                    line=iface_info.get("line", 1),
                                    context=iface_info.get("context", "")
                                )
                            )
                        
                    iface = Interface(
                        name=iface_info.get("name", "unknown"),
                        element_type="interface",
                        language="go",
                        file=iface_file,
                        line=iface_info.get("line", 1),
                        docstring=iface_info.get("docstring", ""),
                        methods=methods,
                        context=iface_info.get("context", "")
                    )
                    module.interfaces.append(iface)
                    
                return module
                
            except (KeyError, StopIteration) as e:
                logger.error(f"Error processing package data: {e}")
                logger.debug(f"Error details: {type(e).__name__} - {str(e)}")
                logger.debug(f"AST data: {ast_data}")
                return self._create_default_module(package_path)
                
        except Exception as e:
            logger.error(f"Error in _parse_package for {package_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._create_default_module(package_path)
            
    def get_dependencies(self, element: CodeElement) -> Dict[str, List[CodeElement]]:
        """Get all dependencies of a code element"""
        deps = {
            "imports": [],
            "calls": [],
            "implements": [],
            "embeds": []
        }
        
        if isinstance(element, Module):
            # Add import dependencies
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
            # Add call dependencies
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
            if hasattr(element, "implements") and element.implements:
                # Direct implementation list
                deps["implements"] = [
                    CodeElement(
                        name=impl,
                        element_type="interface",
                        language="go",
                        file="",
                        line=0
                    )
                    for impl in element.implements
                ]
            elif self.repository:
                # Infer implementations from method signatures
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
        """Get the inheritance graph for a module"""
        inheritance_graph = {}
        
        # Add interface inheritance
        for iface in module.interfaces:
            inheritance_graph[iface.name] = iface.extends
            
        # Add struct embedding
        for struct in module.classes:
            if hasattr(struct, "parent_classes"):
                inheritance_graph[struct.name] = struct.parent_classes
                
        return inheritance_graph
        
    def get_symbol_table(self, module: Module) -> Dict[str, CodeElement]:
        """Get a symbol table for a module"""
        symbols = {}
        
        # Add functions
        for func in module.functions:
            symbols[func.name] = func
            
        # Add structs
        for struct in module.classes:
            symbols[struct.name] = struct
            
            # Add struct methods
            for method in struct.methods:
                symbols[f"{struct.name}.{method.name}"] = method
                
        # Add interfaces
        for iface in module.interfaces:
            symbols[iface.name] = iface
            
        return symbols
        
    def _struct_implements_interface(self, struct: Class, interface: Interface) -> bool:
        """Check if a struct implements an interface"""
        if not hasattr(struct, "methods") or not struct.methods:
            return False
            
        struct_methods = {method.name for method in struct.methods}
        interface_methods = {method.name for method in interface.methods}
        
        # All interface methods must be implemented by the struct
        return interface_methods.issubset(struct_methods) 