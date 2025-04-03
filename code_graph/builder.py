from typing import Dict, List, Optional, Any
import networkx as nx
import logging
from pathlib import Path
from core.models import CodeRepository, Module, CodeElement
from .embedders import create_embedder

logger = logging.getLogger(__name__)

class SemanticGraphBuilder:
    """Builds semantic graphs from code analysis results"""
    
    def __init__(self, code_embedder=None, doc_embedder=None):
        """Initialize builder with embedders.
        
        Args:
            code_embedder: Code embedder, will create default if None
            doc_embedder: Documentation embedder, will create default if None
        """
        # Use the factory to create appropriate embedders based on configuration
        self.code_embedder = code_embedder or create_embedder("code")
        self.doc_embedder = doc_embedder or create_embedder("doc")
        
    def build_from_repository(self, code_repo: CodeRepository) -> nx.DiGraph:
        """Build semantic graph from a code repository.
        
        Args:
            code_repo: Code repository to analyze
            
        Returns:
            nx.DiGraph: The semantic graph with nodes and relationships
        """
        semantic_graph = nx.DiGraph()
        
        if not code_repo or not hasattr(code_repo, 'modules'):
            logger.warning("Code repository does not have modules attribute")
            return semantic_graph
        
        logger.debug(f"Converting code repository to semantic graph: type={type(code_repo)}, modules={len(code_repo.modules) if hasattr(code_repo, 'modules') else 'N/A'}")
        
        # Create a dictionary to map function names to their IDs for later call relationship building
        function_name_to_id = {}
        method_name_to_id = {}
        
        # Add nodes for modules
        for module_name, module in code_repo.modules.items():
            # Skip if module is None
            if module is None:
                logger.warning(f"Module {module_name} is None, skipping")
                continue

            logger.debug(f"Processing module: name={module_name}, type={type(module)}")
                
            module_id = f"module:{module_name}"
            node_id = module_id  # Use the same ID format for the node_id field
            
            # Handle case where module attributes are None
            module_file_path = getattr(module, 'file_path', None) or str(module_name)
            
            semantic_graph.add_node(
                module_id,
                type='module',
                name=module_name,
                node_id=node_id,
                file_path=module_file_path,
                line_number=0,  # Default line number for modules
                code_context=f"Module {module_name}",
                doc_context=""  # Empty doc context as default
            )
            
            # Add functions
            if not hasattr(module, 'functions'):
                logger.warning(f"Module {module_name} has no functions attribute")
                continue
                
            logger.debug(f"Processing functions for module {module_name}: count={len(module.functions)}")
            for func in module.functions:
                try:
                    func_name = func.name
                    func_id = f"func:{module_name}:{func_name}"
                    node_id = func_id  # Use the same ID format for the node_id field
                    
                    # Store function name to ID mapping for call relationships
                    function_name_to_id[func_name] = func_id
                    
                    # Handle case where function attributes might be None
                    func_file_path = getattr(func, 'file', None) or module_file_path
                    func_source = getattr(func, 'context', None) or f"Function {func_name}"
                    func_line_number = getattr(func, 'line', 0) or 0  # Default to 0 if not available
                    func_docstring = getattr(func, 'docstring', None) or ""
                    
                    # Store function calls information for later processing
                    func_calls = getattr(func, 'calls', [])
                    
                    semantic_graph.add_node(
                        func_id,
                        type='function',
                        name=func_name,
                        node_id=node_id,
                        file_path=func_file_path,
                        line_number=func_line_number,
                        code_context=func_source,
                        doc_context=func_docstring,
                        calls=func_calls  # Store the calls in node attributes
                    )
                    # Add edge from module to function
                    semantic_graph.add_edge(module_id, func_id, type='contains')
                except AttributeError as e:
                    logger.error(f"AttributeError processing function in module {module_name}: {e}")
                    logger.error(f"Function object type: {type(func)}, dir(func): {dir(func)}")
                    continue
            
            # Add classes/structs
            if not hasattr(module, 'classes'):
                logger.warning(f"Module {module_name} has no classes attribute")
                continue
                
            logger.debug(f"Processing classes for module {module_name}: count={len(module.classes)}")
            for cls in module.classes:
                try:
                    class_name = cls.name
                    class_id = f"class:{module_name}:{class_name}"
                    node_id = class_id  # Use the same ID format for the node_id field
                    
                    # Handle case where class attributes might be None
                    cls_file_path = getattr(cls, 'file', None) or module_file_path
                    cls_source = getattr(cls, 'context', None) or f"Class {class_name}"
                    cls_line_number = getattr(cls, 'line', 0) or 0  # Default to 0 if not available
                    cls_docstring = getattr(cls, 'docstring', None) or ""
                    
                    semantic_graph.add_node(
                        class_id,
                        type='class',
                        name=class_name,
                        node_id=node_id,
                        file_path=cls_file_path,
                        line_number=cls_line_number,
                        code_context=cls_source,
                        doc_context=cls_docstring
                    )
                    # Add edge from module to class
                    semantic_graph.add_edge(module_id, class_id, type='contains')
                    
                    # Add methods
                    if not hasattr(cls, 'methods'):
                        logger.warning(f"Class {class_name} in module {module_name} has no methods attribute")
                        continue
                    
                    if not isinstance(cls.methods, list):
                        logger.warning(f"Methods attribute of class {class_name} is not a list: {type(cls.methods)}")
                        continue
                        
                    logger.debug(f"Processing methods for class {class_name}: count={len(cls.methods)}")
                    for method in cls.methods:
                        try:
                            method_name = method.name
                            method_id = f"method:{module_name}:{class_name}:{method_name}"
                            node_id = method_id  # Use the same ID format for the node_id field
                            
                            # Store method name to ID mapping
                            method_name_to_id[method_name] = method_id
                            
                            # Handle case where method attributes might be None
                            method_file_path = getattr(method, 'file', None) or cls_file_path
                            method_source = getattr(method, 'context', None) or f"Method {method_name}"
                            method_line_number = getattr(method, 'line', 0) or 0  # Default to 0 if not available
                            method_docstring = getattr(method, 'docstring', None) or ""
                            
                            # Store method calls information for later processing
                            method_calls = getattr(method, 'calls', [])
                            
                            semantic_graph.add_node(
                                method_id,
                                type='method',
                                name=method_name,
                                node_id=node_id,
                                file_path=method_file_path,
                                line_number=method_line_number,
                                code_context=method_source,
                                doc_context=method_docstring,
                                calls=method_calls  # Store the calls in node attributes
                            )
                            # Add edge from class to method
                            semantic_graph.add_edge(class_id, method_id, type='contains')
                        except AttributeError as e:
                            logger.error(f"AttributeError processing method in class {class_name}: {e}")
                            logger.error(f"Method object type: {type(method)}, dir(method): {dir(method)}")
                            continue
                except AttributeError as e:
                    logger.error(f"AttributeError processing class in module {module_name}: {e}")
                    logger.error(f"Class object type: {type(cls)}, dir(cls): {dir(cls)}")
                    continue
        
        # Add call relationships
        logger.info("Adding call relationships...")
        for node_id, node_data in semantic_graph.nodes(data=True):
            if 'calls' in node_data and node_data['calls']:
                caller_id = node_id
                for called_func in node_data['calls']:
                    # Try different formats of the called function name
                    callee_id = None
                    
                    # Try full path first
                    if called_func in function_name_to_id:
                        callee_id = function_name_to_id[called_func]
                    elif called_func in method_name_to_id:
                        callee_id = method_name_to_id[called_func]
                    else:
                        # Try to extract the simple name (last part after the last dot)
                        simple_name = called_func.split('.')[-1] if '.' in called_func else called_func
                        if simple_name in function_name_to_id:
                            callee_id = function_name_to_id[simple_name]
                        elif simple_name in method_name_to_id:
                            callee_id = method_name_to_id[simple_name]
                    
                    if callee_id:
                        semantic_graph.add_edge(caller_id, callee_id, type='calls')
                    else:
                        logger.debug(f"Could not find target for call: {called_func}")
        
        # Generate and add embeddings
        logger.info("Generating embeddings for nodes...")
        for node_id, node_data in semantic_graph.nodes(data=True):
            try:
                # Get content for embedding
                code_content = node_data.get('code_context', '')
                doc_content = node_data.get('doc_context', '')
                
                # Generate code embedding
                if code_content:
                    code_embedding = self.code_embedder.embed(code_content)
                    semantic_graph.nodes[node_id]['code_embedding'] = code_embedding
                else:
                    # Generate empty embedding with right dimensions if no code content
                    semantic_graph.nodes[node_id]['code_embedding'] = [0.0] * self.code_embedder.embedding_dim
                
                # Generate doc embedding
                if doc_content:
                    doc_embedding = self.doc_embedder.embed(doc_content)
                    semantic_graph.nodes[node_id]['doc_embedding'] = doc_embedding
                else:
                    # Generate empty embedding with right dimensions if no doc content
                    semantic_graph.nodes[node_id]['doc_embedding'] = [0.0] * self.doc_embedder.embedding_dim
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for node {node_id}: {e}")
                # Ensure there are default embeddings even when errors occur
                semantic_graph.nodes[node_id]['code_embedding'] = [0.0] * self.code_embedder.embedding_dim
                semantic_graph.nodes[node_id]['doc_embedding'] = [0.0] * self.doc_embedder.embedding_dim
                continue
        
        return semantic_graph 