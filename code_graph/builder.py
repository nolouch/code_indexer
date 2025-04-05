from typing import Dict, List, Optional, Any
import networkx as nx
import logging
from pathlib import Path
from tqdm import tqdm
from core.models import CodeRepository, Module, CodeElement
from .embedders import create_embedder
from .config import get_config
from .db_manager import GraphDBManager

logger = logging.getLogger(__name__)

class SemanticGraphBuilder:
    """Builds semantic graphs from code analysis results"""
    
    def __init__(self, code_embedder=None, doc_embedder=None, batch_size=None, db_manager=None):
        """Initialize builder with embedders.
        
        Args:
            code_embedder: Code embedder, will create default if None
            doc_embedder: Documentation embedder, will create default if None
            batch_size: Size of batches for embedding operations, uses config default if None
            db_manager: Database manager, will create default if None
        """
        # Use the factory to create appropriate embedders based on configuration
        self.code_embedder = code_embedder or create_embedder("code")
        self.doc_embedder = doc_embedder or create_embedder("doc")
        config = get_config()
        self.batch_size = batch_size or config.get("batch_size", 32)
        self.db_manager = db_manager or GraphDBManager()
        # Maps to track nodes across multiple module graphs
        self.function_name_to_id = {}
        self.method_name_to_id = {}
        # Full repository graph
        self.full_graph = nx.DiGraph()
        
    def build_from_repository(self, code_repo: CodeRepository, repo_path: str, save_per_module: bool = False) -> nx.DiGraph:
        """Build semantic graph from a code repository.
        
        Args:
            code_repo: Code repository to analyze
            repo_path: Path to the repository
            save_per_module: Whether to save each module to DB after processing
            
        Returns:
            nx.DiGraph: The semantic graph with nodes and relationships
        """
        self.full_graph = nx.DiGraph()
        
        if not code_repo or not hasattr(code_repo, 'modules'):
            logger.warning("Code repository does not have modules attribute")
            return self.full_graph
        
        logger.debug(f"Converting code repository to semantic graph: type={type(code_repo)}, modules={len(code_repo.modules) if hasattr(code_repo, 'modules') else 'N/A'}")
        
        # Reset maps
        self.function_name_to_id = {}
        self.method_name_to_id = {}
        
        # List to track all processed module graphs for later relationship processing
        module_graphs = []
        
        # Process each module separately
        for module_name, module in code_repo.modules.items():
            # Skip if module is None
            if module is None:
                logger.warning(f"Module {module_name} is None, skipping")
                continue

            logger.info(f"Processing module: name={module_name}, type={type(module)}")
            
            # Create a module subgraph
            module_graph = self._process_module(module_name, module)
            
            # Apply embeddings to the module graph
            self._apply_embeddings_to_graph(module_graph)
            
            # Save to database if required
            if save_per_module and self.db_manager and self.db_manager.available:
                logger.info(f"Saving module {module_name} to database")
                self.db_manager.save_graph(module_graph, repo_path)
            
            # Keep track of module graph
            module_graphs.append(module_graph)
            
            # Merge the module graph into the full graph
            self.full_graph = nx.compose(self.full_graph, module_graph)
            
        # Process cross-module call relationships
        self._process_call_relationships(self.full_graph)
        
        # Save just the cross-module edges if we're saving per module
        if save_per_module and self.db_manager and self.db_manager.available:
            # Create a graph that contains only the cross-module edges
            cross_module_graph = nx.DiGraph()
            
            # Add all nodes to the cross-module graph (needed for edge references)
            for node, data in self.full_graph.nodes(data=True):
                cross_module_graph.add_node(node, **data)
            
            # Find edges that span across modules
            for source, target, data in self.full_graph.edges(data=True):
                edge_type = data.get('type', '')
                # Find the modules these nodes belong to
                source_module = source.split(':')[1] if ':' in source else None
                target_module = target.split(':')[1] if ':' in target else None
                
                # If it's a call edge and the modules are different, it's a cross-module edge
                if edge_type == 'calls' and source_module != target_module:
                    cross_module_graph.add_edge(source, target, **data)
            
            if len(cross_module_graph.edges()) > 0:
                logger.info(f"Saving {len(cross_module_graph.edges())} cross-module relationships to database")
                self.db_manager.save_graph_edges(cross_module_graph, repo_path)
        
        # If not saving per module, save the full graph at once
        if not save_per_module:
            # Apply embeddings to the full graph (if they weren't added per module)
            if not any('code_embedding' in data for _, data in self.full_graph.nodes(data=True)):
                self._apply_embeddings_to_graph(self.full_graph)
            
            # Save the entire graph at once
            if self.db_manager and self.db_manager.available:
                logger.info(f"Saving complete graph to database with {len(self.full_graph.nodes())} nodes")
                self.db_manager.save_graph(self.full_graph, repo_path)
        
        return self.full_graph
    
    def _process_module(self, module_name: str, module: Module) -> nx.DiGraph:
        """Process a single module and return its graph.
        
        Args:
            module_name: Name of the module
            module: Module object
            
        Returns:
            nx.DiGraph: The module's semantic graph
        """
        module_graph = nx.DiGraph()
        
        module_id = f"module:{module_name}"
        node_id = module_id  # Use the same ID format for the node_id field
        
        # Handle case where module attributes are None
        module_file_path = getattr(module, 'file_path', None) or str(module_name)
        
        module_graph.add_node(
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
        if hasattr(module, 'functions'):
            logger.info(f"Processing functions for module {module_name}: count={len(module.functions)}")
            for func in module.functions:
                try:
                    func_name = func.name
                    func_id = f"func:{module_name}:{func_name}"
                    node_id = func_id  # Use the same ID format for the node_id field
                    
                    # Store function name to ID mapping for call relationships
                    self.function_name_to_id[func_name] = func_id
                    
                    # Handle case where function attributes might be None
                    func_file_path = getattr(func, 'file', None) or module_file_path
                    func_source = getattr(func, 'context', None) or f"Function {func_name}"
                    func_line_number = getattr(func, 'line', 0) or 0  # Default to 0 if not available
                    func_docstring = getattr(func, 'docstring', None) or ""
                    
                    # Store function calls information for later processing
                    func_calls = getattr(func, 'calls', [])
                    
                    module_graph.add_node(
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
                    module_graph.add_edge(module_id, func_id, type='contains')
                except AttributeError as e:
                    logger.error(f"AttributeError processing function in module {module_name}: {e}")
                    logger.error(f"Function object type: {type(func)}, dir(func): {dir(func)}")
                    continue
        else:
            logger.warning(f"Module {module_name} has no functions attribute")
        
        # Add classes/structs
        if hasattr(module, 'classes'):
            logger.info(f"Processing classes for module {module_name}: count={len(module.classes)}")
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
                    
                    module_graph.add_node(
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
                    module_graph.add_edge(module_id, class_id, type='contains')
                    
                    # Add methods
                    if hasattr(cls, 'methods') and isinstance(cls.methods, list):
                        logger.info(f"Processing methods for class {class_name}: count={len(cls.methods)}")
                        for method in cls.methods:
                            try:
                                method_name = method.name
                                method_id = f"method:{module_name}:{class_name}:{method_name}"
                                node_id = method_id  # Use the same ID format for the node_id field
                                
                                # Store method name to ID mapping
                                self.method_name_to_id[method_name] = method_id
                                
                                # Handle case where method attributes might be None
                                method_file_path = getattr(method, 'file', None) or cls_file_path
                                method_source = getattr(method, 'context', None) or f"Method {method_name}"
                                method_line_number = getattr(method, 'line', 0) or 0  # Default to 0 if not available
                                method_docstring = getattr(method, 'docstring', None) or ""
                                
                                # Store method calls information for later processing
                                method_calls = getattr(method, 'calls', [])
                                
                                module_graph.add_node(
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
                                module_graph.add_edge(class_id, method_id, type='contains')
                            except AttributeError as e:
                                logger.error(f"AttributeError processing method in class {class_name}: {e}")
                                logger.error(f"Method object type: {type(method)}, dir(method): {dir(method)}")
                                continue
                    else:
                        logger.warning(f"Class {class_name} in module {module_name} has no methods attribute or it's not a list")
                except AttributeError as e:
                    logger.error(f"AttributeError processing class in module {module_name}: {e}")
                    logger.error(f"Class object type: {type(cls)}, dir(cls): {dir(cls)}")
                    continue
        else:
            logger.warning(f"Module {module_name} has no classes attribute")
        
        return module_graph
    
    def _process_call_relationships(self, graph: nx.DiGraph):
        """Process call relationships in the graph.
        
        Args:
            graph: The graph to process
        """
        logger.info("Adding call relationships...")
        for node_id, node_data in graph.nodes(data=True):
            if 'calls' in node_data and node_data['calls']:
                caller_id = node_id
                for called_func in node_data['calls']:
                    # Try different formats of the called function name
                    callee_id = None
                    
                    # Try full path first
                    if called_func in self.function_name_to_id:
                        callee_id = self.function_name_to_id[called_func]
                    elif called_func in self.method_name_to_id:
                        callee_id = self.method_name_to_id[called_func]
                    else:
                        # Try to extract the simple name (last part after the last dot)
                        simple_name = called_func.split('.')[-1] if '.' in called_func else called_func
                        if simple_name in self.function_name_to_id:
                            callee_id = self.function_name_to_id[simple_name]
                        elif simple_name in self.method_name_to_id:
                            callee_id = self.method_name_to_id[simple_name]
                    
                    if callee_id and callee_id in graph:
                        graph.add_edge(caller_id, callee_id, type='calls')
                    else:
                        logger.debug(f"Could not find target for call: {called_func}")
    
    def _apply_embeddings_to_graph(self, graph: nx.DiGraph):
        """Apply embeddings to all nodes in the graph.
        
        Args:
            graph: The graph to apply embeddings to
        """
        logger.info("Generating embeddings for nodes...")
        
        # Batch processing for embeddings
        node_ids = list(graph.nodes())
        total_nodes = len(node_ids)
        logger.info(f"Processing embeddings for {total_nodes} nodes with batch size {self.batch_size}")
        
        # Calculate total number of batches
        total_batches = (total_nodes + self.batch_size - 1) // self.batch_size
        
        # Process in batches with progress bar
        for batch_start in tqdm(range(0, total_nodes, self.batch_size), total=total_batches, 
                               desc=f"Batches (size={self.batch_size})", unit="batch"):
            batch_end = min(batch_start + self.batch_size, total_nodes)
            batch_node_ids = node_ids[batch_start:batch_end]
            
            # Collect content for the current batch
            batch_code_contents = []
            batch_doc_contents = []
            
            for node_id in batch_node_ids:
                node_data = graph.nodes[node_id]
                code_content = node_data.get('code_context', '')
                doc_content = node_data.get('doc_context', '')
                
                batch_code_contents.append(code_content)
                batch_doc_contents.append(doc_content)
            
            # Generate embeddings for code content
            code_embeddings = self.code_embedder.batch_embed(batch_code_contents)
            
            # Generate embeddings for doc content
            doc_embeddings = self.doc_embedder.batch_embed(batch_doc_contents)
            
            # Add embeddings back to nodes
            for i, node_id in enumerate(batch_node_ids):
                graph.nodes[node_id]['code_embedding'] = code_embeddings[i]
                graph.nodes[node_id]['doc_embedding'] = doc_embeddings[i]
                
        logger.info(f"Generated embeddings for all {total_nodes} nodes")
        return graph 