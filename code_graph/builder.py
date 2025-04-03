from typing import Dict, List, Optional, Any
import networkx as nx
from pathlib import Path
from core.models import CodeRepository, Module, CodeElement
from .embedders.code import CodeEmbedder
from .embedders.doc import DocEmbedder
from .relations import SemanticRelation
import os
import logging

logger = logging.getLogger(__name__)

class SemanticGraphBuilder:
    """Builds and persists semantic graphs from code analysis results"""
    
    def __init__(self, graph: nx.MultiDiGraph, code_embedder: CodeEmbedder, doc_embedder: DocEmbedder):
        """Initialize builder with graph and embedders.
        
        Args:
            graph: Graph to build
            code_embedder: Code embedder
            doc_embedder: Documentation embedder
        """
        self.graph = graph
        self.code_embedder = code_embedder
        self.doc_embedder = doc_embedder
        
    def build_from_repository(self, repo: CodeRepository) -> nx.MultiDiGraph:
        """Build semantic graph from a code repository.
        
        Args:
            repo: Code repository to analyze
            
        Returns:
            NetworkX graph with semantic information
        """
        # Check if repo is None
        if repo is None:
            return self.graph
            
        # Add modules and their elements
        for module_name, module in repo.modules.items():
            try:
                if module is not None:
                    self._add_module(module)
            except Exception as e:
                logger.error(f"Error adding module {module_name}: {e}")
                
        # Add semantic relations
        try:
            self._add_semantic_relations()
        except Exception as e:
            logger.error(f"Error adding semantic relations: {e}")
        
        # Add embeddings
        try:
            self._add_embeddings()
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
        
        return self.graph
        
    def _add_module(self, module: Module):
        """Add a module and its elements to the graph.
        
        Args:
            module: Module to add to graph
        """
        # Check if module is None
        if module is None:
            return
            
        # Prioritize using the context field
        file_path = getattr(module, 'file', None)
        module_context = getattr(module, 'context', '')
        
        # If no context is available, read from file
        if not module_context and file_path:
            module_context = self._get_module_content(file_path)
        
        # Add module node
        self.graph.add_node(
            module.name,
            type="module",
            language=module.language,
            file=file_path,
            code_content=module_context
        )
        
        # Add functions
        for func in module.functions:
            func_file = getattr(func, 'file', file_path)
            func_line = getattr(func, 'line', 1)
            
            # Prioritize using the context field
            func_context = getattr(func, 'context', '')
            
            # If no context is available, try to read from file
            if not func_context and func_file:
                func_context = self._get_code_content(func_file, func_line)
                
            self.graph.add_node(
                f"{module.name}.{func.name}",
                type="function",
                language=func.language,
                file=func_file,
                line=func_line,
                docstring=getattr(func, 'docstring', ''),
                code_content=func_context
            )
            self.graph.add_edge(
                module.name,
                f"{module.name}.{func.name}",
                type="contains"
            )
            
        # Add classes
        for cls in module.classes:
            cls_file = getattr(cls, 'file', file_path)
            cls_line = getattr(cls, 'line', 1)
            
            # Prioritize using the context field
            cls_context = getattr(cls, 'context', '')
            
            # If no context is available, try to read from file
            if not cls_context and cls_file:
                cls_context = self._get_code_content(cls_file, cls_line)
                
            self.graph.add_node(
                f"{module.name}.{cls.name}",
                type="class",
                language=cls.language,
                file=cls_file,
                line=cls_line,
                docstring=getattr(cls, 'docstring', ''),
                code_content=cls_context
            )
            self.graph.add_edge(
                module.name,
                f"{module.name}.{cls.name}",
                type="contains"
            )
            
        # Add interfaces
        for iface in module.interfaces:
            iface_file = getattr(iface, 'file', file_path)
            iface_line = getattr(iface, 'line', 1)
            
            # Prioritize using the context field
            iface_context = getattr(iface, 'context', '')
            
            # If no context is available, try to read from file
            if not iface_context and iface_file:
                iface_context = self._get_code_content(iface_file, iface_line)
                
            self.graph.add_node(
                f"{module.name}.{iface.name}",
                type="interface",
                language=iface.language,
                file=iface_file,
                line=iface_line,
                docstring=getattr(iface, 'docstring', ''),
                code_content=iface_context
            )
            self.graph.add_edge(
                module.name,
                f"{module.name}.{iface.name}",
                type="contains"
            )
            
    def _add_semantic_relations(self):
        """Add semantic relationships between nodes."""
        # In mock testing, graph.nodes(data=True) might return a dict instead of iterable
        nodes_data = self.graph.nodes(data=True)
        # If it's a dict, convert it to iterable of (node, data) pairs
        if isinstance(nodes_data, dict):
            nodes_data = nodes_data.items()
            
        for node, data in nodes_data:
            if data.get("type") == "function":
                # Add call relations
                for other_node in self.graph.nodes():
                    if other_node != node and self._has_call_relation(node, other_node):
                        self.graph.add_edge(
                            node, other_node,
                            type="calls",
                            weight=self._get_call_weight(node, other_node)
                        )
                        
            elif data.get("type") == "class":
                # Add inheritance relations
                for other_node in self.graph.nodes():
                    if other_node != node and self._has_inheritance_relation(node, other_node):
                        self.graph.add_edge(
                            node, other_node,
                            type="inherits",
                            weight=self._get_inheritance_weight(node, other_node)
                        )
                        
    def _add_embeddings(self):
        """Add code and documentation embeddings to nodes."""
        # In mock testing, graph.nodes(data=True) might return a dict instead of iterable
        nodes_data = self.graph.nodes(data=True)
        # If it's a dict, convert it to iterable of (node, data) pairs
        if isinstance(nodes_data, dict):
            nodes_data = nodes_data.items()
            
        # Collect all code contents for batch embedding
        nodes_with_code = []
        code_contents = []
        
        # Collect all docstrings for batch embedding  
        nodes_with_doc = []
        docstrings = []
        
        for node, data in nodes_data:
            # Prefer context field if available, otherwise use code_content
            if "code_content" in data:
                if not data["code_content"] and data.get("type") != "module":
                    # Try to get context from the node's file
                    try:
                        element_type = data.get("type", "")
                        # Since we now have context fields in our objects, prioritize it
                        # For legacy data, we might still need to read from file
                        nodes_with_code.append(node)
                        code_contents.append(data["code_content"])
                    except Exception as e:
                        logger.warning(f"Error getting code content for {node}: {e}")
                else:
                    nodes_with_code.append(node)
                    code_contents.append(data["code_content"])
                    
            # Get documentation embedding
            if "docstring" in data and data["docstring"]:
                nodes_with_doc.append(node)
                docstrings.append(data["docstring"])
        
        # Batch embed code
        if nodes_with_code:
            try:
                code_embeddings = self.code_embedder.batch_embed(code_contents)
                
                for i, node in enumerate(nodes_with_code):
                    if i < len(code_embeddings):
                        self.graph.nodes[node]["code_embedding"] = code_embeddings[i]
            except Exception as e:
                logger.error(f"Error batch embedding code: {e}")
                # Fall back to individual embedding
                for i, node in enumerate(nodes_with_code):
                    if i < len(code_contents):
                        try:
                            self.graph.nodes[node]["code_embedding"] = self.code_embedder.embed(code_contents[i])
                        except Exception as e2:
                            logger.error(f"Error embedding code for {node}: {e2}")
                
        # Batch embed docstrings
        if nodes_with_doc:
            try:
                doc_embeddings = self.doc_embedder.batch_embed(docstrings)
                
                for i, node in enumerate(nodes_with_doc):
                    if i < len(doc_embeddings):
                        self.graph.nodes[node]["doc_embedding"] = doc_embeddings[i]
            except Exception as e:
                logger.error(f"Error batch embedding docs: {e}")
                # Fall back to individual embedding
                for i, node in enumerate(nodes_with_doc):
                    if i < len(docstrings):
                        try:
                            self.graph.nodes[node]["doc_embedding"] = self.doc_embedder.embed(docstrings[i])
                        except Exception as e2:
                            logger.error(f"Error embedding doc for {node}: {e2}")
                
    def _get_module_content(self, file_path: str) -> Optional[str]:
        """Get the content of a module from its file."""
        try:
            # Try to open the file using the absolute path
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Error reading module content from {file_path}: {e}")
            return None
                
    def _get_code_content(self, file_path: str, line: int, context: int = 5) -> Optional[str]:
        """Get code content with context lines."""
        try:
            # Try to open the file using the absolute path
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                # Adjust line number to 0-index and bounds
                line_index = max(0, line - 1)
                start = max(0, line_index - context)
                end = min(len(lines), line_index + context + 1)
                
                return ''.join(lines[start:end])
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Error reading code content from {file_path}: {e}")
            return None
            
    def _has_call_relation(self, func1: str, func2: str) -> bool:
        """Check if there is a call relation between functions."""
        # Implementation depends on language-specific analysis
        return False
        
    def _has_inheritance_relation(self, cls1: str, cls2: str) -> bool:
        """Check if there is an inheritance relation between classes."""
        # Implementation depends on language-specific analysis
        return False
        
    def _get_call_weight(self, func1: str, func2: str) -> float:
        """Get the weight of a call relation."""
        # Implementation depends on language-specific analysis
        return 1.0
        
    def _get_inheritance_weight(self, cls1: str, cls2: str) -> float:
        """Get the weight of an inheritance relation."""
        # Implementation depends on language-specific analysis
        return 1.0 