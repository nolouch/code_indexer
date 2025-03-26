from typing import Dict, List, Optional, Any
import networkx as nx
from pathlib import Path
from core.models import CodeRepository, Module, CodeElement
from .embedders.code import CodeEmbedder
from .embedders.doc import DocEmbedder
from .relations import SemanticRelation
from .db_manager import GraphDBManager

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
        # Add modules and their elements
        for module in repo.modules.values():
            self._add_module(module)
            
        # Add semantic relations
        self._add_semantic_relations()
        
        # Add embeddings
        self._add_embeddings()
        
        return self.graph
        
    def _add_module(self, module: Module):
        """Add a module and its elements to the graph.
        
        Args:
            module: Module to add to graph
        """
        # Get module content once
        module_content = self._get_module_content(module.file)
        
        # Add module node
        self.graph.add_node(
            module.name,
            type="module",
            language=module.language,
            file=module.file,
            code_content=module_content
        )
        
        # Add functions
        for func in module.functions:
            func_code = self._get_code_content(func.file, func.line) or module_content
            self.graph.add_node(
                f"{module.name}.{func.name}",
                type="function",
                language=func.language,
                file=func.file,
                line=func.line,
                docstring=func.docstring,
                code_content=func_code
            )
            self.graph.add_edge(
                module.name,
                f"{module.name}.{func.name}",
                type="contains"
            )
            
        # Add classes
        for cls in module.classes:
            cls_code = self._get_code_content(cls.file, cls.line) or module_content
            self.graph.add_node(
                f"{module.name}.{cls.name}",
                type="class",
                language=cls.language,
                file=cls.file,
                line=cls.line,
                docstring=cls.docstring,
                code_content=cls_code
            )
            self.graph.add_edge(
                module.name,
                f"{module.name}.{cls.name}",
                type="contains"
            )
            
        # Add interfaces
        for iface in module.interfaces:
            iface_code = self._get_code_content(iface.file, iface.line) or module_content
            self.graph.add_node(
                f"{module.name}.{iface.name}",
                type="interface",
                language=iface.language,
                file=iface.file,
                line=iface.line,
                docstring=iface.docstring,
                code_content=iface_code
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
            
        for node, data in nodes_data:
            # Get code embedding
            if "code_content" in data:
                data["code_embedding"] = self.code_embedder.embed(data["code_content"])
                    
            # Get documentation embedding
            if "docstring" in data and data["docstring"]:
                data["doc_embedding"] = self.doc_embedder.embed(data["docstring"])
                
    def _get_module_content(self, file_path: str) -> Optional[str]:
        """Get entire module content.
        
        Args:
            file_path: Path to module file
            
        Returns:
            Module content or None if file not found
        """
        try:
            with open(file_path) as f:
                return f.read()
        except Exception:
            return None
                
    def _get_code_content(self, file_path: str, line: int, context: int = 5) -> Optional[str]:
        """Get code content with context.
        
        Args:
            file_path: Path to source file
            line: Target line number
            context: Number of context lines before and after
            
        Returns:
            Code content with context or None if file not found
        """
        try:
            with open(file_path) as f:
                lines = f.readlines()
                start = max(0, line - context - 1)
                end = min(len(lines), line + context)
                return ''.join(lines[start:end])
        except Exception:
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
        """Get weight of call relation."""
        # Could be based on call frequency, proximity, etc.
        return 1.0
        
    def _get_inheritance_weight(self, cls1: str, cls2: str) -> float:
        """Get weight of inheritance relation."""
        # Could be based on number of inherited methods, etc.
        return 1.0