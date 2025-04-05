#!/usr/bin/env python3
import logging
import os
import json
import sys
import argparse
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer

from setting.base import DATABASE_URI
from setting.embedding import EMBEDDING_MODEL
from code_graph.db_manager import GraphDBManager
from parsers.go.parser import GoParser
from code_graph.builder import SemanticGraphBuilder
from code_graph.embedders.code import CodeEmbedder
from code_graph.embedders.doc import DocEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Repository:
    """Repository class, which describes the repository."""

    def __init__(self, path, name=None, semantic_graph=None, db_manager=None):
        """Initialize Repository.

        Args:
            path (str): Path to repository.
            name (str, optional): Name of the repository.
            semantic_graph (nx.DiGraph, optional): Semantic graph to use.
            db_manager (GraphDBManager, optional): Database manager to use.
        """
        self.path = Path(path)
        self.name = name or self.path.name
        self.semantic_graph = semantic_graph or nx.DiGraph()
        self.db_manager = db_manager
        
    def vector_search(self, query: str, limit: int = 10, use_doc_embedding: bool = False) -> 'VectorSearchResults':
        """Search for nodes in the semantic graph using vector search.

        Args:
            query (str): Query to search for.
            limit (int, optional): Maximum number of results to return.
            use_doc_embedding (bool, optional): Whether to use document embedding.

        Returns:
            VectorSearchResults: Search results.
        """
        if self.db_manager is None:
            logger.warning("Vector search requires a database connection, but no db_manager is provided.")
            return VectorSearchResults(self, query, [])
            
        results = self.db_manager.vector_search(
            repo_path=str(self.path),
            query=query,
            limit=limit,
            use_doc_embedding=use_doc_embedding
        )
        return VectorSearchResults(self, query, results)
        
    def search(self, query, max_results=10):
        """Search for nodes in the semantic graph.

        Args:
            query (str): Query to search for.
            max_results (int, optional): Maximum number of results to return.

        Returns:
            SearchResults: Search results.
        """
        # First try vector search if db_manager is available
        if self.db_manager is not None:
            try:
                return self.vector_search(query, max_results)
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to local search: {e}")
        
        # Fallback to local search
        matches = []
        
        # First try exact text matching
        exact_matches = []
        for node_id, node_data in self.semantic_graph.nodes(data=True):
            for field in ['content', 'name', 'docstring', 'signature']:
                if field in node_data and query.lower() in str(node_data[field]).lower():
                    exact_matches.append((node_id, node_data))
                    break
        
        # If we have enough exact matches, use them
        if len(exact_matches) >= max_results:
            return SearchResults(self, query, exact_matches[:max_results])
        
        # Otherwise, try semantic matching if we have embeddings
        try:
            from llm.embedding import get_sentence_transformer
            embedding_model = get_sentence_transformer(EMBEDDING_MODEL["name"])
            query_embedding = embedding_model.encode(query)
            
            # Calculate semantic similarity for nodes with embeddings
            semantic_matches = []
            for node_id, node_data in self.semantic_graph.nodes(data=True):
                embedding = node_data.get('embedding')
                if embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                    # Add similarity score to node data
                    node_data_with_sim = node_data.copy()
                    node_data_with_sim['similarity'] = float(similarity)
                    semantic_matches.append((node_id, node_data_with_sim))
            
            # Sort by similarity
            semantic_matches.sort(key=lambda x: x[1].get('similarity', 0), reverse=True)
            
            # Combine exact and semantic matches
            combined_matches = exact_matches + [match for match in semantic_matches if match[0] not in {m[0] for m in exact_matches}]
            
            matches = combined_matches[:max_results]
        except Exception as e:
            logger.warning(f"Semantic matching failed, using only exact matches: {e}")
            matches = exact_matches
        
        return SearchResults(self, query, matches[:max_results])
        
    def get_node(self, node_id):
        """Get node from semantic graph.

        Args:
            node_id (str): ID of the node to get.

        Returns:
            tuple: Node ID and node data.
        """
        if node_id in self.semantic_graph:
            return node_id, self.semantic_graph.nodes[node_id]
        return None

class SearchResults:
    """Search results class, provides result processing and display functions"""
    
    def __init__(self, repository, query, results=None):
        self.repository = repository
        self.query = query
        self.results = results or []
        
        # If results were not provided, perform a simple search
        if results is None:
            # Simple keyword search
            keywords = query.lower().split()
            
            # Search in node attributes
            for node, attrs in repository.semantic_graph.nodes(data=True):
                node_str = str(node).lower()
                attrs_str = str(attrs).lower()
                
                if any(keyword in node_str or keyword in attrs_str for keyword in keywords):
                    self.results.append((node, attrs))
                
    def expand(self):
        """Expand search results, including related nodes"""
        expanded_results = self.results.copy()
        
        # For each result, add its neighbors
        for node, _ in self.results:
            if node in self.repository.semantic_graph:
                # Add neighbors
                for neighbor in self.repository.semantic_graph.neighbors(node):
                    neighbor_attrs = self.repository.semantic_graph.nodes[neighbor]
                    expanded_results.append((neighbor, neighbor_attrs))
                
                # Add predecessors
                for pred in self.repository.semantic_graph.predecessors(node):
                    pred_attrs = self.repository.semantic_graph.nodes[pred]
                    expanded_results.append((pred, pred_attrs))
                    
        # Remove duplicates, maintain order
        unique_results = []
        seen = set()
        
        for node, attrs in expanded_results:
            if node not in seen:
                seen.add(node)
                unique_results.append((node, attrs))
                
        expanded = SearchResults(self.repository, self.query)
        expanded.results = unique_results
        return expanded
        
    def explain(self):
        """Explain search results"""
        explanation = f"Search query: {self.query}\n"
        explanation += f"Found {len(self.results)} results\n\n"
        
        # Group results by type
        results_by_type = {}
        for node, attrs in self.results:
            node_type = attrs.get('type', 'unknown')
            if node_type not in results_by_type:
                results_by_type[node_type] = []
            results_by_type[node_type].append((node, attrs))
            
        # Generate explanation for each type
        for node_type, nodes in results_by_type.items():
            explanation += f"{node_type.capitalize()} matches ({len(nodes)}):\n"
            
            for node, attrs in nodes[:5]:  # Limit to 5 examples per type
                if node_type == 'module':
                    explanation += f"  - Module: {node}\n"
                elif node_type == 'import':
                    explanation += f"  - Import: {node}\n"
                elif node_type == 'struct' or node_type == 'class':
                    explanation += f"  - {node_type.capitalize()}: {node} (in {attrs.get('file', 'unknown')})\n"
                elif node_type == 'interface':
                    explanation += f"  - Interface: {node} with {len(attrs.get('methods', []))} methods\n"
                elif node_type == 'function':
                    explanation += f"  - Function: {node} ({', '.join(p.get('type', '') for p in attrs.get('parameters', []))})"
                    if attrs.get('return_types'):
                        explanation += f" -> {', '.join(attrs.get('return_types', []))}\n"
                    else:
                        explanation += "\n"
                elif node_type == 'method':
                    explanation += f"  - Method: {node}\n"
                else:
                    explanation += f"  - {node}\n"
                    
            if len(nodes) > 5:
                explanation += f"  - ... and {len(nodes) - 5} more\n"
                
            explanation += "\n"
            
        return explanation

class VectorSearchResults:
    """Class for handling vector search results"""
    
    def __init__(self, repository, query, results):
        self.repository = repository
        self.query = query
        self.results = results or []
    
    def explain(self, detailed=False):
        """Explain search results in a human-readable format.
        
        Args:
            detailed (bool): Whether to include detailed information.
            
        Returns:
            str: Human-readable explanation of search results.
        """
        if not self.results:
            return f"No results found for query: {self.query}"
        
        # Ensure results is an iterable that's not a string
        if isinstance(self.results, (str, bytes)):
            return f"Invalid results format for query: {self.query}"
            
        # Try to get length of results
        try:
            result_count = len(self.results)
        except (TypeError, AttributeError):
            # If results doesn't support len(), treat it as a single item
            result_count = 1
            actual_results = [self.results]
        else:
            actual_results = self.results
            
        explanation = [f"Found {result_count} results for query: {self.query}\n"]
        
        for i, result in enumerate(actual_results):
            # Handle both dictionary format from DB and node_id/node_data tuples
            if isinstance(result, tuple) and len(result) == 2:
                node_id, node_data = result
                similarity = node_data.get('similarity', 0)
            else:
                # Assume result is a dictionary from DB query
                node_id = result.get('id', f"result_{i}")
                node_data = result
                similarity = result.get('similarity', 0)
                
            node_type = node_data.get('type', 'unknown')
            name = node_data.get('name', 'unnamed')
            file_path = node_data.get('file_path', 'unknown')
            content = node_data.get('content', '')
            
            explanation.append(f"{i+1}. [{node_type}] {name} ({similarity:.2f})")
            explanation.append(f"   File: {file_path}")
            
            if detailed:
                # Add start and end lines if available
                if 'start_line' in node_data and 'end_line' in node_data:
                    explanation.append(f"   Lines: {node_data['start_line']}-{node_data['end_line']}")
                
                # Add a snippet of content
                if content:
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    explanation.append(f"   Preview: {content_preview}")
                
                explanation.append("")
        
        return "\n".join(explanation)

class SimpleEmbedder:
    """Simple embedder that converts text to a random vector.
    
    In a production environment, this would use a proper embedding model.
    """
    
    def __init__(self, dim=384):
        """Initialize embedder with the specified dimension.
        
        Args:
            dim (int): Dimension of embeddings to generate.
        """
        self.dim = dim
        
        # Try to import sentence-transformers
        try:
            from llm.embedding import get_sentence_transformer
            self.model = get_sentence_transformer(EMBEDDING_MODEL["name"])
            self.use_model = True
            logger.info("Using SentenceTransformer for embeddings")
        except ImportError:
            logger.warning("SentenceTransformer not available, using random embeddings")
            self.use_model = False
        
    def embed(self, text):
        """Generate embedding for text.
        
        Args:
            text (str): Text to embed.
            
        Returns:
            np.array: Embedding vector.
        """
        if not text:
            return np.zeros(self.dim)
            
        if self.use_model:
            try:
                embedding = self.model.encode(text)
                # Ensure correct dimension
                if len(embedding) < self.dim:
                    # Pad with zeros
                    embedding = np.pad(embedding, (0, self.dim - len(embedding)))
                elif len(embedding) > self.dim:
                    # Truncate
                    embedding = embedding[:self.dim]
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding with model: {e}")
                # Fall back to random embedding
                
        # Generate random embedding (for testing only)
        # In a real system, use a proper embedding model
        return np.random.rand(self.dim)

class CodeIndexer:
    """Code indexer, uses existing modules to analyze code and build an index"""
    
    def __init__(self, embedding_dim=384, db_uri=DATABASE_URI, disable_db=False):
        """Initialize the code indexer.

        Args:
            embedding_dim (int): The dimension of the embeddings to use.
            db_uri (str, optional): The URI of the database to connect to.
            disable_db (bool, optional): Whether to disable database functionality.
        """
        self.embedding_dim = embedding_dim
        self.disable_db = disable_db
        self.db_manager = None
        self.repositories = {}
        
        # Initialize parsers
        self.parsers = {
            'golang': GoParser()
        }
        
        # Only initialize the database manager if not disabled and URI is provided
        if not disable_db and db_uri is not None:
            try:
                from code_graph.db_manager import GraphDBManager
                self.db_manager = GraphDBManager(db_url=db_uri)
                logger.info(f"Connected to database with URI: {db_uri}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                self.disable_db = True
            
        # Initialize embedders
        self.code_embedder = CodeEmbedder(embedding_dim)
        self.doc_embedder = DocEmbedder(embedding_dim)
        
        # Initialize graph builder
        self.graph_builder = SemanticGraphBuilder(self.code_embedder, self.doc_embedder, db_manager=self.db_manager)
        
    def index_repository(self, codebase_path, go_package="", exclude_tests=True):
        """Index a repository, analyzing its code and creating a semantic graph.

        Args:
            codebase_path (str): Path to the repository to index.
            go_package (str, optional): Specific Go package name to analyze (e.g. 'main').
            exclude_tests (bool, optional): Whether to exclude test files (files ending with _test.go) from Go analysis.

        Returns:
            Repository: Repository object containing the semantic graph.
        """
        repo_path = Path(codebase_path)
        logger.info(f"Indexing repository at: {repo_path}")
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Detect language and select appropriate parser
        language = self._detect_language(repo_path)
        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}")
        
        parser = self.parsers[language]
        logger.info(f"Using parser for language: {language}")
        
        # Create a new repository object
        repository = Repository(
            path=str(repo_path),
            semantic_graph=nx.DiGraph(),
            db_manager=self.db_manager if not self.disable_db else None
        )
        
        # Parse the codebase
        code_repo = None
        try:
            # Use parse_directory for GoParser
            if language == 'golang':
                logger.info(f"Parsing Go repository at: {repo_path}")
                # Pass language-specific parameters if applicable
                specific_package = go_package if language == 'golang' else ""
                
                logger.info(f"Go parser options - package: {specific_package or 'all'}, exclude tests: {exclude_tests}")
                code_repo = parser.parse_directory(
                    str(repo_path),
                    specific_package=specific_package,
                    exclude_tests=exclude_tests
                )
                logger.debug(f"Parser returned code repository: {type(code_repo)}")
                
                if code_repo is None:
                    logger.error("Parser returned None for code repository")
                    return repository
                    
                if not hasattr(code_repo, 'modules'):
                    logger.error("Parser returned a code repository without 'modules' attribute")
                    logger.debug(f"code_repo attributes: {dir(code_repo)}")
                    return repository
                    
                logger.info(f"Successfully parsed repository with {len(code_repo.modules)} modules")
                
                # Convert to semantic graph using the builder
                logger.info("Converting to semantic graph...")
                semantic_graph = self.graph_builder.build_from_repository(code_repo, str(repo_path), True)
            else:
                # For other parsers that might use a different method
                raise ValueError(f"Parsing for language {language} is not implemented")
        except Exception as e:
            logger.error(f"Error parsing repository: {e}")
            # Get more detailed error info
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return repository
        
        if not semantic_graph or not semantic_graph.nodes():
            logger.warning(f"No semantic graph was created for repository at {repo_path}")
            return repository
        
        logger.info(f"Created semantic graph with {len(semantic_graph.nodes())} nodes and {len(semantic_graph.edges())} edges")
        
        # Store the semantic graph in the repository
        repository.semantic_graph = semantic_graph
        
        # Always save to JSON file
        output_file = os.path.join(str(repo_path), "code_graph.json")
        self.generate_index(semantic_graph, output_file)
        logger.info(f"Saved semantic graph to JSON file: {output_file}")
        
        # Store in database if enabled
        if not self.disable_db and self.db_manager:
            try:
                logger.info("Storing semantic graph in database")
                self.db_manager.save_graph(semantic_graph, str(repo_path))
                logger.info("Successfully stored semantic graph in database")
            except Exception as e:
                logger.error(f"Failed to store semantic graph in database: {e}")
        
        # Add repository to the list of indexed repositories
        self.repositories[str(repo_path)] = repository
        
        return repository

    def _convert_to_semantic_graph(self, code_repo):
        """Convert a code repository to a semantic graph.
        
        Args:
            code_repo: The code repository object returned by the parser.
            
        Returns:
            nx.DiGraph: The semantic graph.
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
            
            # Handle case where module attributes are None
            module_file_path = getattr(module, 'file_path', None) or str(module_name)
            
            semantic_graph.add_node(
                module_id,
                type='module',
                name=module_name,
                file_path=module_file_path,
                content=f"Module {module_name}"
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
                    
                    # Store function name to ID mapping for call relationships
                    function_name_to_id[func_name] = func_id
                    
                    # Handle case where function attributes might be None
                    func_file_path = getattr(func, 'file', None) or module_file_path
                    func_source = getattr(func, 'context', None) or f"Function {func_name}"
                    func_signature = ""  # Not available in the current model
                    func_docstring = getattr(func, 'docstring', None) or ""
                    
                    # Store function calls information for later processing
                    func_calls = getattr(func, 'calls', [])
                    
                    semantic_graph.add_node(
                        func_id,
                        type='function',
                        name=func_name,
                        file_path=func_file_path,
                        content=func_source,
                        signature=func_signature,
                        docstring=func_docstring,
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
                    
                    # Handle case where class attributes might be None
                    cls_file_path = getattr(cls, 'file', None) or module_file_path
                    cls_source = getattr(cls, 'context', None) or f"Class {class_name}"
                    cls_docstring = getattr(cls, 'docstring', None) or ""
                    
                    semantic_graph.add_node(
                        class_id,
                        type='class',
                        name=class_name,
                        file_path=cls_file_path,
                        content=cls_source,
                        docstring=cls_docstring
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
                            if isinstance(method, str):
                                logger.warning(f"Method '{method}' in class {class_name} is a string, expected Function object")
                                # Create a simple method node with the string as name
                                method_id = f"method:{module_name}:{class_name}:{method}"
                                
                                # Store method name to ID mapping
                                method_name_to_id[method] = method_id
                                
                                semantic_graph.add_node(
                                    method_id,
                                    type='method',
                                    name=method,
                                    file_path=cls_file_path,
                                    content=f"Method {method}",
                                    signature="",
                                    docstring=""
                                )
                                # Add edge from class to method
                                semantic_graph.add_edge(class_id, method_id, type='contains')
                                continue
                                
                            method_name = method.name
                            method_id = f"method:{module_name}:{class_name}:{method_name}"
                            
                            # Store method name to ID mapping
                            method_name_to_id[method_name] = method_id
                            
                            # Handle case where method attributes might be None
                            method_file_path = getattr(method, 'file', None) or cls_file_path
                            method_source = getattr(method, 'context', None) or f"Method {method_name}"
                            method_signature = ""  # Not available in the current model
                            method_docstring = getattr(method, 'docstring', None) or ""
                            
                            # Store method calls information for later processing
                            method_calls = getattr(method, 'calls', [])
                            
                            semantic_graph.add_node(
                                method_id,
                                type='method',
                                name=method_name,
                                file_path=method_file_path,
                                content=method_source,
                                signature=method_signature,
                                docstring=method_docstring,
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
                    # Check if the called function is in our mapping
                    if called_func in function_name_to_id:
                        callee_id = function_name_to_id[called_func]
                        logger.debug(f"Adding call relationship: {caller_id} calls {callee_id}")
                        semantic_graph.add_edge(caller_id, callee_id, type='calls')
                    elif called_func in method_name_to_id:
                        callee_id = method_name_to_id[called_func]
                        logger.debug(f"Adding call relationship: {caller_id} calls {callee_id}")
                        semantic_graph.add_edge(caller_id, callee_id, type='calls')
        
        return semantic_graph

    def _generate_embeddings_for_nodes(self, graph, code_embedder, doc_embedder):
        """Generate embeddings for all nodes in the graph"""
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            
            # Generate code embedding
            code_content = ""
            if node_type == 'function' or node_type == 'method':
                # For functions, use signature + code
                signature = attrs.get('signature', '')
                code_content = signature
                
            elif node_type == 'class' or node_type == 'struct':
                # For classes, use class definition
                name = attrs.get('name', '')
                fields = attrs.get('fields', [])
                code_content = f"class {name} {{ {', '.join(fields)} }}"
                
            elif node_type == 'interface':
                # For interfaces, use method signatures
                name = attrs.get('name', '')
                methods = attrs.get('methods', [])
                code_content = f"interface {name} {{ {', '.join(methods)} }}"
                
            else:
                # Default for other node types
                code_content = str(node_id)
                
            # Generate doc embedding
            doc_content = attrs.get('docstring', '')
            if not doc_content and node_type == 'function':
                # If no docstring, use signature as fallback
                doc_content = attrs.get('signature', '')
                
            # Generate and store embeddings
            if code_content:
                attrs['code_embedding'] = code_embedder.embed(code_content)
                
            if doc_content:
                attrs['doc_embedding'] = doc_embedder.embed(doc_content)
    
    def _detect_language(self, path):
        """Detect the main programming language of a repository.

        Args:
            path (Path): Path to the repository.

        Returns:
            str: Detected programming language (e.g., 'golang', 'python').
        """
        # Count files by extension
        file_counts = {}
        
        # Walk through the directory and count file extensions
        for root, _, files in os.walk(path):
            for file in files:
                # Skip hidden files and directories
                if file.startswith('.') or any(part.startswith('.') for part in Path(root).parts):
                    continue
                    
                # Get the file extension
                ext = os.path.splitext(file)[1].lower()
                if ext:
                    file_counts[ext] = file_counts.get(ext, 0) + 1
        
        logger.info(f"File extension counts: {file_counts}")
        
        # Determine language based on file extensions
        if '.go' in file_counts:
            return 'golang'
        elif '.py' in file_counts:
            return 'python'
        elif '.js' in file_counts or '.ts' in file_counts:
            return 'javascript'
        elif '.java' in file_counts:
            return 'java'
        elif '.c' in file_counts or '.cpp' in file_counts or '.h' in file_counts:
            return 'c++'
        else:
            # Default to the most common extension
            if file_counts:
                most_common_ext = max(file_counts.items(), key=lambda x: x[1])[0]
                logger.warning(f"Could not determine language, defaulting to most common extension: {most_common_ext}")
                return most_common_ext.lstrip('.')
            else:
                logger.warning("No files found in repository")
                return 'unknown'
    
    def generate_index(self, semantic_graph, output_file='code_graph.json'):
        """Generate an index file from the semantic graph.

        Args:
            semantic_graph (nx.DiGraph): Semantic graph to generate index from.
            output_file (str, optional): Path to the output file.
        
        Returns:
            str: Path to the generated index file.
        """
        logger.info(f"Generating index file: {output_file}")
        
        if not semantic_graph or not semantic_graph.nodes():
            logger.warning("No semantic graph to index")
            return None
        
        # Convert the graph to a serializable format
        index = {
            'nodes': {},
            'edges': []
        }
        
        # Add nodes to the index
        for node_id, node_data in semantic_graph.nodes(data=True):
            # Create a copy of node data to avoid modifying the original
            node_info = dict(node_data)
            
            # Remove non-serializable attributes like embeddings
            if 'embedding' in node_info:
                del node_info['embedding']
            
            index['nodes'][node_id] = node_info
        
        # Add edges to the index
        for source, target, edge_data in semantic_graph.edges(data=True):
            # Create a copy of edge data to avoid modifying the original
            edge_info = dict(edge_data)
            
            # Add edge to the index
            index['edges'].append({
                'source': source,
                'target': target,
                'data': edge_info
            })
        
        # Write the index to a file
        with open(output_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Generated index file with {len(index['nodes'])} nodes and {len(index['edges'])} edges")
        return output_file
    
    def load_repository(self, repo_path, go_package="", exclude_tests=True):
        """Load a repository from database.
        
        Args:
            repo_path: Path to the repository
            go_package (str, optional): Specific Go package name to analyze (e.g. 'main').
            exclude_tests (bool, optional): Whether to exclude test files (files ending with _test.go) from Go analysis.
            
        Returns:
            Repository object or None if not found
        """
        if self.disable_db or not self.db_manager:
            logger.warning("Database is disabled. Cannot load repository.")
            return None
            
        try:
            logger.info(f"Loading repository from database: {repo_path}")
            
            # Check if repository exists in database
            repo_stats = self.db_manager.get_repository_stats(repo_path)
            if not repo_stats:
                logger.warning(f"Repository not found in database: {repo_path}")
                
                # If repository not found in the database, try indexing it
                logger.info(f"Attempting to index repository: {repo_path}")
                return self.index_repository(repo_path, go_package=go_package, exclude_tests=exclude_tests)
                
            # Load graph from database
            graph = self.db_manager.load_graph(repo_path)
            if not graph:
                logger.warning(f"No graph found for repository: {repo_path}")
                return None
                
            # Create repository object with the correct parameters
            repo = Repository(
                path=repo_path,
                name=repo_stats.get('repository', Path(repo_path).name),
                semantic_graph=graph,
                db_manager=self.db_manager
            )
            self.repositories[repo_path] = repo
            
            logger.info(f"Successfully loaded repository with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
            return repo
            
        except Exception as e:
            logger.error(f"Error loading repository: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def vector_search(self, repo_path, query, limit=10, use_doc_embedding=False, go_package="", exclude_tests=True):
        """Execute vector-based semantic search.
        
        Args:
            repo_path: Path to the repository
            query: Search query
            limit: Maximum number of results to return
            use_doc_embedding: Whether to use documentation embeddings
            go_package: Specific Go package name to analyze (e.g. 'main')
            exclude_tests: Whether to exclude test files (files ending with _test.go) from Go analysis
            
        Returns:
            Search results or None if not found
        """
        if self.disable_db or not self.db_manager:
            logger.warning("Database is disabled. Cannot perform vector search.")
            return None
            
        # Load repository if not already loaded
        repo = None
        if repo_path in self.repositories:
            repo = self.repositories[repo_path]
        else:
            repo = self.load_repository(repo_path, go_package=go_package, exclude_tests=exclude_tests)
            
        if not repo:
            logger.warning(f"Repository not found: {repo_path}")
            return None
            
        # Execute vector search and get the results directly
        search_results = repo.db_manager.vector_search(
            repo_path=str(repo.path),
            query=query,
            limit=limit,
            use_doc_embedding=use_doc_embedding
        )
        return search_results


def main():
    parser = argparse.ArgumentParser(description='Code Indexer')
    parser.add_argument('codebase', help='Path to the codebase to analyze')
    parser.add_argument('--output', '-o', default='index.json', help='Output file for the index')
    parser.add_argument('--search', '-s', help='Search query')
    parser.add_argument('--explain', '-e', action='store_true', help='Explain search results')
    parser.add_argument('--expand', '-x', action='store_true', help='Expand search results')
    parser.add_argument('--disable-db', '-d', action='store_true', help='Disable database operations')
    parser.add_argument('--db-uri', default=DATABASE_URI, help='Database URI')
    parser.add_argument('--vector-search', '-v', help='Vector-based semantic search query')
    parser.add_argument('--use-doc-embedding', action='store_true', help='Use documentation embeddings for search')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit search results')
    parser.add_argument('--embedding-dim', type=int, default=384, help='Dimension for embeddings')
    parser.add_argument('--detailed', action='store_true', help='Show detailed search results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set the logging level')
    
    # Go language specific options
    parser.add_argument('--go-package', help='Specific Go package name to analyze (e.g. "main")')
    parser.add_argument('--exclude-tests', action='store_true', 
                        help='Exclude test files (files ending with _test.go) from Go analysis')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        log_level = getattr(logging, args.log_level)
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)
    
    logger.debug("Debug logging enabled")
    
    try:
        indexer = CodeIndexer(
            disable_db=args.disable_db,
            db_uri=args.db_uri,
            embedding_dim=args.embedding_dim
        )
        
        # Handle vector search
        if args.vector_search and not args.disable_db:
            # Try to load the repository
            logger.info(f"Loading repository from database: {args.codebase}")
            repository = indexer.load_repository(
                args.codebase,
                go_package=args.go_package,
                exclude_tests=args.exclude_tests
            )
            
            if repository:
                # Execute vector search
                logger.info(f"Executing vector search for query: {args.vector_search}")
                results = indexer.vector_search(
                    args.codebase,
                    args.vector_search,
                    limit=args.limit,
                    use_doc_embedding=args.use_doc_embedding,
                    go_package=args.go_package,
                    exclude_tests=args.exclude_tests
                )
                
                if results:
                    result_handler = VectorSearchResults(repository, args.vector_search, results)
                    print(result_handler.explain(detailed=args.detailed))
                else:
                    print(f"No results found for vector search: {args.vector_search}")
            else:
                logger.error(f"Repository not found in database: {args.codebase}")
                print(f"Please index the repository first: {args.codebase}")
                sys.exit(1)
        
        # Handle regular search or indexing
        elif args.search:
            # Index the repository if not already indexed
            logger.info(f"Indexing repository: {args.codebase}")
            repository = indexer.index_repository(
                args.codebase,
                go_package=args.go_package,
                exclude_tests=args.exclude_tests
            )
            
            # Execute search
            results = repository.search(args.search)
            
            if args.expand:
                results = results.expand()
                
            if args.explain:
                explanation = results.explain()
                print(explanation)
            else:
                print(f"Found {len(results.results)} results.")
            
        else:
            # Just index the repository
            logger.info(f"Indexing repository: {args.codebase}")
            repository = indexer.index_repository(
                args.codebase,
                go_package=args.go_package,
                exclude_tests=args.exclude_tests
            )
            
            # Print some stats
            if repository and repository.semantic_graph:
                graph = repository.semantic_graph
                node_count = len(graph.nodes())
                edge_count = len(graph.edges())
                
                print(f"Repository indexed: {args.codebase}")
                print(f"  - Total nodes: {node_count}")
                print(f"  - Total relationships: {edge_count}")
                
                # Print node types if available
                node_types = {}
                for _, attrs in graph.nodes(data=True):
                    node_type = attrs.get('type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                    
                if node_types:
                    print("\nNode types:")
                    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {node_type}: {count}")
                
                # Print relationship types if available
                edge_types = {}
                for _, _, attrs in graph.edges(data=True):
                    edge_type = attrs.get('type', 'unknown')
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                    
                if edge_types:
                    print("\nRelationship types:")
                    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {edge_type}: {count}")
            else:
                print(f"No semantic graph created for repository: {args.codebase}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 