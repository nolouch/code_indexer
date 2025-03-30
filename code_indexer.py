import os
import sys
import argparse
import networkx as nx
import json
from pathlib import Path
import re
import logging
from typing import Dict, List, Optional, Any

# Import core modules
from core.models import CodeRepository, Module, Function, Class, Interface, CodeElement
from core.parser import LanguageParser

# Import language parsers
from parsers.go.parser import GoParser

# Import semantic graph building components
from semantic_graph.builder import SemanticGraphBuilder
from semantic_graph.embedders.code import CodeEmbedder
from semantic_graph.embedders.doc import DocEmbedder
from semantic_graph.relations import RelationType, SemanticRelation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Repository:
    """Repository class, includes search and result processing functions"""
    
    def __init__(self, path, graph):
        self.path = path
        self.graph = graph
        
    def search(self, query):
        """Execute semantic search"""
        logger.info(f"Searching for: {query}")
        return SearchResults(self, query)
        
class SearchResults:
    """Search results class, provides result processing and display functions"""
    
    def __init__(self, repository, query):
        self.repository = repository
        self.query = query
        self.results = []
        
        # Simple keyword search
        keywords = query.lower().split()
        
        # Search in node attributes
        for node, attrs in repository.graph.nodes(data=True):
            node_str = str(node).lower()
            attrs_str = str(attrs).lower()
            
            if any(keyword in node_str or keyword in attrs_str for keyword in keywords):
                self.results.append((node, attrs))
                
    def expand(self):
        """Expand search results, including related nodes"""
        expanded_results = self.results.copy()
        
        # For each result, add its neighbors
        for node, _ in self.results:
            if node in self.repository.graph:
                # Add neighbors
                for neighbor in self.repository.graph.neighbors(node):
                    neighbor_attrs = self.repository.graph.nodes[neighbor]
                    expanded_results.append((neighbor, neighbor_attrs))
                
                # Add predecessors
                for pred in self.repository.graph.predecessors(node):
                    pred_attrs = self.repository.graph.nodes[pred]
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

class SimpleEmbedder:
    """Simple embedder implementation, used for code and document embedding"""
    
    def embed(self, text):
        """Simple embedding implementation, returns fixed-dimension vector"""
        # Return a fixed-length vector
        # Actual implementation would use a more complex model
        return [0.0] * 384  # Return a 384-dimensional zero vector
        
    def batch_embed(self, texts):
        """Batch embedding implementation, embeds multiple texts"""
        # Simply call the embed method for each text separately
        return [self.embed(text) for text in texts]

class CodeIndexer:
    """Code indexer, uses existing modules to analyze code and build an index"""
    
    def __init__(self):
        self.repositories = {}
        self.parsers = {
            'golang': GoParser()
        }
        
    def index_repository(self, codebase_path):
        """Index a code repository and return a Repository object"""
        path = Path(codebase_path)
        
        # Check if path exists
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return Repository(path, nx.DiGraph())
        
        # Detect language
        language = self._detect_language(path)
        
        if language not in self.parsers:
            logger.warning(f"Language {language} is not supported yet.")
            return Repository(path, nx.DiGraph())
            
        # Use corresponding parser to parse the codebase
        logger.info(f"Parsing {language} codebase at {path}")
        parser = self.parsers[language]
        
        try:
            code_repo = parser.parse_directory(str(path))
            
            # Check if any modules were parsed
            if not code_repo.modules:
                logger.warning(f"No modules found in {path}. Check if it's a valid {language} codebase.")
            else:
                logger.info(f"Successfully parsed {len(code_repo.modules)} modules")
            
            # Build semantic graph
            # Create simple embedder instances
            code_embedder = SimpleEmbedder()
            doc_embedder = SimpleEmbedder()
            
            # Create graph and semantic graph builder
            graph = nx.MultiDiGraph()
            graph_builder = SemanticGraphBuilder(graph, code_embedder, doc_embedder)
            
            # Build semantic graph
            logger.info(f"Building semantic graph for {len(code_repo.modules)} modules...")
            graph = graph_builder.build_from_repository(code_repo)
            
            # Count nodes and edges in the graph
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            logger.info(f"Created semantic graph with {node_count} nodes and {edge_count} edges")
            
            # Create and store Repository object
            repo_obj = Repository(path, graph)
            self.repositories[str(path)] = repo_obj
            return repo_obj
            
        except Exception as e:
            logger.error(f"Error indexing repository: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty graph to avoid program crash
            return Repository(path, nx.DiGraph())
    
    def _detect_language(self, path):
        """Detect the main programming language of the codebase"""
        extensions = {}
        
        for root, _, files in os.walk(path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1
        
        if not extensions:
            logger.warning("No files found in the codebase.")
            return None
            
        # Sort by frequency
        sorted_exts = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Detected extensions: {sorted_exts}")
        
        # Map extensions to languages
        if '.go' in extensions:
            logger.info("Detected Golang codebase.")
            return 'golang'
            
        # Default to unsupported
        logger.warning("Unable to detect a supported language. Currently only Golang is supported.")
        return None
    
    def generate_index(self, repo_path, output_file='index.json'):
        """Generate and save index file"""
        if repo_path not in self.repositories:
            logger.warning(f"Repository {repo_path} not found. Index it first.")
            return
            
        repo = self.repositories[repo_path]
        graph = repo.graph
        
        data = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': str(node)}
            # Filter complex objects, only keep basic data types and necessary attributes
            for k, v in attrs.items():
                if k in ['type', 'language', 'file', 'line', 'docstring']:
                    node_data[k] = v
                elif isinstance(v, (str, int, float, bool)):
                    node_data[k] = v
            data['nodes'].append(node_data)
            
        # Handle multiple edges
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'source': str(source),
                'target': str(target)
            }
            # Filter complex objects, only keep basic data types and necessary attributes
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    edge_data[k] = v
            data['edges'].append(edge_data)
            
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Index generated and saved to {output_file}")
        
        # Generate statistics
        node_types = {}
        for _, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        logger.info("\nIndex Statistics:")
        for node_type, count in sorted(node_types.items()):
            logger.info(f"- {node_type}: {count}")
        logger.info(f"- edges: {graph.number_of_edges()}")


def main():
    parser = argparse.ArgumentParser(description='Code Indexer')
    parser.add_argument('codebase', help='Path to the codebase to analyze')
    parser.add_argument('--output', '-o', default='index.json', help='Output file for the index')
    parser.add_argument('--search', '-s', help='Search query')
    parser.add_argument('--explain', '-e', action='store_true', help='Explain search results')
    parser.add_argument('--expand', '-x', action='store_true', help='Expand search results')
    
    args = parser.parse_args()
    
    try:
        indexer = CodeIndexer()
        repo = indexer.index_repository(args.codebase)
        
        if args.search:
            results = repo.search(args.search)
            
            if args.expand:
                results = results.expand()
                
            if args.explain:
                explanation = results.explain()
                print(explanation)
        else:
            indexer.generate_index(str(Path(args.codebase)), args.output)
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 