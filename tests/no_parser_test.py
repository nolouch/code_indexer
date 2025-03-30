#!/usr/bin/env python
import logging
import os
import sys
import json
from pathlib import Path
import networkx as nx

# Add parent directory to sys.path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# Set empty DATABASE_URI for testing
os.environ["DATABASE_URI"] = ""

from code_indexer import CodeIndexer, Repository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_graph():
    """Create a test graph without using parsers."""
    # Create a directed graph
    graph = nx.DiGraph()
    
    # Add module node
    graph.add_node("module:main", 
                 type="module", 
                 name="main", 
                 file_path="main.go", 
                 content="package main")
    
    # Add struct node
    graph.add_node("struct:main:Person", 
                 type="class", 
                 name="Person", 
                 file_path="main.go", 
                 content="type Person struct { Name string, Age int }", 
                 docstring="Person represents a person with basic information")
    
    # Add edge from module to struct
    graph.add_edge("module:main", "struct:main:Person", type="contains")
    
    # Add function node
    graph.add_node("func:main:NewPerson", 
                 type="function", 
                 name="NewPerson", 
                 file_path="main.go", 
                 content="func NewPerson(name string, age int) *Person { return &Person{Name: name, Age: age} }", 
                 docstring="NewPerson creates a new person with the given information",
                 signature="func NewPerson(name string, age int) *Person")
    
    # Add edge from module to function
    graph.add_edge("module:main", "func:main:NewPerson", type="contains")
    
    # Add method node
    graph.add_node("method:main:Person:Greet", 
                 type="method", 
                 name="Greet", 
                 file_path="main.go", 
                 content="func (p *Person) Greet() string { return fmt.Sprintf(\"Hello, my name is %s and I am %d years old\", p.Name, p.Age) }", 
                 docstring="Greet returns a greeting from the person",
                 signature="func (p *Person) Greet() string")
    
    # Add edge from struct to method
    graph.add_edge("struct:main:Person", "method:main:Person:Greet", type="contains")
    
    return graph

def main():
    """Test the code indexer with a manually created graph."""
    # Create a test repository directory
    repo_path = Path("./tests/test_repo")
    if not repo_path.exists():
        repo_path.mkdir(parents=True, exist_ok=True)
    
    # Create a test graph
    semantic_graph = create_test_graph()
    logger.info(f"Created test graph with {len(semantic_graph.nodes())} nodes and {len(semantic_graph.edges())} edges")
    
    # Initialize code indexer with database disabled
    indexer = CodeIndexer(disable_db=True)
    
    # Create a repository manually
    repository = Repository(
        path=str(repo_path),
        semantic_graph=semantic_graph
    )
    
    # Save the graph to a JSON file
    output_file = os.path.join(str(repo_path), "code_graph.json")
    indexer.generate_index(semantic_graph, output_file)
    logger.info(f"Saved semantic graph to JSON file: {output_file}")
    
    # Check if index file was created
    if os.path.exists(output_file):
        logger.info(f"Index file was created: {output_file}")
        
        # Load the index file and check the content
        with open(output_file, "r") as f:
            index_data = json.load(f)
            
        logger.info(f"Index contains {len(index_data['nodes'])} nodes and {len(index_data['edges'])} edges")
        
        # Log node types
        node_types = {}
        for node_id, node_data in index_data['nodes'].items():
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        logger.info(f"Node types: {node_types}")
        
        # Log edge types
        edge_types = {}
        for edge in index_data['edges']:
            edge_type = edge['data'].get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
        logger.info(f"Edge types: {edge_types}")
    
    # Try a simple search
    logger.info("Testing search functionality")
    search_queries = ["Person", "Greet", "main"]
    
    for query in search_queries:
        results = repository.search(query, max_results=5)
        logger.info(f"Search for '{query}' returned {len(results.results)} results")
        if results.results:
            logger.info(results.explain())
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 