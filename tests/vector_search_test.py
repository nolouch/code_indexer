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

# Setting DATABASE_URI to None for testing
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
    """Test vector search functionality."""
    # Check if sentence-transformers is installed
    try:
        import sentence_transformers
        logger.info("sentence-transformers is installed")
    except ImportError:
        logger.warning("sentence-transformers is not installed, trying to install it")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            logger.info("Successfully installed sentence-transformers")
            import sentence_transformers
        except Exception as e:
            logger.error(f"Failed to install sentence-transformers: {e}")
            logger.warning("Continuing with test, but semantic search may not work")
    
    # Create a test repository directory
    repo_path = Path("./tests/test_repo")
    if not repo_path.exists():
        repo_path.mkdir(parents=True, exist_ok=True)
    
    # Create a test graph
    semantic_graph = create_test_graph()
    logger.info(f"Created test graph with {len(semantic_graph.nodes())} nodes and {len(semantic_graph.edges())} edges")
    
    # Initialize code indexer with database disabled
    indexer = CodeIndexer(disable_db=True)
    
    # Add embeddings to nodes
    logger.info("Adding embeddings to nodes")
    for node_id, node_data in semantic_graph.nodes(data=True):
        content = node_data.get('content', '')
        if content:
            node_data['embedding'] = indexer.embedder.embed(content)
    
    # Create a repository manually
    repository = Repository(
        path=str(repo_path),
        semantic_graph=semantic_graph
    )
    
    # Save the graph to a JSON file
    output_file = os.path.join(str(repo_path), "code_graph.json")
    indexer.generate_index(semantic_graph, output_file)
    logger.info(f"Saved semantic graph to JSON file: {output_file}")
    
    # Try vector search (will fall back to local search since database is disabled)
    logger.info("Testing search with semantic queries:")
    search_queries = [
        "How to create a person?", 
        "How does greeting work?", 
        "What is the Person class?"
    ]
    
    for query in search_queries:
        logger.info(f"Searching for: {query}")
        results = repository.search(query, max_results=5)
        logger.info(f"Search for '{query}' returned {len(results.results)} results")
        if results.results:
            logger.info(results.explain())
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 