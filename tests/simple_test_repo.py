#!/usr/bin/env python
import logging
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to sys.path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# Import after setting up path
from code_indexer import CodeIndexer, Repository

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_repository():
    """Create a test repository with Go files."""
    repo_path = Path("./tests/test_repo")
    
    # Clear existing test repo if it exists
    if repo_path.exists():
        for file in repo_path.glob("**/*"):
            if file.is_file():
                file.unlink()
    else:
        repo_path.mkdir(parents=True, exist_ok=True)
    
    # Create a Go file with sample content
    go_file = repo_path / "main.go"
    
    go_content = """
package main

import (
    "fmt"
)

// Person represents a person with basic information
type Person struct {
    Name    string
    Age     int
}

// Greet returns a greeting from the person
func (p *Person) Greet() string {
    return fmt.Sprintf("Hello, my name is %s and I am %d years old", p.Name, p.Age)
}

// NewPerson creates a new person with the given information
func NewPerson(name string, age int) *Person {
    return &Person{
        Name: name,
        Age:  age,
    }
}

func main() {
    p := NewPerson("John", 30)
    fmt.Println(p.Greet())
}
"""
    
    with open(go_file, "w") as f:
        f.write(go_content)
    
    logger.info(f"Created test repository at {repo_path}")
    return str(repo_path)

def main():
    """Test the code indexer on a simple repository."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the code indexer with a simple repository")
    parser.add_argument("--db", dest="db_uri", help="Database connection URI (e.g., mysql+pymysql://root@127.0.0.1:4000/test)")
    parser.add_argument("--disable-db", dest="disable_db", action="store_true", help="Disable database functionality")
    args = parser.parse_args()

    # Set database URI if provided
    if args.db_uri:
        os.environ["DATABASE_URI"] = args.db_uri
        logger.info(f"Using database: {args.db_uri}")
    else:
        os.environ["DATABASE_URI"] = ""
        
    # Create a test repository
    repo_path = create_test_repository()
    
    # Initialize code indexer with database disabled if specified
    indexer = CodeIndexer(disable_db=args.disable_db)
    
    # Index the repository
    logger.info(f"Indexing repository: {repo_path}")
    repository = indexer.index_repository(repo_path)
    
    # Check if index file was created
    index_file = os.path.join(repo_path, "code_graph.json")
    if os.path.exists(index_file):
        logger.info(f"Index file was created: {index_file}")
        
        # Load the index file and check the content
        with open(index_file, "r") as f:
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
    else:
        logger.warning(f"Index file was not created: {index_file}")
    
    # Try a simple search
    if repository and hasattr(repository, 'search'):
        logger.info("Testing search functionality")
        search_queries = ["Person", "Greet", "main"]
        
        for query in search_queries:
            results = repository.search(query, max_results=5)
            logger.info(f"Search for '{query}' returned {len(results.results)} results")
            
    # If database is enabled, print database statistics
    if not args.disable_db and indexer.db_manager and indexer.db_manager.available:
        try:
            logger.info("Retrieving database statistics")
            stats = indexer.db_manager.get_repository_stats(repo_path)
            logger.info(f"Database statistics: {json.dumps(stats, indent=2, default=str)}")
            
            # Check database content
            from sqlalchemy import text
            from setting.db import SessionLocal
            
            with SessionLocal() as session:
                # Get repository record
                repo_query = "SELECT id, name, nodes_count, edges_count FROM repositories WHERE path = :path"
                repo_result = session.execute(text(repo_query), {"path": repo_path}).fetchone()
                
                if repo_result:
                    logger.info(f"Repository record: id={repo_result.id}, name={repo_result.name}, nodes={repo_result.nodes_count}, edges={repo_result.edges_count}")
                    
                    # Count nodes
                    node_count = session.execute(
                        text("SELECT COUNT(*) FROM nodes WHERE repository_id = :repo_id"),
                        {"repo_id": repo_result.id}
                    ).scalar()
                    logger.info(f"Node count in database: {node_count}")
                    
                    # Count edges
                    edge_count = session.execute(
                        text("SELECT COUNT(*) FROM edges WHERE repository_id = :repo_id"),
                        {"repo_id": repo_result.id}
                    ).scalar()
                    logger.info(f"Edge count in database: {edge_count}")
                    
                    # Check edge types
                    edge_types_query = """
                        SELECT relation_type, COUNT(*) as count 
                        FROM edges 
                        WHERE repository_id = :repo_id 
                        GROUP BY relation_type
                    """
                    edge_types_result = session.execute(text(edge_types_query), {"repo_id": repo_result.id}).fetchall()
                    logger.info(f"Edge types in database: {', '.join(f'{r.relation_type}({r.count})' for r in edge_types_result)}")
                    
                    # Check if edges use numeric IDs
                    edges_sample_query = """
                        SELECT e.id, e.source_id, e.target_id, e.relation_type,
                               src.node_id as source_node_id, tgt.node_id as target_node_id
                        FROM edges e
                        JOIN nodes src ON e.source_id = src.id
                        JOIN nodes tgt ON e.target_id = tgt.id
                        WHERE e.repository_id = :repo_id
                        LIMIT 5
                    """
                    edges_sample = session.execute(text(edges_sample_query), {"repo_id": repo_result.id}).fetchall()
                    logger.info("Sample edges from database:")
                    for edge in edges_sample:
                        logger.info(f"Edge {edge.id}: {edge.source_id}({edge.source_node_id}) -[{edge.relation_type}]-> {edge.target_id}({edge.target_node_id})")
                    
        except Exception as e:
            logger.error(f"Error retrieving database statistics: {e}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    main() 