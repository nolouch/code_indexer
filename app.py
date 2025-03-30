#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from pathlib import Path


from code_indexer import CodeIndexer, VectorSearchResults
from setting.base import DATABASE_URI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Code Indexer')
    
    # Required repository path
    parser.add_argument('repo_path', help='Path to the code repository to analyze')
    
    # Database options
    parser.add_argument('--disable-db', action='store_true', help='Disable database operations')
    parser.add_argument('--db-uri', default=DATABASE_URI, help='Database URI (overrides default)')
    
    # Search options
    parser.add_argument('--search', '-s', help='Text search query')
    parser.add_argument('--vector-search', '-v', help='Vector-based semantic search query')
    parser.add_argument('--use-doc-embedding', action='store_true', help='Use documentation embeddings for search')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit search results')
    parser.add_argument('--embedding-dim', type=int, default=384, help='Dimension for embeddings')
    
    # Output options
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed search results')
    
    args = parser.parse_args()
    
    try:
        # Create code indexer
        indexer = CodeIndexer(
            disable_db=args.disable_db,
            db_uri=args.db_uri,
            embedding_dim=args.embedding_dim
        )
        
        # Check if repository path exists
        repo_path = Path(args.repo_path).resolve()
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            sys.exit(1)
        
        # Handle vector search
        if args.vector_search and not args.disable_db:
            # Try to load the repository
            logger.info(f"Loading repository from database: {repo_path}")
            repository = indexer.load_repository(str(repo_path))
            
            if repository:
                # Execute vector search
                logger.info(f"Executing vector search for query: {args.vector_search}")
                results = indexer.vector_search(
                    str(repo_path),
                    args.vector_search,
                    limit=args.limit,
                    use_doc_embedding=args.use_doc_embedding
                )
                
                if results:
                    result_handler = VectorSearchResults(repository, args.vector_search, results)
                    print(result_handler.explain(detailed=args.detailed))
                else:
                    print(f"No results found for vector search: {args.vector_search}")
            else:
                logger.error(f"Repository not found in database: {repo_path}")
                print(f"Please index the repository first: {repo_path}")
                sys.exit(1)
                
        # Handle regular search or indexing
        elif args.search:
            # Index the repository if not already indexed
            logger.info(f"Indexing repository: {repo_path}")
            repository = indexer.index_repository(str(repo_path))
            
            # Execute search (for now this is a placeholder)
            logger.info(f"Searching for: {args.search}")
            print(f"Search functionality is not implemented yet. Use --vector-search for semantic search.")
            
        else:
            # Just index the repository
            logger.info(f"Indexing repository: {repo_path}")
            repository = indexer.index_repository(str(repo_path))
            
            # Print some stats
            if repository and repository.semantic_graph:
                graph = repository.semantic_graph
                node_count = len(graph.nodes())
                edge_count = len(graph.edges())
                
                print(f"Repository indexed: {repo_path}")
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
                print(f"No semantic graph created for repository: {repo_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 