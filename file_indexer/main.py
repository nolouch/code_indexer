import argparse
import os
from pathlib import Path

# Use absolute import
from file_indexer.indexer import CodeIndexer

def main():
    parser = argparse.ArgumentParser(description='Index code files into a database with vector search')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index code files')
    index_parser.add_argument('directory', help='Directory to index')
    index_parser.add_argument('--db', default='mysql+pymysql://root@localhost:4000/code_index', 
                            help='Database connection string (default: mysql+pymysql://root@localhost:4000/code_index)')
    index_parser.add_argument('--no-embeddings', action='store_true', 
                            help='Skip generating embeddings')
    index_parser.add_argument('--model', default='all-MiniLM-L6-v2',
                            help='Embedding model to use (default: all-MiniLM-L6-v2)')
    index_parser.add_argument('--chunk-size', type=int, default=1024*1024,
                            help='Chunk size for large files in bytes (default: 1MB)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar code')
    search_parser.add_argument('query', help='Query text to search for')
    search_parser.add_argument('--db', default='mysql+pymysql://root@localhost:4000/code_index', 
                             help='Database connection string (default: mysql+pymysql://root@localhost:4000/code_index)')
    search_parser.add_argument('--limit', type=int, default=10, 
                             help='Maximum number of results (default: 10)')
    search_parser.add_argument('--full-content', action='store_true',
                            help='Get full file content (instead of just the preview)')
    
    # Get entire file command
    get_parser = subparsers.add_parser('get', help='Get the complete content of a file')
    get_parser.add_argument('file_id', type=int, help='File ID')
    get_parser.add_argument('--db', default='mysql+pymysql://root@localhost:4000/code_indexer',
                           help='Database connection string (default: mysql+pymysql://root@localhost:4000/code_index)')
    get_parser.add_argument('--output', help='Output file path (prints to console if not specified)')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        directory = Path(args.directory).resolve()
        if not directory.exists() or not directory.is_dir():
            print(f"Error: {directory} is not a valid directory")
            return 1
        
        print(f"Indexing directory: {directory}")
        indexer = CodeIndexer(
            db_path=args.db,
            embedding_model=args.model
        )
        
        indexer.index_directory(
            directory, 
            generate_embeddings=not args.no_embeddings
        )
        
        indexer.close()
        
    elif args.command == 'search':
        if not args.query:
            print("Error: Query cannot be empty")
            return 1
        
        indexer = CodeIndexer(db_path=args.db)
        results = indexer.search_similar(args.query, limit=args.limit)
        
        print(f"Found {len(results)} results:")
        for i, (file, similarity, preview) in enumerate(results, 1):
            language_info = f"[{file.language}]" if hasattr(file, 'language') and file.language else ""
            print(f"{i}. {file.file_path} {language_info} (similarity: {similarity:.4f}) [ID: {file.id}]")
            
            # Print file content preview
            print(f"   {preview}")
            print()
        
        indexer.close()
    
    elif args.command == 'get':
        # Get complete file content
        from file_indexer.database import CodeFile, FileChunk, get_session, get_engine
        
        engine = get_engine(args.db)
        session = get_session(engine)
        
        # Query file and all its chunks
        file = session.query(CodeFile).filter_by(id=args.file_id).first()
        
        if not file:
            print(f"Error: File with ID {args.file_id} not found")
            return 1
            
        print(f"File: {file.file_path}")
        print(f"Size: {file.file_size} bytes, Chunks: {file.chunks_count}")
        print(f"Language: {file.language}")
        print()
        
        # Get all chunks and combine in order
        chunks = session.query(FileChunk).filter_by(file_id=file.id).order_by(FileChunk.chunk_index).all()
        
        # Combine complete content
        full_content = "".join(chunk.content for chunk in chunks)
        
        # Output content
        if args.output:
            # Write to file
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(full_content)
            print(f"File content written to {args.output}")
        else:
            # Print to console
            print("File content:")
            print("=" * 80)
            print(full_content)
            print("=" * 80)
        
        session.close()
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 