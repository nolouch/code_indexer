import argparse
import os
from pathlib import Path
import sys

# 添加项目根目录到路径，以便导入setting模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setting.embedding import EMBEDDING_MODEL
from setting.base import DATABASE_URI

# Use absolute import
from file_indexer.indexer import CodeIndexer
from file_indexer.database import CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description='Index code files into a database with vector search')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 获取默认数据库连接字符串
    default_db_uri = DATABASE_URI or 'mysql+pymysql://root@localhost:4000/code_index'
    default_model = EMBEDDING_MODEL["name"]
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index code files')
    index_parser.add_argument('directory', help='Directory to index')
    index_parser.add_argument('--db', default=default_db_uri, 
                            help=f'Database connection string (default: {default_db_uri})')
    index_parser.add_argument('--no-embeddings', action='store_true', 
                            help='Skip generating embeddings')
    index_parser.add_argument('--model', default=default_model,
                            help=f'Embedding model to use (default: {default_model})')
    index_parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                            help=f'Chunk size for large files in bytes (default: {CHUNK_SIZE} bytes)')
    index_parser.add_argument('--ignore-tests', action='store_true',
                            help='Ignore test files and directories')
    index_parser.add_argument('--repo-name', type=str, default=None,
                            help='Repository name to use (default: directory name)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar code')
    search_parser.add_argument('query', help='Query text to search for')
    search_parser.add_argument('--db', default=default_db_uri, 
                             help=f'Database connection string (default: {default_db_uri})')
    search_parser.add_argument('--limit', type=int, default=10, 
                             help='Maximum number of results (default: 10)')
    search_parser.add_argument('--no-content', action='store_true',
                            help='Do not show file content in results')
    search_parser.add_argument('--max-chunks', type=int, default=None,
                            help='Maximum number of chunks to show (default: all chunks)')
    search_parser.add_argument('--repository', type=str, default=None,
                            help='Repository name to filter results')
    search_parser.add_argument('--search-type', type=str, choices=['vector', 'full_text'], default='vector',
                            help='Type of search to perform (default: vector)')
    
    # Get entire file command
    get_parser = subparsers.add_parser('get', help='Get the complete content of a file')
    get_parser.add_argument('file_id', type=int, help='File ID')
    get_parser.add_argument('--db', default=default_db_uri,
                           help=f'Database connection string (default: {default_db_uri})')
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
            embedding_model=args.model,
            ignore_tests=args.ignore_tests
        )
        
        indexer.index_directory(
            directory, 
            generate_embeddings=not args.no_embeddings,
            repo_name=args.repo_name
        )
        
        indexer.close()
        
    elif args.command == 'search':
        if not args.query:
            print("Error: Query cannot be empty")
            return 1
        
        indexer = CodeIndexer(db_path=args.db)
        results = indexer.search_similar(
            args.query, 
            limit=args.limit,
            show_content=not args.no_content,
            max_chunks=args.max_chunks,
            repository=args.repository,
            search_type=args.search_type
        )
        
        # Display search type and repository info if filter was applied
        search_type_display = "Vector Search" if args.search_type == 'vector' else "Full Text Search"
        print(f"Search type: {search_type_display}")
        if args.repository:
            print(f"Searching in repository: {args.repository}")
            
        print(f"Found {len(results)} results:")
        for i, (file, similarity, preview) in enumerate(results, 1):
            language_info = f"[{file.language}]" if hasattr(file, 'language') and file.language else ""
            repo_info = f"[Repo: {file.repo_name}]" if hasattr(file, 'repo_name') and file.repo_name else ""
            
            print(f"{i}. {file.file_path} {language_info} {repo_info} (similarity: {similarity:.4f}) [ID: {file.id}]")
            
            # Print file content preview if requested
            if not args.no_content:
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