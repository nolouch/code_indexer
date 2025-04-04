# Code File Indexer

A tool for indexing code files into a TiDB database, supporting semantic search and chunked storage for large files.

## Features

- Scans directories for code files
- Indexes file paths and contents into a database
- Generates vector embeddings for semantic search
- Utilizes TiDB's native vector search capabilities
- Automatically chunks large files for better performance
- Provides a simple command-line interface for indexing and searching

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Create a TiDB database:

```sql
CREATE DATABASE code_index;
```

## Usage

### Indexing Files

To index code files in a directory:

```bash
python code_indexer_cli.py index /path/to/your/code
```

Options:
- `--db`: Specify a custom database connection string (default: `mysql+pymysql://root@localhost:4000/code_index`)
- `--no-embeddings`: Skip generating embeddings (faster but disables semantic search)
- `--model`: Specify a different embedding model (default: 'all-MiniLM-L6-v2')
- `--chunk-size`: Specify chunk size for large files in bytes (default: 1MB)

### Searching Files

To search for code similar to a query:

```bash
python code_indexer_cli.py search "your search query"
```

Options:
- `--db`: Specify a custom database connection string (default: `mysql+pymysql://root@localhost:4000/code_index`)
- `--limit`: Maximum number of results to return (default: 10)

### Getting Complete File Content

To retrieve the complete content of a file by ID:

```bash
python code_indexer_cli.py get 123
```

Options:
- `--db`: Specify a custom database connection string (default: `mysql+pymysql://root@localhost:4000/code_index`)
- `--output`: Output file path (prints to console if not specified)

## TiDB Vector Support

This tool leverages TiDB's built-in vector capabilities:
- Stores embeddings as TiDB VECTOR data type
- Creates vector indexes for fast similarity search
- Uses TiDB's VEC_COSINE_DISTANCE function for semantic ranking
- Uses LONGTEXT for storing large file contents

## Large File Handling

The tool automatically handles large files:
- Files larger than 100MB are truncated
- File content is automatically chunked (default 1MB per chunk)
- When generating embeddings, representative parts of the code are extracted:
  - First section of the file (imports, header comments)
  - Function and class definitions
  - Important comments
- This ensures efficiency while maintaining semantic meaning

If TiDB is not available, the tool will fall back to in-memory vector search.

## Database Structure

- `code_files` table: Stores file metadata and embedding vectors
- `file_chunks` table: Stores chunked file contents

## Troubleshooting

If you encounter database errors:
- Make sure TiDB is running and accessible
- Ensure your database connection string is correct
- Check that the code_index database exists
- Ensure your TiDB version supports vector data types and functions 