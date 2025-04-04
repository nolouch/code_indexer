# API Documentation

This document provides information about the HTTP API endpoints available in the service.

## Base URL

```
http://localhost:8000
```

## Common Response Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Service not initialized

## File Indexer API Endpoints

### Vector Search

Performs semantic vector search to find files conceptually related to your query.

**Endpoint:** `POST /file_indexer/vector_search`

**Request Body:**
```json
{
  "query": "implement transaction",
  "limit": 10,
  "show_content": true,
  "sibling_chunk_num": 2,
  "repository": "tidb"
}
```

**Parameters:**
- `query`: The search query string (required)
- `limit`: Maximum number of results to return (default: 10)
- `show_content`: Whether to include file content in results (default: true)
- `sibling_chunk_num`: Number of sibling chunks to include around each match (default: 1)
- `repository`: Repository name to filter results (optional)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/file_indexer/vector_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "limit": 10,
    "show_content": true
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "id": 42,
      "file_path": "/path/to/transaction.go",
      "language": "go",
      "repo_name": "tidb",
      "similarity": 0.92,
      "content": "// TransactionImpl implements the Transaction interface..."
    }
  ]
}
```

**Note**: Vector search requires TiDB with vector extension.

### Full Text Search

Performs text-based search using SQL LIKE queries for exact text matches.

**Endpoint:** `POST /file_indexer/full_text_search`

**Request Body:**
```json
{
  "query": "createTransaction",
  "limit": 5,
  "show_content": true,
  "repository": "tidb"
}
```

**Parameters:**
Same as Vector Search endpoint.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/file_indexer/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "createTransaction",
    "limit": 5,
    "show_content": true
  }'
```

### Get File by Path

Retrieves a file's content by its path with support for chunk pagination.

**Endpoint:** `GET /file_indexer/file`

**Query Parameters:**
- `file_path`: Path to the file (required)
- `sibling_chunk_num`: Number of sibling chunks to include around each match (default: 1)
- `offset`: Chunk offset for pagination, 0-based (optional, default: 0)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/file_indexer/file?file_path=%2Fpath%2Fto%2Ffile.go&offset=0&sibling_chunk_num=5" \
  -H "Accept: application/json"
```

**Example Response:**
```json
{
  "id": 123,
  "file_path": "/path/to/file.go",
  "language": "go",
  "repo_name": "tidb",
  "file_size": 25600,
  "chunks_count": 10,
  "match_method": "exact match",
  "content": "package main\n\nimport (\"fmt\")\n\nfunc main() {...}",
  "pagination": {
    "offset": 0,
    "limit": 5,
    "total_chunks": 10,
    "has_more": true,
    "next_offset": 5
  }
}
```

**Pagination:**

To retrieve a file in chunks, use the `offset` and `sibling_chunk_num` parameters:
1. Start with `offset=0` and a suitable `sibling_chunk_num` value
2. Use the `next_offset` value from the response for subsequent requests
3. Continue until `has_more` is false

### Get File by ID

Retrieves a file's content by its ID with pagination support.

**Endpoint:** `GET /file_indexer/file/{file_id}`

**Parameters:**
- `file_id`: ID of the file (path parameter)
- `sibling_chunk_num`: Number of sibling chunks to include around each match (query parameter, default: 1)
- `offset`: Chunk offset for pagination, 0-based (query parameter, optional, default: 0)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/file_indexer/file/123?offset=5&sibling_chunk_num=5" \
  -H "Accept: application/json"
```

**Example Response:**
```json
{
  "id": 123,
  "file_path": "/path/to/file.go",
  "language": "go",
  "repo_name": "tidb",
  "file_size": 25600,
  "chunks_count": 10,
  "content": "package main\n\nimport (\"fmt\")\n\nfunc main() {...}",
  "pagination": {
    "offset": 5,
    "limit": 5,
    "total_chunks": 10,
    "has_more": false,
    "next_offset": null
  }
}
```

**Pagination:**
Works the same way as the Get File by Path endpoint.

### Legacy Search (Deprecated)

**Endpoint:** `GET /file_indexer/search`

**Query Parameters:**
- `query`: The search query text (required)
- `limit`: Maximum number of results to return (default: 10)
- `show_content`: Whether to include file content in results (default: true)
- `sibling_chunk_num`: Number of sibling chunks to include around each match (default: 1)
- `repository`: Repository name to filter results (optional)
- `search_type`: Type of search to perform - "vector" or "full_text" (default: "vector")

**Example:**
```bash
curl -X GET "http://localhost:8000/file_indexer/search?query=implement%20transaction&search_type=vector"
```

**Note**: This endpoint is deprecated. Please use the dedicated vector or full-text search endpoints instead.

## Knowledge Base API Endpoints

### TiDB Knowledge Graph

Searches the TiDB knowledge graph for relevant information.

**Endpoint:** `GET /tidb_doc`

**Query Parameters:**
- `query`: The search query text (required)
- `top_k`: Top K results to return (default: 10)
- `threshold`: Similarity score threshold (default: 0.5)

**Example:**
```bash
curl -X GET "http://localhost:8000/tidb_doc?query=how%20to%20use%20transactions"
```

### Best Practices

Searches for best practices related to a given query.

**Endpoint:** `GET /best_practices`

**Query Parameters:**
- `query`: The search query text (required)
- `top_k`: Top K results to return (default: 10)
- `threshold`: Similarity score threshold (default: 0.5)

**Example:**
```bash
curl -X GET "http://localhost:8000/best_practices?query=code%20optimization"
```

## Code Graph API Endpoints

### Code Graph Search

Searches for code elements in a specific repository using vector search for semantic similarity.

**Endpoint:** `POST /code_graph/search`

**Request Body:**
```json
{
  "query": "implement transaction",
  "repository_name": "tidb",
  "limit": 5,
  "use_doc_embedding": false
}
```

**Parameters:**
- `query`: The search query text (required)
- `repository_name`: Name of the repository (required)
- `limit`: Maximum number of results to return (default: 10)
- `use_doc_embedding`: Whether to search using documentation embeddings instead of code embeddings (default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/code_graph/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "repository_name": "tidb",
    "limit": 5
  }'
```

### Code Graph Full Text Search

Searches for code elements in a specific repository using SQL LIKE queries for exact text matches in code context.

**Endpoint:** `POST /code_graph/full_text_search`

**Request Body:**
```json
{
  "query": "implement transaction",
  "repository_name": "tidb",
  "limit": 5
}
```

**Parameters:**
- `query`: The search query text (required)
- `repository_name`: Name of the repository (required)
- `limit`: Maximum number of results to return (default: 10)

**Example:**
```bash
curl -X POST "http://localhost:8000/code_graph/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RestoreAndRewriteMetaKVFiles",
    "repository_name": "tidb",
    "limit": 5
  }'
```

### Repository Statistics

Gets statistics about a repository.

**Endpoint:** `GET /code_graph/repositories/stats`

**Query Parameters:**
- `repository_name`: Name of the repository (required)

**Example:**
```bash
curl -X GET "http://localhost:8000/code_graph/repositories/stats?repository_name=tidb"
```

### List Repositories

Lists all indexed repositories.

**Endpoint:** `GET /code_graph/repositories`

**Example:**
```bash
curl -X GET "http://localhost:8000/code_graph/repositories"
```

## System API Endpoints

### API Status

Gets information about the API and its services.

**Endpoint:** `GET /`

**Example:**
```bash
curl -X GET "http://localhost:8000/"
```

**Example Response:**
```json
{
  "status": "running",
  "services": {
    "llm_client": true,
    "best_practices_kb": true,
    "code_graph_db": {
      "initialized": true,
      "available": true,
      "uri_configured": true
    },
    "file_indexer": true
  },
  "docs": "/docs"
}
```

### Database Check

Checks the database connection status.

**Endpoint:** `GET /db_check`

**Example:**
```bash
curl -X GET "http://localhost:8000/db_check"
```

**Example Response:**
```json
{
  "status": "ok",
  "connection": true,
  "test_query": true,
  "repositories_count": 5,
  "database_uri": "***"
}
```

## Best Practices

1. **Use specific queries** for better results
2. **Prefer vector search** for conceptual searches
   - Use `/file_indexer/vector_search` for file content
   - Use `/code_graph/search` for code elements
3. **Use full-text search** for exact keyword matches
   - Use `/file_indexer/full_text_search` for file content
   - Use `/code_graph/full_text_search` for code elements
4. **Filter by repository** to narrow down results
5. **Use pagination** for large files by setting `offset` and `sibling_chunk_num`

## Common Usage Patterns

### Working with Large Files

For large files, use pagination to retrieve content in chunks:

```bash
# First request: get first 5 chunks
curl -X GET "http://localhost:8000/file_indexer/file/123?offset=0&sibling_chunk_num=5"

# Second request: get next 5 chunks using the next_offset from previous response
curl -X GET "http://localhost:8000/file_indexer/file/123?offset=5&sibling_chunk_num=5"
```

### Finding and Retrieving Files

1. First, search for files using vector or full-text search
2. From the search results, get the file ID
3. Use the file ID to retrieve the complete file content

```bash
# Step 1: Search for files
curl -X POST "http://localhost:8000/file_indexer/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "createTransaction", "limit": 5, "show_content": false}'

# Step 2: Get the complete file using the ID from search results
curl -X GET "http://localhost:8000/file_indexer/file/42"
```

### Finding Code Elements

1. Search for code elements using either vector or full-text search based on your needs
2. Use vector search for conceptual matches and full-text search for exact text matches

```bash
# Vector search for conceptual matches
curl -X POST "http://localhost:8000/code_graph/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "implement transaction",
    "repository_name": "tidb",
    "limit": 5
  }'

# Full-text search for exact matches
curl -X POST "http://localhost:8000/code_graph/full_text_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "createTransaction",
    "repository_name": "tidb",
    "limit": 5
  }'
``` 