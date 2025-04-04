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
  "max_chunks": 2,
  "repository": "tidb"
}
```

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

Retrieves a file's content by its path.

**Endpoint:** `GET /file_indexer/file`

**Query Parameters:**
- `file_path`: Path to the file (required)
- `max_chunks`: Maximum chunks to return (optional)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/file_indexer/file?file_path=%2Fpath%2Fto%2Ffile.go" \
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
  "chunks_count": 3,
  "match_method": "exact match",
  "content": "package main\n\nimport (\"fmt\")\n\nfunc main() {...}"
}
```

## Other API Endpoints

### API Status

**Endpoint:** `GET /`

**Example:**
```bash
curl -X GET "http://localhost:8000/"
```

### Database Check

**Endpoint:** `GET /db_check`

**Example:**
```bash
curl -X GET "http://localhost:8000/db_check"
```

### TiDB Knowledge Graph

**Endpoint:** `GET /tidb_doc`

**Example:**
```bash
curl -X GET "http://localhost:8000/tidb_doc?query=how%20to%20use%20transactions"
```

### Best Practices

**Endpoint:** `GET /best_practices`

**Example:**
```bash
curl -X GET "http://localhost:8000/best_practices?query=code%20optimization"
```

### Code Graph Search

**Endpoint:** `POST /code_graph/search`

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

### Repository Statistics

**Endpoint:** `GET /code_graph/repositories/stats`

**Example:**
```bash
curl -X GET "http://localhost:8000/code_graph/repositories/stats?repository_name=tidb"
```

### List Repositories

**Endpoint:** `GET /code_graph/repositories`

**Example:**
```bash
curl -X GET "http://localhost:8000/code_graph/repositories"
```

## Best Practices

1. **Use specific queries** for better results
2. **Prefer vector search** for conceptual searches
3. **Use full-text search** for exact keyword matches
4. **Filter by repository** to narrow down results
5. **Limit content size** with max_chunks for large files 