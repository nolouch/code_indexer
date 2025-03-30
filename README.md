# Code Indexer

A semantic graph-based code search and understanding system that implements the Search-Expand-Refine paradigm for repository-level codebase analysis.

## Features

- **Multi-Language Code Analysis**
  - AST-based code parsing
  - Dependency analysis
  - Call graph generation
  - Type inference

- **Semantic Graph Construction**
  - Code-to-graph conversion
  - Semantic relationship extraction
  - Cross-file dependency linking
  - Documentation integration

- **Neural Code Search**
  - Semantic code embedding
  - Hybrid retrieval (structural + semantic)
  - Context-aware search
  - Multi-modal query support

- **Graph-based Expansion**
  - Relevance-guided exploration
  - Cross-reference tracking
  - Usage pattern mining
  - Context reconstruction

- **LLM-powered Refinement**
  - Query understanding
  - Result reranking
  - Code explanation
  - Answer synthesis

## Project Structure

```
code_indexer/
├── core/                 # Core abstractions and interfaces
│   ├── models/         # Data models and schemas
│   ├── parser.py      # Language parser interface
│   └── graph.py       # Graph interface
├── parsers/             # Language-specific parsers
│   ├── go/           
│   ├── python/
│   └── java/
├── semantic_graph/      # Semantic graph construction
│   ├── builder.py     # Graph builder
│   ├── embedders/     # Code embedders
│   └── relations.py   # Semantic relations
├── retrieval/           # Search components
│   ├── semantic.py    # Semantic search
│   ├── hybrid.py      # Hybrid search
│   └── ranker.py      # Result ranking
├── refinement/          # Result refinement
│   ├── llm.py         # LLM integration
│   ├── explainer.py   # Code explanation
│   └── synthesizer.py # Answer synthesis
└── tools/               # Language-specific tools
    ├── go/
    ├── python/
    └── java/
```

## Installation

```bash
# Install with poetry
poetry install

# Or with pip
pip install .
```

## Usage

```python
from code_indexer import CodeIndexer

# Initialize indexer
indexer = CodeIndexer()

# Index a repository
repo = indexer.index_repository("path/to/repo")

# Perform semantic search
results = repo.search("implement oauth authentication")

# Expand search results
expanded = results.expand()

# Refine and explain results
explanation = results.explain()
```

## Running Tests

The project uses pytest for unit and integration tests. To run the tests, follow these steps:

### Prerequisites

- Ensure you have all dependencies installed
- For Go parser tests, you need to have Go installed on your system

### Running All Tests

```bash
# Using Poetry
poetry run pytest

# Using Python directly
python -m pytest
```

### Running Specific Test Files

```bash
# Run tests for a specific module
python -m pytest tests/test_semantic_graph_builder.py

# Run tests for Go parser
python -m pytest tests/test_go_parser.py

# Run tests for the indexers
python -m tests.indexers.test_indexer
python -m tests.indexers.test_simple_indexer
```

### Test Directory Structure

```
tests/
├── indexers/                # Tests for code indexers
│   ├── test_indexer.py      # Tests for the database-enabled indexer
│   └── test_simple_indexer.py # Tests for the standalone indexer
├── test_go_parser.py        # Tests for Go parser
├── test_semantic_graph_builder.py # Tests for graph builder
└── test_core_models.py      # Tests for core data models
```

### Running Tests with Verbose Output

```bash
# For more detailed test output
python -m pytest -v

# With full traceback
python -m pytest --tb=native
```

### Test Coverage

```bash
# Run tests with coverage report
python -m pytest --cov=code_indexer

# Generate HTML coverage report
python -m pytest --cov=code_indexer --cov-report=html
```

## Supported Languages

Currently supported languages:
- Go
- Python (coming soon)
- Java (coming soon)
- TypeScript (coming soon)

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.