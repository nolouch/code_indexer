"""
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
""" 