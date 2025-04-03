# Code Indexer

A semantic graph-based code search and understanding system that leverages LLM and vector embeddings for intelligent code analysis and retrieval.

## Features

- **Intelligent Code Analysis**
  - AST-based code parsing
  - Semantic understanding
  - Cross-file dependency analysis
  - Documentation integration

- **Vector-based Search**
  - Code embedding with state-of-the-art models
  - Semantic similarity search
  - Context-aware retrieval
  - Multi-modal query support

- **LLM Integration**
  - Natural language query understanding
  - Code explanation generation
  - Best practices analysis
  - Documentation generation

- **Knowledge Management**
  - Code knowledge base construction
  - Semantic relationship extraction
  - Context preservation
  - Reusable code patterns

## Project Structure

```
code_indexer/
├── core/                 # Core abstractions and interfaces
├── parsers/             # Language-specific code parsers
├── code_graph/          # Code graph construction and analysis
├── knowledgebase/       # Knowledge base management
├── retrieval/           # Search and retrieval components
├── llm/                 # LLM integration and prompts
├── utils/              # Utility functions and helpers
├── setting/            # Configuration and settings
├── tools/              # Development and analysis tools
└── tests/              # Test suite
```

## Key Components

- `code_indexer.py`: Main indexing and analysis logic
- `api.py`: REST API endpoints
- `app.py`: Web application interface
- `build_doc.ipynb`: Documentation generation notebook
- `build_best_practices.ipynb`: Best practices analysis notebook

## Installation

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
pip install -r requirements.txt
```

## Configuration

The system requires several environment variables to be set:

- OpenAI API credentials for LLM integration
- Vector database configuration
- Other service-specific settings

Create a `.env` file in the project root with the necessary configurations.

## Usage

### Starting the API Server

```bash
uvicorn api:app --reload
```

### Running the Web Interface

```bash
python app.py
```

### Using the Python API

```python
from code_indexer import CodeIndexer

# Initialize the indexer
indexer = CodeIndexer()

# Index a codebase
indexer.index_repository("path/to/repo")

# Perform semantic search
results = indexer.search("implement authentication flow")

# Generate documentation
docs = indexer.generate_documentation()
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=code_indexer

# Run specific test file
pytest tests/test_specific.py
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking

Run formatters:
```bash
black .
isort .
```

## Current Status

- [x] Core indexing functionality
- [x] Vector-based search
- [x] LLM integration
- [x] Basic API endpoints
- [x] Documentation generation
- [ ] Complete language support
- [ ] Advanced graph analysis
- [ ] UI improvements

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see the LICENSE file for details.