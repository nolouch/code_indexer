"""
# Code Indexer Tools

This directory contains language-specific tools used by the code indexer for parsing and analyzing source code.

## Directory Structure

```
tools/
├── go/                  # Go language tools
│   ├── analyzer/       # Go AST analyzer
│   │   └── ast_analyzer.go
│   └── go.mod         # Go module file
├── python/             # Python language tools
│   └── analyzer/      # Python AST analyzer (coming soon)
├── java/              # Java language tools
│   └── analyzer/      # Java parser tools (coming soon)
└── typescript/        # TypeScript tools
    └── analyzer/      # TypeScript analyzer (coming soon)
```

## Language-specific Tools

### Go Tools
- `ast_analyzer`: A Go tool that uses go/ast to analyze Go source code and output JSON-formatted AST information.

### Python Tools (Coming Soon)
- AST analyzer using Python's built-in ast module
- Type inference tools
- Import graph analyzer

### Java Tools (Coming Soon)
- JavaParser-based code analyzer
- Bytecode analyzer
- Maven/Gradle dependency analyzer

### TypeScript Tools (Coming Soon)
- TypeScript Compiler API-based analyzer
- Type system analyzer
- Module dependency graph generator

## Adding New Tools

When adding tools for a new language:

1. Create a new directory under `tools/` with your language name
2. Add an `analyzer` subdirectory for parsing tools
3. Include any necessary build files (e.g., go.mod, package.json)
4. Document tool usage and requirements in this README
5. Update the corresponding parser in `parsers/` to use the new tools

## Building Tools

Each language's tools may have different build requirements. Please refer to the language-specific README files in each directory for detailed build instructions.
""" 