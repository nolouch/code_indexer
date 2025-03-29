#!/usr/bin/env python3
import argparse
from pathlib import Path
from code_indexer import CodeIndexer

def main():
    parser = argparse.ArgumentParser(description='Code Indexer')
    parser.add_argument('codebase', help='Path to the codebase to analyze')
    parser.add_argument('--output', '-o', default='index.json', help='Output file for the index')
    parser.add_argument('--search', '-s', help='Search query')
    parser.add_argument('--explain', '-e', action='store_true', help='Explain search results')
    parser.add_argument('--expand', '-x', action='store_true', help='Expand search results')
    
    args = parser.parse_args()
    
    indexer = CodeIndexer()
    repo = indexer.index_repository(args.codebase)
    
    if args.search:
        results = repo.search(args.search)
        
        if args.expand:
            results = results.expand()
            
        if args.explain:
            explanation = results.explain()
            print(explanation)
    else:
        indexer.generate_index(str(Path(args.codebase)), args.output)

if __name__ == '__main__':
    main() 