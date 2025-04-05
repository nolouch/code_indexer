#!/bin/bash

# Set environment variables
export FILEINDER_DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY_HERE"
export FILEINDER_LLM_PROVIDER="deepseek"
export FILEINDER_LLM_MODEL="deepseek-coder-v2-instruct"
export FILEINDER_GENERATE_COMMENTS="true"

# Directory path to index
CODE_DIRECTORY="$1"
if [ -z "$CODE_DIRECTORY" ]; then
  echo "Please provide a code directory path to index, for example: ./run_deepseek.sh /path/to/your/code"
  exit 1
fi

# Index code and generate LLM comments
echo "Starting indexing directory: $CODE_DIRECTORY"
echo "Using DeepSeek provider with model: $FILEINDER_LLM_MODEL"
python -m file_indexer.main index "$CODE_DIRECTORY"

# Display completion message
echo ""
echo "Indexing complete! You can now search code using the following command:"
echo "python -m file_indexer.main search \"your search query\" --use-comments"
echo ""
echo "If you only want to generate comments for already indexed files, use:"
echo "python -m file_indexer.main comments" 