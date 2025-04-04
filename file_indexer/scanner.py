import os
import fnmatch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CodeScanner:
    def __init__(self, extensions=None, ignore_dirs=None, ignore_patterns=None):
        """Initialize the code scanner with filters"""
        # Default code file extensions to index
        self.extensions = extensions or [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.cs', '.scala', '.swift', '.kt'
        ]
        
        # Default directories to ignore
        self.ignore_dirs = ignore_dirs or [
            '.git', 'node_modules', 'venv', '.venv', 'env', '.env', 
            '__pycache__', 'build', 'dist', 'target', 'out', 'bin'
        ]
        
        # Default file patterns to ignore
        self.ignore_patterns = ignore_patterns or [
            '*.min.js', '*.min.css', '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dylib', 
            '*.dll', '*.exe', '*.o', '*.a', '*.lib', '*.class', '*.jar', '*.war',
            '*.log', '*.lock', '*.bak', '*.swp'
        ]
    
    def should_ignore(self, path):
        """Check if a path should be ignored"""
        path_parts = path.parts
        
        # Check if any part of the path is in ignore_dirs
        for part in path_parts:
            if part in self.ignore_dirs:
                return True
        
        # Check if the file matches any ignore pattern
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
        
        return False
    
    def has_valid_extension(self, file_path):
        """Check if a file has a valid code extension"""
        return any(file_path.endswith(ext) for ext in self.extensions)
    
    def scan_directory(self, root_dir):
        """Scan a directory and yield all valid code files"""
        root_path = Path(root_dir).resolve()
        logger.info(f"Scanning directory: {root_path}")
        
        for path in root_path.rglob('*'):
            if path.is_file() and self.has_valid_extension(str(path)) and not self.should_ignore(path):
                yield path.resolve()
    
    def get_file_content(self, file_path):
        """Get the content of a file with proper error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None 