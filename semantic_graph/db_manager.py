from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import pymysql
import networkx as nx
import os
import configparser
from pathlib import Path

from .models import Node, Repository, GraphMetadata, node_relationships

class GraphData:
    """Graph data table."""
    __tablename__ = 'graphs'
    
    def __init__(self, repo_path, nodes, edges):
        self.repo_path = repo_path
        self.nodes = nodes
        self.edges = edges

class GraphDBManager:
    """Manages graph persistence in database."""
    
    def __init__(self, db_url: str = None):
        """Initialize database connection.
        
        Args:
            db_url: Database connection URL (not used with direct PyMySQL connection)
        """
        # Load credentials from environment variables or config file
        db_host = os.environ.get("DB_HOST", "gateway01.ap-southeast-1.prod.aws.tidbcloud.com")
        db_port = int(os.environ.get("DB_PORT", "4000"))
        db_user = os.environ.get("DB_USER", "")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_name = os.environ.get("DB_NAME", "sample_data")
        ssl_ca = os.environ.get("DB_SSL_CA", "/etc/ssl/cert.pem")
        
        # If environment variables are not set, try to load from config file
        if not db_user or not db_password:
            config = configparser.ConfigParser()
            config_path = os.environ.get("DB_CONFIG_PATH", "config/database.ini")
            
            if os.path.exists(config_path):
                config.read(config_path)
                if "database" in config:
                    db_section = config["database"]
                    db_host = db_section.get("host", db_host)
                    db_port = int(db_section.get("port", str(db_port)))
                    db_user = db_section.get("user", db_user)
                    db_password = db_section.get("password", db_password)
                    db_name = db_section.get("database", db_name)
                    ssl_ca = db_section.get("ssl_ca", ssl_ca)
        
        # Ensure credentials are provided
        if not db_user or not db_password:
            raise ValueError("Database credentials not found. Please set DB_USER and DB_PASSWORD environment variables or provide them in the config file.")
            
        self.connection = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            ssl_verify_cert=True,
            ssl_verify_identity=True,
            ssl_ca=ssl_ca
        )
        self._setup_tables()
        
    def _setup_tables(self):
        """Create necessary tables if they don't exist."""
        cursor = self.connection.cursor()
        
        # Create graphs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graphs (
                repo_path VARCHAR(255) PRIMARY KEY,
                nodes JSON,
                edges JSON
            )
        """)
        
        # Create repositories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repositories (
                id INT AUTO_INCREMENT PRIMARY KEY,
                path VARCHAR(255) UNIQUE,
                name VARCHAR(255),
                last_indexed VARCHAR(255),
                repo_data JSON
            )
        """)
        
        # Create graph_metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                repository_id INT,
                node_count INT,
                edge_count INT,
                last_updated VARCHAR(255),
                embedder_config JSON,
                FOREIGN KEY (repository_id) REFERENCES repositories(id)
            )
        """)
        
        self.connection.commit()
        cursor.close()
        
    def save_graph(self, graph: nx.MultiDiGraph, repo_path: str):
        """Save graph to database.
        
        Args:
            graph: Graph to save
            repo_path: Repository path as identifier
        """
        cursor = self.connection.cursor()
        
        # Convert graph to JSON-serializable format
        nodes = dict(graph.nodes(data=True))
        edges = [
            {
                "source": u,
                "target": v,
                "key": k,
                "data": d
            }
            for u, v, k, d in graph.edges(data=True, keys=True)
        ]
        
        # Convert to JSON strings
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        # Save to database
        cursor.execute("""
            INSERT INTO graphs (repo_path, nodes, edges)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
            nodes = %s, edges = %s
        """, (repo_path, nodes_json, edges_json, nodes_json, edges_json))
        
        self.connection.commit()
        cursor.close()
        
    def load_graph(self, repo_path: str) -> nx.MultiDiGraph:
        """Load graph from database.
        
        Args:
            repo_path: Repository path as identifier
            
        Returns:
            Loaded graph or None if not found
        """
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM graphs WHERE repo_path = %s", (repo_path,))
        
        data = cursor.fetchone()
        cursor.close()
        
        if not data:
            return None
            
        # Create new graph
        graph = nx.MultiDiGraph()
        
        # Add nodes
        nodes = json.loads(data["nodes"])
        for node, attrs in nodes.items():
            graph.add_node(node, **attrs)
            
        # Add edges
        edges = json.loads(data["edges"])
        for edge in edges:
            graph.add_edge(
                edge["source"],
                edge["target"],
                key=edge["key"],
                **edge["data"]
            )
            
        return graph
        
    def _get_or_create_repository(self, path: str) -> tuple:
        """Get existing repository or create new one"""
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM repositories WHERE path = %s", (path,))
        repo = cursor.fetchone()
        
        if not repo:
            now = datetime.now().isoformat()
            name = path.split('/')[-1]
            repo_data = json.dumps({})
            
            cursor.execute("""
                INSERT INTO repositories (path, name, last_indexed, repo_data)
                VALUES (%s, %s, %s, %s)
            """, (path, name, now, repo_data))
            
            self.connection.commit()
            
            # Get the inserted ID
            cursor.execute("SELECT LAST_INSERT_ID() as id")
            repo_id = cursor.fetchone()["id"]
            repo = {"id": repo_id, "name": name}
            
        cursor.close()
        return repo
        
    def get_repository_stats(self, repository_path: str) -> Dict[str, Any]:
        """Get statistics about the stored graph
        
        Args:
            repository_path: Path to the code repository
            
        Returns:
            Dictionary containing graph statistics
        """
        cursor = self.connection.cursor(pymysql.cursors.DictCursor)
        
        # Get repository
        cursor.execute("SELECT * FROM repositories WHERE path = %s", (repository_path,))
        repo = cursor.fetchone()
        
        if not repo:
            cursor.close()
            return {}
            
        # Get metadata
        cursor.execute("SELECT * FROM graph_metadata WHERE repository_id = %s", (repo["id"],))
        metadata = cursor.fetchone()
        
        cursor.close()
        
        if not metadata:
            return {}
            
        return {
            'repository': repo["name"],
            'nodes': metadata["node_count"],
            'edges': metadata["edge_count"],
            'last_updated': metadata["last_updated"],
            'embedder_config': json.loads(metadata["embedder_config"])
        }