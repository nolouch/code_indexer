"""Code graph analysis module."""
from code_graph.models import Node, Edge, Repository
from code_graph.builder import SemanticGraphBuilder
from code_graph.db_manager import GraphDBManager
from code_graph.embedders import create_embedder

# Simplified initialization function for builder with OpenAI embeddings
def create_openai_builder(db_manager=None):
    """Create a semantic graph builder that uses OpenAI embeddings."""
    from setting.embedding import EMBEDDING_MODEL
    
    # Temporarily set provider to OpenAI
    original_provider = EMBEDDING_MODEL.get("provider")
    EMBEDDING_MODEL["provider"] = "openai"
    
    # Create embedders and builder
    code_embedder = create_embedder("code")
    doc_embedder = create_embedder("doc")
    builder = SemanticGraphBuilder(code_embedder, doc_embedder, db_manager=db_manager)
    
    # Restore original provider setting
    EMBEDDING_MODEL["provider"] = original_provider
    
    return builder

# 提供一个便捷函数来构建并保存按模块构建的图
def build_and_save_per_module(code_repo, repo_path, db_manager=None):
    """构建代码图并按模块保存到数据库。
    
    Args:
        code_repo: 代码仓库对象
        repo_path: 仓库路径
        db_manager: 可选的数据库管理器，如果不提供则创建默认实例
        
    Returns:
        nx.DiGraph: 构建的完整语义图
    """
    db_manager = db_manager or GraphDBManager()
    builder = SemanticGraphBuilder(db_manager=db_manager)
    return builder.build_from_repository(code_repo, repo_path, save_per_module=True) 