#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path

from code_indexer import CodeIndexer, VectorSearchResults
from setting.base import DATABASE_URI

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_search(repo_path, db_uri, query="Person class", limit=10):
    """测试向量搜索功能"""
    print(f"使用数据库: {db_uri}")
    print(f"搜索仓库: {repo_path}")
    print(f"搜索查询: {query}")
    print(f"结果限制: {limit}")
    print("-" * 60)
    
    # 检测数据库类型
    db_type = "unknown"
    if db_uri:
        if 'tidb' in db_uri:
            db_type = "tidb"
        elif 'mysql' in db_uri:
            db_type = "mysql"
        elif 'sqlite' in db_uri:
            db_type = "sqlite"
    print(f"检测到数据库类型: {db_type}")
    
    # 创建索引器
    indexer = CodeIndexer(
        disable_db=False,
        db_uri=db_uri,
        embedding_dim=384
    )
    
    # 先索引仓库
    print("\n1. 索引仓库...")
    repository = indexer.index_repository(repo_path)
    if not repository or not repository.semantic_graph:
        print("错误: 索引仓库失败")
        return
    
    print(f"成功索引仓库，包含 {len(repository.semantic_graph.nodes())} 个节点和 {len(repository.semantic_graph.edges())} 条边")
    
    # 执行向量搜索
    print("\n2. 执行向量搜索...")
    results = indexer.vector_search(
        repo_path,
        query,
        limit=limit,
        use_doc_embedding=False
    )
    
    if not results:
        print("未找到结果")
        return
    
    # 展示搜索结果
    print("\n3. 搜索结果:")
    result_handler = VectorSearchResults(repository, query, results)
    result_text = result_handler.explain(detailed=True)
    print(result_text)
    
    print("\n测试完成")

def main():
    parser = argparse.ArgumentParser(description='测试向量搜索功能')
    parser.add_argument('repo_path', help='仓库路径')
    parser.add_argument('--db-uri', default=DATABASE_URI, help='数据库URI')
    parser.add_argument('--query', default='Person class', help='搜索查询')
    parser.add_argument('--limit', type=int, default=10, help='结果限制')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    repo_path = str(Path(args.repo_path).resolve())
    
    # 执行测试
    test_vector_search(repo_path, args.db_uri, args.query, args.limit)

if __name__ == '__main__':
    main() 