#!/usr/bin/env python3
"""
诊断检索模块状态
"""

import os
import sys
from pathlib import Path

def check_bm25():
    """检查BM25模块"""
    print("🔍 检查BM25模块...")
    
    # 获取项目根目录
    current_file = Path(__file__)
    test_layer = current_file.parent.parent  # 测试与质量保障层
    project_root = test_layer.parent  # 项目根目录
    
    # 检查索引文件
    index_path = project_root / "检索与知识层" / "BM25" / "data" / "optimized_index" / "optimized_index.pkl.gz"
    if index_path.exists():
        print(f"✅ BM25索引文件存在: {index_path}")
        file_size = index_path.stat().st_size / (1024*1024)
        print(f"   文件大小: {file_size:.1f} MB")
    else:
        print(f"❌ BM25索引文件不存在: {index_path}")
        return False
    
    # 尝试导入BM25模块
    try:
        bm25_path = project_root / "检索与知识层" / "BM25"
        sys.path.insert(0, str(bm25_path))
        from bm25_retrieval.core.search_engine import BM25SearchEngine
        print("✅ BM25模块导入成功")
        return True
    except Exception as e:
        print(f"❌ BM25模块导入失败: {e}")
        return False

def check_vector():
    """检查向量检索模块"""
    print("\n🔍 检查向量检索模块...")
    
    # 获取项目根目录
    current_file = Path(__file__)
    test_layer = current_file.parent.parent
    project_root = test_layer.parent
    
    # 检查向量数据库
    vector_path = project_root / "检索与知识层" / "faiss_rag" / "向量数据库_faiss"
    if vector_path.exists():
        print(f"✅ 向量数据库目录存在: {vector_path}")
        
        # 检查SQLite文件
        sqlite_file = vector_path / "chroma.sqlite3"
        if sqlite_file.exists():
            file_size = sqlite_file.stat().st_size / (1024*1024)
            print(f"   SQLite文件大小: {file_size:.1f} MB")
        else:
            print("   ⚠️ SQLite文件不存在")
    else:
        print(f"❌ 向量数据库目录不存在: {vector_path}")
        return False
    
    # 尝试导入向量模块
    try:
        faiss_rag_path = project_root / "检索与知识层" / "faiss_rag"
        sys.path.insert(0, str(faiss_rag_path))
        from vector_retrieval_system.vector_retrieval import VectorRetrieval
        print("✅ 向量检索模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 向量检索模块导入失败: {e}")
        return False

def check_graph():
    """检查图检索模块"""
    print("\n🔍 检查图检索模块...")
    
    # 获取项目根目录
    current_file = Path(__file__)
    test_layer = current_file.parent.parent
    project_root = test_layer.parent
    
    # 检查Neo4j dump文件
    dump_path = project_root / "检索与知识层" / "Graphrag" / "Knowledge_Graph" / "neo4j.dump"
    if dump_path.exists():
        file_size = dump_path.stat().st_size / (1024*1024)
        print(f"✅ Neo4j dump文件存在: {dump_path}")
        print(f"   文件大小: {file_size:.1f} MB")
    else:
        print(f"❌ Neo4j dump文件不存在: {dump_path}")
    
    # 尝试导入图模块
    try:
        application_layer = project_root / "应用协调层"
        sys.path.insert(0, str(application_layer))
        from langchain.adapters.graph_adapter import GraphRetrievalAdapter
        print("✅ 图检索模块导入成功")
        
        # 尝试连接Neo4j
        try:
            adapter = GraphRetrievalAdapter(
                neo4j_uri="neo4j://127.0.0.1:7687",
                username="neo4j",
                password="hx1230047"
            )
            print("✅ Neo4j连接成功")
            return True
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            print("   请确保Neo4j服务正在运行")
            return False
            
    except Exception as e:
        print(f"❌ 图检索模块导入失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("检索模块诊断工具")
    print("=" * 60)
    
    bm25_ok = check_bm25()
    vector_ok = check_vector()
    graph_ok = check_graph()
    
    print("\n" + "=" * 60)
    print("诊断结果总结:")
    print("=" * 60)
    print(f"BM25模块: {'✅ 正常' if bm25_ok else '❌ 异常'}")
    print(f"向量检索: {'✅ 正常' if vector_ok else '❌ 异常'}")
    print(f"图检索: {'✅ 正常' if graph_ok else '❌ 异常'}")
    
    if not bm25_ok:
        print("\n🔧 BM25问题解决建议:")
        print("1. 检查索引文件是否存在")
        print("2. 重新构建BM25索引")
        print("3. 检查文件权限")
    
    if not graph_ok:
        print("\n🔧 图检索问题解决建议:")
        print("1. 启动Neo4j服务")
        print("2. 检查Neo4j连接配置")
        print("3. 导入知识图谱数据")
    
    print(f"\n当前只有 {'向量检索' if vector_ok else '无'} 模块可用")
    print("这就是为什么只看到vector: 0.5000结果的原因")

if __name__ == "__main__":
    main()
