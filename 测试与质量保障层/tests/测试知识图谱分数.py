#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试知识图谱分数
直接检查知识图谱检索返回的分数情况
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "应用协调层"))

from 应用协调层.middle.adapters.graph_adapter import GraphRetrievalAdapter
from 应用协调层.middle.utils.entity_extractor import get_entity_extractor
from 应用协调层.middle.utils.entity_config import get_entity_config

async def test_knowledge_graph_scores():
    """测试知识图谱检索分数"""
    print("🔍 测试知识图谱检索分数")
    print("=" * 60)
    
    # 初始化知识图谱检索系统
    try:
        # 获取CSV文件路径
        config = get_entity_config()
        csv_path = config.get_kg_csv_path()
        csv_path = str(project_root / csv_path)
        
        # 创建知识图谱适配器
        graph_adapter = GraphRetrievalAdapter(
            neo4j_uri="neo4j://127.0.0.1:7687",
            username="neo4j",
            password="hx1230047",
            database="neo4j"
        )
        
        print("✅ 知识图谱适配器初始化成功")
        
    except Exception as e:
        print(f"❌ 知识图谱适配器初始化失败: {e}")
        return
    
    # 测试查询
    test_queries = [
        "红花可以治疗乳癖吗？",
        "人参有什么功效？",
        "失眠应该用什么穴位？",
        "黄芪和当归可以一起用吗？",
        "感冒了应该吃什么中药？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 测试查询 {i}: {query}")
        print("-" * 50)
        
        try:
            # 执行知识图谱搜索 - 使用complex_query_search方法
            results = await graph_adapter.complex_query_search(query, top_k=10)
            
            print(f"✅ 检索到 {len(results)} 个结果")
            
            # 显示前5个结果的分数
            for j, result in enumerate(results[:5], 1):
                score = getattr(result, 'score', 0)
                content = getattr(result, 'content', '')
                source = getattr(result, 'source', 'unknown')
                
                # 处理不同类型的内容
                if isinstance(content, dict):
                    if 'description' in content:
                        content_preview = content['description'][:100] + "..." if len(content['description']) > 100 else content['description']
                    elif 'relation' in content:
                        content_preview = f"关系: {content.get('relation', '')} -> {content.get('target', '')}"
                    else:
                        content_preview = str(content)[:100] + "..."
                else:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                
                print(f"  {j}. 分数: {score:.4f}")
                print(f"     来源: {source}")
                print(f"     内容: {content_preview}")
                print()
            
            # 统计分数分布
            scores = [getattr(r, 'score', 0) for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            
            print(f"📊 分数统计:")
            print(f"   平均分数: {avg_score:.4f}")
            print(f"   最高分数: {max_score:.4f}")
            print(f"   最低分数: {min_score:.4f}")
            
            # 检查是否有低分问题
            low_scores = [s for s in scores if s < 0.1]
            if low_scores:
                print(f"⚠️  发现 {len(low_scores)} 个低分结果 (< 0.1)")
            
            # 分析结果类型分布
            source_types = {}
            for result in results:
                source = getattr(result, 'source', 'unknown')
                source_types[source] = source_types.get(source, 0) + 1
            
            print(f"📈 结果类型分布:")
            for source_type, count in source_types.items():
                print(f"   {source_type}: {count} 个")
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
    
    print("\n🎯 测试完成")

def test_entity_extraction_scores():
    """测试实体提取分数"""
    print("\n🔍 测试实体提取分数")
    print("=" * 60)
    
    try:
        # 获取CSV文件路径
        config = get_entity_config()
        csv_path = config.get_kg_csv_path()
        csv_path = str(project_root / csv_path)
        
        # 创建知识图谱模式实体提取器
        entity_extractor = get_entity_extractor(use_kg=True, csv_file_path=csv_path)
        
        print("✅ 实体提取器初始化成功")
        
    except Exception as e:
        print(f"❌ 实体提取器初始化失败: {e}")
        return
    
    # 测试查询
    test_queries = [
        "红花可以治疗乳癖吗？",
        "人参有什么功效？",
        "失眠应该用什么穴位？",
        "黄芪和当归可以一起用吗？",
        "感冒了应该吃什么中药？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 测试查询 {i}: {query}")
        print("-" * 50)
        
        try:
            # 提取实体
            entities = entity_extractor.extract(query)
            
            print(f"✅ 提取到 {len(entities)} 个实体")
            
            # 显示提取的实体
            for j, entity in enumerate(entities, 1):
                print(f"  {j}. 实体: {entity}")
            
            # 如果有知识图谱系统，测试关系查询
            if hasattr(entity_extractor, 'kg_system') and entity_extractor.kg_system:
                print("\n🔗 知识图谱关系查询:")
                result = entity_extractor.kg_system.process_query(query)
                kg_results = result.get('kg_results', {})
                
                print(f"   匹配实体: {kg_results.get('matched_entities', 0)}/{kg_results.get('total_entities', 0)}")
                print(f"   总关系数: {kg_results.get('total_relations', 0)}")
                print(f"   覆盖率: {kg_results.get('coverage_rate', 0):.1%}")
                
                # 显示前几个关系
                relations = kg_results.get('relations', [])
                for rel_info in relations[:3]:  # 只显示前3个实体的关系
                    entity_name = rel_info['entity']
                    entity_relations = rel_info['relations']
                    print(f"   {entity_name} -> {len(entity_relations)} 个关系:")
                    for rel in entity_relations[:2]:  # 只显示前2个关系
                        print(f"     {rel['relation']}: {rel['target']}")
            
        except Exception as e:
            print(f"❌ 实体提取失败: {e}")
    
    print("\n🎯 实体提取测试完成")

async def test_hybrid_retrieval_scores():
    """测试混合检索中的智能路由和分别评估向量检索与知识图谱检索质量"""
    print("\n🔍 测试混合检索中的智能路由和检索质量")
    print("=" * 60)
    
    try:
        from 应用协调层.middle.core.retrieval_coordinator import HybridRetrievalCoordinator
        from 应用协调层.middle.adapters.graph_adapter import GraphRetrievalAdapter
        from 应用协调层.middle.adapters.simple_vector_adapter import SimpleVectorAdapter
        from 应用协调层.middle.models.data_models import RetrievalConfig, RetrievalSource
        
        # 创建向量检索适配器
        vector_adapter = SimpleVectorAdapter(
            persist_directory=str(project_root / "检索与知识层" / "faiss_rag" / "向量数据库_faiss"),
            model_path=r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base"
        )
        
        # 创建知识图谱适配器
        graph_adapter = GraphRetrievalAdapter(
            neo4j_uri="neo4j://127.0.0.1:7687",
            username="neo4j",
            password="hx1230047",
            database="neo4j"
        )
        
        # 创建混合检索协调器并设置两个适配器
        hybrid_coordinator = HybridRetrievalCoordinator(
            vector_adapter=vector_adapter,
            graph_adapter=graph_adapter
        )
        
        print("✅ 混合检索协调器初始化成功")
        
    except Exception as e:
        print(f"❌ 混合检索协调器初始化失败: {e}")
        return
    
    # 测试查询
    test_queries = [
        "红花可以治疗乳癖吗？",
        "人参有什么功效？",
        "失眠应该用什么穴位？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 测试查询 {i}: {query}")
        print("-" * 50)
        
        try:
            # 1. 显示智能路由分类结果
            print("🧠 智能路由分类:")
            if hybrid_coordinator.query_classifier:
                query_type, confidence = hybrid_coordinator.query_classifier.classify_with_confidence(query)
                weights = hybrid_coordinator.query_classifier.get_fusion_weights(query_type)
                print(f"   查询类型: {query_type.value}")
                print(f"   分类置信度: {confidence:.3f}")
                print(f"   推荐权重: vector={weights['vector']:.1f}, graph={weights['graph']:.1f}")
            else:
                print("   智能路由未启用，使用默认混合检索")
            
            # 2. 分别测试向量检索质量（独立测试）
            print("\n🔍 向量检索质量测试:")
            try:
                vector_config = RetrievalConfig(
                    enable_vector=True,
                    enable_graph=False,
                    top_k=5
                )
                vector_results = await vector_adapter.search(query, top_k=5)
                print(f"   检索结果数: {len(vector_results)} 个")
                
                if vector_results:
                    vector_scores = [getattr(r, 'score', 0) for r in vector_results]
                    avg_vector_score = sum(vector_scores) / len(vector_scores)
                    max_vector_score = max(vector_scores)
                    min_vector_score = min(vector_scores)
                    
                    print(f"   平均分数: {avg_vector_score:.4f}")
                    print(f"   最高分数: {max_vector_score:.4f}")
                    print(f"   最低分数: {min_vector_score:.4f}")
                    
                    # 显示前2个结果
                    for j, result in enumerate(vector_results[:2], 1):
                        score = getattr(result, 'score', 0)
                        content = getattr(result, 'content', '')
                        content_preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"   {j}. 分数: {score:.4f} - {content_preview}")
                else:
                    print("   无检索结果")
                    
            except Exception as e:
                print(f"   向量检索失败: {e}")
            
            # 3. 分别测试知识图谱检索质量（独立测试）
            print("\n🔗 知识图谱检索质量测试:")
            try:
                graph_config = RetrievalConfig(
                    enable_vector=False,
                    enable_graph=True,
                    top_k=5
                )
                graph_results = await graph_adapter.complex_query_search(query, top_k=5)
                print(f"   检索结果数: {len(graph_results)} 个")
                
                if graph_results:
                    graph_scores = [getattr(r, 'score', 0) for r in graph_results]
                    avg_graph_score = sum(graph_scores) / len(graph_scores)
                    max_graph_score = max(graph_scores)
                    min_graph_score = min(graph_scores)
                    
                    print(f"   平均分数: {avg_graph_score:.4f}")
                    print(f"   最高分数: {max_graph_score:.4f}")
                    print(f"   最低分数: {min_graph_score:.4f}")
                    
                    # 显示前2个结果
                    for j, result in enumerate(graph_results[:2], 1):
                        score = getattr(result, 'score', 0)
                        content = getattr(result, 'content', '')
                        content_preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"   {j}. 分数: {score:.4f} - {content_preview}")
                else:
                    print("   无检索结果")
                    
            except Exception as e:
                print(f"   知识图谱检索失败: {e}")
            
            # 4. 混合检索测试（显示融合效果和原始分数）
            print("\n🔄 混合检索融合测试:")
            try:
                hybrid_results = await hybrid_coordinator.retrieve(query)
                print(f"   融合结果数: {len(hybrid_results)} 个")
                
                if hybrid_results:
                    # 分析融合结果的来源贡献
                    vector_contributing = [r for r in hybrid_results if RetrievalSource.VECTOR in getattr(r, 'contributing_sources', [])]
                    graph_contributing = [r for r in hybrid_results if RetrievalSource.GRAPH in getattr(r, 'contributing_sources', [])]
                    
                    print(f"   向量贡献: {len(vector_contributing)} 个")
                    print(f"   知识图谱贡献: {len(graph_contributing)} 个")
                    
                    # 显示融合分数分布
                    fused_scores = [getattr(r, 'fused_score', 0) for r in hybrid_results]
                    avg_fused_score = sum(fused_scores) / len(fused_scores)
                    max_fused_score = max(fused_scores)
                    min_fused_score = min(fused_scores)
                    
                    print(f"   融合平均分数: {avg_fused_score:.4f}")
                    print(f"   融合最高分数: {max_fused_score:.4f}")
                    print(f"   融合最低分数: {min_fused_score:.4f}")
                    
                    # 显示原始分数信息（从source_scores中获取）
                    print("\n   📊 原始分数分析:")
                    for i, result in enumerate(hybrid_results[:3], 1):
                        source_scores = getattr(result, 'source_scores', {})
                        print(f"   结果{i}: 融合分数={getattr(result, 'fused_score', 0):.4f}")
                        for source, score in source_scores.items():
                            print(f"     {source}原始分数: {score:.4f}")
                else:
                    print("   无融合结果")
                    
            except Exception as e:
                print(f"   混合检索失败: {e}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n🎯 混合检索质量评估测试完成")

async def main():
    """主函数"""
    print("🚀 知识图谱分数测试")
    print("=" * 60)
    
    # 测试知识图谱检索分数
    await test_knowledge_graph_scores()
    
    # 测试实体提取分数
    test_entity_extraction_scores()
    
    # 测试混合检索中的知识图谱分数
    await test_hybrid_retrieval_scores()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("\n📋 测试总结:")
    print("1. 知识图谱检索分数测试")
    print("2. 实体提取分数测试")
    print("3. 混合检索中的知识图谱分数测试")

if __name__ == "__main__":
    asyncio.run(main())
