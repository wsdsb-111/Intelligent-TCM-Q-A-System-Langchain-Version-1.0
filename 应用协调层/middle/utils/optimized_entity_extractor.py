"""
优化的实体提取器 - 基于知识图谱的实体提取
使用CSV知识图谱数据作为实体库，实现高效的实体提取和关系查询
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import logging

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class OptimizedEntityExtractor:
    """优化的实体提取器 - 基于知识图谱"""
    
    def __init__(self, csv_file_path: Optional[str] = None):
        """
        初始化实体提取器
        
        Args:
            csv_file_path: 知识图谱CSV文件路径，如果为None则使用默认路径
        """
        self.csv_file_path = csv_file_path or self._get_default_csv_path()
        self.entities = set()  # 所有实体名称
        self.kg_relations = {}  # 知识图谱关系数据
        self.loaded = False
        self.kg_full_path = None  # 完整知识图谱文件路径（用于查询关系）
        
    def _get_default_csv_path(self) -> str:
        """获取默认的CSV文件路径"""
        # 默认路径：测试与质量保障层/testdataset/merged_datasets_classified.csv
        # 这个文件包含融合后的分类数据集
        default_path = project_root / "测试与质量保障层" / "testdataset" / "merged_datasets_classified.csv"
        return str(default_path)
    
    def _get_entities_only_csv_path(self) -> str:
        """获取纯实体CSV文件路径（不包含关系词）"""
        entities_path = project_root / "测试与质量保障层" / "testdataset" / "merged_datasets_classified.csv"
        return str(entities_path)
    
    def load_kg_data(self) -> bool:
        """加载知识图谱数据"""
        if not os.path.exists(self.csv_file_path):
            logger.error(f"知识图谱CSV文件不存在: {self.csv_file_path}")
            return False
        
        try:
            logger.info(f"🔄 加载融合分类数据: {self.csv_file_path}")
            
            import pandas as pd
            
            # 尝试读取CSV文件，可能没有列名
            try:
                df = pd.read_csv(self.csv_file_path, encoding='utf-8', header=0)
                
                # 检查是否有'术语'列
                if '术语' in df.columns:
                    terms = df['术语'].astype(str).tolist()
                else:
                    # 如果没有'术语'列，使用第一列
                    terms = df.iloc[:, 0].astype(str).tolist()
            except:
                # 如果失败，尝试不使用header读取
                df = pd.read_csv(self.csv_file_path, encoding='utf-8', header=None)
                terms = df.iloc[:, 0].astype(str).tolist()
            
            entity_count = 0
            for term in terms:
                if term and term != 'nan':
                    self.entities.add(term)
                    entity_count += 1
                
                if entity_count % 10000 == 0:
                    logger.info(f"   已加载 {entity_count} 个实体...")
            
            logger.info(f"✅ 融合分类数据加载完成!")
            logger.info(f"   实体数量: {len(self.entities)}")
            logger.info(f"   ℹ️  已加载融合后的分类数据集")
            
            # 加载完整知识图谱关系数据
            self._load_kg_relations()
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 实体数据加载失败: {e}")
            return False
    
    def _load_kg_relations(self):
        """
        加载完整知识图谱关系数据（可选功能）
        
        注意：这个功能是可选的，主要用于离线测试。
        在生产环境中，关系查询应该使用 Neo4j 图数据库（graph_adapter.py）。
        如果完整的 CSV 文件不存在，不影响实体提取功能。
        """
        # 完整知识图谱文件路径
        self.kg_full_path = project_root / "测试与质量保障层" / "testdataset" / "knowledge_graph_merged_deduplicated.csv"
        
        if not os.path.exists(self.kg_full_path):
            logger.info(f"ℹ️  完整知识图谱CSV文件不存在，跳过关系数据加载")
            logger.info(f"   这不影响实体提取功能，关系查询将使用 Neo4j 数据库")
            return
        
        try:
            logger.info(f"🔄 加载知识图谱关系数据...")
            
            relation_count = 0
            with open(self.kg_full_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    source_nar = row.get('source_nar', '').strip()
                    target_nar = row.get('target_nar', '').strip()
                    relations = row.get('relations', '').strip()
                    
                    # 只加载实体的关系（source和target都在实体集合中）
                    if source_nar in self.entities and target_nar in self.entities and relations:
                        if source_nar not in self.kg_relations:
                            self.kg_relations[source_nar] = []
                        
                        self.kg_relations[source_nar].append({
                            'target': target_nar,
                            'relation': relations,
                            'description': row.get('description', ''),
                            'weight': float(row.get('weight', 1.0)),
                            'confidence': float(row.get('confidence', 0.7))
                        })
                        
                        relation_count += 1
            
            logger.info(f"✅ 关系数据加载完成! 共 {relation_count} 条关系")
            
        except Exception as e:
            logger.warning(f"⚠️ 关系数据加载失败: {e}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表，每个实体包含mention, start, end, confidence等信息
        """
        if not self.loaded:
            logger.warning("⚠️ 实体库未加载，请先加载数据")
            return []
        
        entities = []
        
        # 过滤实体：只保留长度>=2的实体，避免单字干扰
        # 对于中医领域，大部分有意义的实体都是2个字以上
        filtered_entities = [e for e in self.entities if len(e) >= 2]
        
        # 按长度排序，优先匹配长实体
        sorted_entities = sorted(filtered_entities, key=len, reverse=True)
        
        for entity_name in sorted_entities:
            # 在文本中查找实体
            start = 0
            while True:
                pos = text.find(entity_name, start)
                if pos == -1:
                    break
                
                # 检查是否被其他已匹配的实体覆盖
                is_overlapped = False
                for existing_entity in entities:
                    if (pos < existing_entity['end'] and 
                        pos + len(entity_name) > existing_entity['start']):
                        is_overlapped = True
                        break
                
                if not is_overlapped:
                    entities.append({
                        'mention': entity_name,
                        'start': pos,
                        'end': pos + len(entity_name),
                        'confidence': 1.0,
                        'method': 'kg_rule'
                    })
                
                start = pos + 1
        
        # 按位置排序
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def query_kg_relations(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        查询知识图谱关系
        
        Args:
            entities: 实体列表
            
        Returns:
            知识图谱查询结果
        """
        results = {
            'total_entities': len(entities),
            'matched_entities': 0,
            'total_relations': 0,
            'relations': []
        }
        
        for entity in entities:
            entity_name = entity['mention']
            if entity_name in self.kg_relations:
                results['matched_entities'] += 1
                relations = self.kg_relations[entity_name]
                results['total_relations'] += len(relations)
                
                results['relations'].append({
                    'entity': entity_name,
                    'relations': relations
                })
        
        results['coverage_rate'] = results['matched_entities'] / results['total_entities'] if results['total_entities'] > 0 else 0
        
        return results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询，提取实体并查询知识图谱
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果，包含实体和知识图谱关系
        """
        # 提取实体
        entities = self.extract_entities(query)
        
        # 查询知识图谱关系
        kg_results = self.query_kg_relations(entities)
        
        return {
            'query': query,
            'entities': entities,
            'entity_count': len(entities),
            'kg_results': kg_results
        }
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """获取实体库统计信息"""
        if not self.loaded:
            return {}
        
        stats = {
            'total_entities': len(self.entities),
            'total_relations': sum(len(relations) for relations in self.kg_relations.values()),
            'entities_with_relations': len(self.kg_relations)
        }
        
        return stats

class OptimizedRetrievalSystem:
    """优化的检索系统 - 集成实体提取和知识图谱查询"""
    
    def __init__(self, csv_file_path: Optional[str] = None):
        self.extractor = OptimizedEntityExtractor(csv_file_path)
        self.initialized = False
    
    def initialize(self) -> bool:
        """初始化系统"""
        if not self.initialized:
            success = self.extractor.load_kg_data()
            self.initialized = success
            return success
        return True
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果
        """
        if not self.initialized:
            logger.warning("系统未初始化，正在初始化...")
            if not self.initialize():
                return {'error': '系统初始化失败'}
        
        return self.extractor.process_query(query)
    
    def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            
        Returns:
            处理结果列表
        """
        if not self.initialized:
            if not self.initialize():
                return [{'error': '系统初始化失败'} for _ in queries]
        
        results = []
        for query in queries:
            result = self.extractor.process_query(query)
            results.append(result)
        
        return results

# 全局实例
_retrieval_system = None

def get_optimized_retrieval_system(csv_file_path: Optional[str] = None) -> OptimizedRetrievalSystem:
    """
    获取优化的检索系统单例
    
    Args:
        csv_file_path: 知识图谱CSV文件路径
        
    Returns:
        OptimizedRetrievalSystem实例
    """
    global _retrieval_system
    
    if _retrieval_system is None:
        _retrieval_system = OptimizedRetrievalSystem(csv_file_path)
    
    return _retrieval_system

def extract_entities_from_query(query: str, csv_file_path: Optional[str] = None) -> List[str]:
    """
    从查询中提取实体（简化接口）
    
    Args:
        query: 用户查询
        csv_file_path: 知识图谱CSV文件路径
        
    Returns:
        实体名称列表
    """
    system = get_optimized_retrieval_system(csv_file_path)
    result = system.process_query(query)
    return [entity['mention'] for entity in result.get('entities', [])]

def get_kg_relations_for_entities(entities: List[str], csv_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取实体的知识图谱关系（简化接口）
    
    Args:
        entities: 实体名称列表
        csv_file_path: 知识图谱CSV文件路径
        
    Returns:
        知识图谱关系结果
    """
    system = get_optimized_retrieval_system(csv_file_path)
    
    # 转换为实体字典格式
    entity_dicts = [{'mention': entity} for entity in entities]
    
    return system.extractor.query_kg_relations(entity_dicts)

# 测试函数
def test_optimized_extractor():
    """测试优化的实体提取器"""
    print("🧪 测试优化的实体提取器")
    print("=" * 50)
    
    # 创建检索系统
    system = OptimizedRetrievalSystem()
    
    # 初始化
    if not system.initialize():
        print("❌ 系统初始化失败")
        return
    
    # 测试查询
    test_queries = [
        "请推荐适合经常口臭的中药",
        "我感觉恶寒，但是一直没有出汗，该怎么办？",
        "建议中药方剂治疗手臂浮肿",
        "红花可以治疗乳癖吗？"
    ]
    
    print(f"\n🔍 测试实体提取和知识图谱查询:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 查询: {query}")
        
        result = system.process_query(query)
        
        print(f"   提取实体: {len(result['entities'])} 个")
        for entity in result['entities']:
            print(f"     - {entity['mention']}")
        
        # 显示知识图谱查询结果
        kg_results = result.get('kg_results', {})
        if kg_results:
            print(f"   知识图谱查询:")
            print(f"     匹配实体: {kg_results['matched_entities']}/{kg_results['total_entities']}")
            print(f"     总关系数: {kg_results['total_relations']}")
            print(f"     覆盖率: {kg_results['coverage_rate']:.1%}")
    
    # 显示统计信息
    stats = system.extractor.get_entity_statistics()
    print(f"\n📊 系统统计:")
    print(f"   总实体数: {stats.get('total_entities', 0)}")
    print(f"   总关系数: {stats.get('total_relations', 0)}")
    print(f"   有关系的实体数: {stats.get('entities_with_relations', 0)}")

if __name__ == "__main__":
    test_optimized_extractor()
