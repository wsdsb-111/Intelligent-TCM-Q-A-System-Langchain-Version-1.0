"""
中医知识图谱CSV导出器
将中医GraphRAG结果转换为CSV格式，便于手动导入Neo4j
"""

import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

from src.models import Entity, Relationship, GraphRAGResult


class CSVExporter:
    """中医知识图谱CSV导出器，负责将中医实体和关系数据转换为CSV格式"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化CSV导出器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.CSVExporter")
        
        # CSV配置
        self.encoding = 'utf-8'
        self.delimiter = ','
        self.quotechar = '"'
        self.quoting = csv.QUOTE_MINIMAL
    
    def export_entities(self, entities: List[Entity], filename_prefix: str) -> str:
        """
        导出实体到CSV文件
        
        Args:
            entities: 实体列表
            filename_prefix: 文件名前缀
            
        Returns:
            str: 生成的CSV文件路径
        """
        if not entities:
            self.logger.warning("实体列表为空，跳过导出")
            return ""
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename_prefix}_entities.csv"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding=self.encoding) as csvfile:
                fieldnames = [
                    'id', 'name', 'type', 'description', 
                    'confidence', 'source_document_id'
                ]
                
                writer = csv.DictWriter(
                    csvfile, 
                    fieldnames=fieldnames,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                    quoting=self.quoting
                )
                
                # 写入标题行
                writer.writeheader()
                
                # 写入实体数据
                for entity in entities:
                    row = {
                        'id': entity.id,
                        'name': self._escape_csv_value(entity.name),
                        'type': entity.type,
                        'description': self._escape_csv_value(entity.description),
                        'confidence': entity.confidence,
                        'source_document_id': entity.source_document_id
                    }
                    writer.writerow(row)
            
            self.logger.info(f"成功导出 {len(entities)} 个实体到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"导出实体CSV失败: {e}")
            raise
    
    def export_relationships(self, relationships: List[Relationship], 
                           entities: List[Entity], filename_prefix: str) -> str:
        """
        导出关系到CSV文件
        
        Args:
            relationships: 关系列表
            entities: 实体列表（用于查找实体名称）
            filename_prefix: 文件名前缀
            
        Returns:
            str: 生成的CSV文件路径
        """
        if not relationships:
            self.logger.warning("关系列表为空，跳过导出")
            return ""
        
        # 创建实体ID到名称的映射
        entity_id_to_name = {entity.id: entity.name for entity in entities}
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename_prefix}_relationships.csv"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding=self.encoding) as csvfile:
                fieldnames = [
                    'source_id', 'target_id', 'source_name', 'target_name',
                    'relationship_type', 'description', 'weight', 
                    'confidence', 'source_document_id'
                ]
                
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                    quoting=self.quoting
                )
                
                # 写入标题行
                writer.writeheader()
                
                # 写入关系数据
                for relationship in relationships:
                    source_name = entity_id_to_name.get(
                        relationship.source_entity_id, "未知实体"
                    )
                    target_name = entity_id_to_name.get(
                        relationship.target_entity_id, "未知实体"
                    )
                    
                    row = {
                        'source_id': relationship.source_entity_id,
                        'target_id': relationship.target_entity_id,
                        'source_name': self._escape_csv_value(source_name),
                        'target_name': self._escape_csv_value(target_name),
                        'relationship_type': relationship.relationship_type,
                        'description': self._escape_csv_value(relationship.description),
                        'weight': relationship.weight,
                        'confidence': relationship.confidence,
                        'source_document_id': relationship.source_document_id
                    }
                    writer.writerow(row)
            
            self.logger.info(f"成功导出 {len(relationships)} 个关系到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"导出关系CSV失败: {e}")
            raise
    
    def export_graphrag_result(self, result: GraphRAGResult, 
                             source_file: str) -> Tuple[str, str]:
        """
        导出GraphRAG结果到CSV文件
        
        Args:
            result: GraphRAG处理结果
            source_file: 源文件路径
            
        Returns:
            Tuple[str, str]: (实体CSV路径, 关系CSV路径)
        """
        # 从源文件路径生成前缀
        source_name = Path(source_file).stem
        
        # 导出实体
        entities_csv = ""
        if result.entities:
            entities_csv = self.export_entities(result.entities, source_name)
        
        # 导出关系
        relationships_csv = ""
        if result.relationships:
            relationships_csv = self.export_relationships(
                result.relationships, result.entities, source_name
            )
        
        return entities_csv, relationships_csv
    
    def _escape_csv_value(self, value: str) -> str:
        """
        转义CSV值中的特殊字符
        
        Args:
            value: 原始值
            
        Returns:
            str: 转义后的值
        """
        if not value:
            return ""
        
        # 移除换行符和制表符
        value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # 移除多余的空格
        value = ' '.join(value.split())
        
        return value
    
    def generate_neo4j_import_guide(self, entities_csv: str, 
                                  relationships_csv: str) -> str:
        """
        生成Neo4j导入指导文档
        
        Args:
            entities_csv: 实体CSV文件路径
            relationships_csv: 关系CSV文件路径
            
        Returns:
            str: 指导文档路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        guide_filename = f"{timestamp}_neo4j_import_guide.txt"
        guide_filepath = self.output_dir / guide_filename
        
        guide_content = f"""Neo4j CSV导入指导
==================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

文件信息:
- 实体文件: {entities_csv}
- 关系文件: {relationships_csv}

导入步骤:
1. 将CSV文件复制到Neo4j的import目录
2. 在Neo4j Browser中执行以下Cypher语句

实体导入:
---------
LOAD CSV WITH HEADERS FROM 'file:///{Path(entities_csv).name}' AS row
CREATE (e:Entity {{
  id: row.id,
  name: row.name,
  type: row.type,
  description: row.description,
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}})

关系导入:
---------
LOAD CSV WITH HEADERS FROM 'file:///{Path(relationships_csv).name}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
CALL apoc.create.relationship(source, row.relationship_type, {{
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}, target) YIELD rel
RETURN count(rel)

备选方案（如果没有APOC插件）:
LOAD CSV WITH HEADERS FROM 'file:///{Path(relationships_csv).name}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
CREATE (source)-[r:关系 {{
  type: row.relationship_type,
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}]->(target)

验证导入:
---------
// 检查实体数量
MATCH (e:Entity) RETURN count(e) as entity_count

// 检查关系数量
MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as relationship_count

// 查看实体类型分布
MATCH (e:Entity) RETURN e.type, count(e) as count ORDER BY count DESC

注意事项:
- 确保CSV文件在Neo4j的import目录中
- 如果遇到编码问题，请确认文件是UTF-8编码
- 建议在导入前备份数据库
"""
        
        try:
            with open(guide_filepath, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            self.logger.info(f"生成Neo4j导入指导: {guide_filepath}")
            return str(guide_filepath)
            
        except Exception as e:
            self.logger.error(f"生成导入指导失败: {e}")
            return ""
    
    def get_export_statistics(self, result: GraphRAGResult) -> Dict[str, Any]:
        """
        获取导出统计信息
        
        Args:
            result: GraphRAG处理结果
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "entity_count": len(result.entities),
            "relationship_count": len(result.relationships),
            "processing_time": result.processing_time,
            "entity_types": self._get_entity_type_distribution(result.entities),
            "relationship_types": self._get_relationship_type_distribution(result.relationships)
        }
    
    def _get_entity_type_distribution(self, entities: List[Entity]) -> Dict[str, int]:
        """获取实体类型分布"""
        distribution = {}
        for entity in entities:
            entity_type = entity.type
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution
    
    def _get_relationship_type_distribution(self, relationships: List[Relationship]) -> Dict[str, int]:
        """获取关系类型分布"""
        distribution = {}
        for relationship in relationships:
            rel_type = relationship.relationship_type
            distribution[rel_type] = distribution.get(rel_type, 0) + 1
        return distribution