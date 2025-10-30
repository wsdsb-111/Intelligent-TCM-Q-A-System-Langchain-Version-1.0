"""
中医知识图谱核心数据模型定义
定义了中医领域知识图谱系统中使用的所有数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class EntityType(Enum):
    """中医实体类型枚举"""
    # 基础实体类型
    PERSON = "PERSON"  # 人物（医生、患者、历史名医等）
    ORGANIZATION = "ORGANIZATION"  # 组织机构（医院、医学院、药房等）
    GEO = "GEO"  # 地理位置（产地、医院位置等）
    EVENT = "EVENT"  # 事件（诊疗过程、学术会议等）
    CONCEPT = "CONCEPT"  # 概念（理论、学说等）
    OBJECT = "OBJECT"  # 物品（医疗器械、工具等）
    TIME = "TIME"  # 时间（朝代、时期等）
    
    # 中医特有实体类型
    SYMPTOM = "SYMPTOM"  # 症状
    DISEASE = "DISEASE"  # 疾病
    SYNDROME = "SYNDROME"  # 证候
    HERB = "HERB"  # 中药
    FORMULA = "FORMULA"  # 方剂
    ACUPOINT = "ACUPOINT"  # 穴位
    MERIDIAN = "MERIDIAN"  # 经络
    ORGAN = "ORGAN"  # 脏腑
    THEORY = "THEORY"  # 理论
    METHOD = "METHOD"  # 治法
    TECHNIQUE = "TECHNIQUE"  # 技术手法
    OTHER = "OTHER"  # 其他


class RelationshipType(Enum):
    """中医关系类型枚举"""
    # 英文关系类型（保持兼容性）
    RELATED_TO = "RELATED_TO"
    BELONGS_TO = "BELONGS_TO"
    LOCATED_IN = "LOCATED_IN"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    OWNS = "OWNS"
    WORKS_FOR = "WORKS_FOR"
    TREATS = "TREATS"
    DIAGNOSES = "DIAGNOSES"
    PRESCRIBES = "PRESCRIBES"
    CONTAINS = "CONTAINS"
    INTERACTS_WITH = "INTERACTS_WITH"
    INDICATES = "INDICATES"
    CONTRAINDICATES = "CONTRAINDICATES"
    SYMPTOM_OF = "SYMPTOM_OF"
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    
    # 中文关系类型
    相关 = "相关"
    归属 = "归属"
    位于 = "位于"
    参与 = "参与"
    拥有 = "拥有"
    工作于 = "工作于"
    治疗 = "治疗"
    诊断 = "诊断"
    开方 = "开方"
    包含 = "包含"
    相互作用 = "相互作用"
    适应症 = "适应症"
    禁忌症 = "禁忌症"
    症状 = "症状"
    引起 = "引起"
    预防 = "预防"
    配伍 = "配伍"
    相克 = "相克"
    相生 = "相生"
    归经 = "归经"
    功效 = "功效"
    主治 = "主治"
    用法 = "用法"
    用量 = "用量"
    炮制 = "炮制"
    产地 = "产地"
    性味 = "性味"
    毒性 = "毒性"
    配伍禁忌 = "配伍禁忌"
    煎服法 = "煎服法"
    注意事项 = "注意事项"


@dataclass
class ProcessedDocument:
    """处理后的文档数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    file_path: str = ""
    file_type: str = ""
    processed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.title and self.file_path:
            import os
            self.title = os.path.basename(self.file_path)


@dataclass
class Entity:
    """实体数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = EntityType.OTHER.value
    description: str = ""
    confidence: float = 0.0
    source_document_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """转换为Neo4j节点属性字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "confidence": self.confidence,
            "source_document_id": self.source_document_id,
            **self.properties
        }


@dataclass
class Relationship:
    """关系数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: str = RelationshipType.RELATED_TO.value
    description: str = ""
    weight: float = 1.0
    confidence: float = 0.0
    source_document_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """转换为Neo4j关系属性字典"""
        return {
            "id": self.id,
            "weight": self.weight,
            "description": self.description,
            "confidence": self.confidence,
            "source_document_id": self.source_document_id,
            **self.properties
        }


@dataclass
class GraphRAGResult:
    """GraphRAG处理结果"""
    document_id: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Neo4jData:
    """Neo4j导入数据"""
    entities_csv_path: str = ""
    relationships_csv_path: str = ""
    entity_count: int = 0
    relationship_count: int = 0
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingStatus:
    """处理状态"""
    document_id: str
    status: str  # pending, processing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0


@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告信息"""
        self.warnings.append(warning)


@dataclass
class ImportResult:
    """导入结果"""
    success: bool = False
    imported_entities: int = 0
    imported_relationships: int = 0
    failed_entities: int = 0
    failed_relationships: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def get_success_rate(self) -> float:
        """计算成功率"""
        total = self.imported_entities + self.imported_relationships + self.failed_entities + self.failed_relationships
        success = self.imported_entities + self.imported_relationships
        return (success / total * 100) if total > 0 else 0.0


@dataclass
class CSVExportResult:
    """CSV导出结果"""
    entities_csv_path: str = ""
    relationships_csv_path: str = ""
    import_guide_path: str = ""
    entity_count: int = 0
    relationship_count: int = 0
    export_time: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)