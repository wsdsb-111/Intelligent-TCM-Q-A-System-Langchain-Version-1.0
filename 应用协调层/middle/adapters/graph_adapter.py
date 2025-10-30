"""
知识图谱检索适配器
封装Neo4j知识图谱，提供实体关系检索功能
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError

from middle.models.data_models import (
    EntityResult, RelationshipResult, PathResult, RetrievalResult,
    RetrievalSource, ModuleConfig
)
from middle.models.exceptions import (
    ModuleUnavailableError, DataSourceError, handle_module_error
)
from middle.utils.logging_utils import get_logger
from middle.utils.entity_extractor import get_entity_extractor


class GraphRetrievalAdapter:
    """知识图谱检索适配器"""
    
    def __init__(self, 
                 neo4j_uri: str = "neo4j://127.0.0.1:7687",
                 username: str = "neo4j", 
                 password: str = "hx1230047",
                 database: str = "neo4j",
                 dump_path: Optional[str] = None,
                 timeout: int = 20,
                 model_service = None,
                 use_llm_entity_extraction: bool = False):
        """
        初始化图检索适配器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            username: 用户名
            password: 密码
            database: 数据库名
            dump_path: dump文件路径（用于数据恢复）
            timeout: 查询超时时间（秒）
            model_service: 模型服务（用于LLM实体提取）
            use_llm_entity_extraction: 是否使用LLM提取实体
        """
        self.neo4j_uri = neo4j_uri
        self.username = username
        self.password = password
        self.database = database
        self.dump_path = dump_path
        self.timeout = timeout
        self.model_service = model_service
        self.use_llm_entity_extraction = use_llm_entity_extraction
        
        self.driver = None
        self.async_driver = None
        self.logger = get_logger(__name__)
        
        # 初始化实体提取器 - 使用知识图谱模式
        from middle.utils.entity_config import get_entity_config
        try:
            config = get_entity_config()
            csv_path = config.get_kg_csv_path()
            # 转换为绝对路径
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            csv_path = str(project_root / csv_path)
            
            self.entity_extractor = get_entity_extractor(
                model_service=model_service,
                use_llm=use_llm_entity_extraction,
                use_kg=True,  # 启用知识图谱模式
                csv_file_path=csv_path
            )
            self.logger.info("知识图谱适配器使用知识图谱模式实体提取器")
        except Exception as e:
            self.logger.warning(f"知识图谱模式实体提取器初始化失败: {e}，使用默认模式")
            self.entity_extractor = get_entity_extractor(
                model_service=model_service,
                use_llm=use_llm_entity_extraction
            )
        
        # 中医实体类型映射
        self.entity_types = {
            "HERB": "中药",
            "FORMULA": "方剂", 
            "SYMPTOM": "症状",
            "DISEASE": "疾病",
            "ACUPOINT": "穴位",
            "MERIDIAN": "经络",
            "ORGAN": "脏腑",
            "THEORY": "理论",
            "METHOD": "治法",
            "TECHNIQUE": "技术",
            "UNKNOWN": "中医实体"  # 处理UNKNOWN类型
        }
        
        # 关系类型映射
        self.relationship_types = {
            "治疗": "TREATS",
            "包含": "CONTAINS",
            "配伍": "COMBINES_WITH",
            "归经": "BELONGS_TO_MERIDIAN",
            "功效": "HAS_EFFECT",
            "主治": "MAIN_TREATMENT",
            "适应症": "INDICATION",
            "禁忌症": "CONTRAINDICATION",
            "相互作用": "INTERACTS_WITH",
            "相生": "GENERATES",
            "相克": "RESTRAINS"
        }
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            # 创建同步驱动
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.username, self.password)
            )
            
            # 创建异步驱动
            self.async_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.username, self.password)
            )
            
            # 测试连接
            self._test_connection()
            self.logger.info(f"成功连接到Neo4j数据库: {self.neo4j_uri}")
            
        except (ServiceUnavailable, AuthError, ConfigurationError) as e:
            error_msg = f"无法连接到Neo4j数据库: {str(e)}"
            self.logger.error(error_msg)
            raise ModuleUnavailableError("graph", error_msg)
    
    def _test_connection(self):
        """测试数据库连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
        except Exception as e:
            raise DataSourceError("neo4j", f"连接测试失败: {str(e)}")
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
        if self.async_driver:
            # 同步关闭异步驱动
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    asyncio.create_task(self.async_driver.close())
                else:
                    # 如果没有运行，直接运行
                    loop.run_until_complete(self.async_driver.close())
            except RuntimeError:
                # 没有事件循环，使用新的事件循环
                try:
                    asyncio.run(self.async_driver.close())
                except Exception as e:
                    # 忽略关闭错误
                    pass
    
    def __del__(self):
        """析构函数，确保连接关闭"""
        try:
            # 只关闭同步驱动，避免异步关闭错误
            if self.driver:
                self.driver.close()
            # 不关闭异步驱动，让垃圾回收器处理
        except Exception:
            pass
    
    # 基础查询方法
    
    def _run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行同步查询"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"查询执行失败: {query}, 错误: {str(e)}")
            raise DataSourceError("neo4j", f"查询执行失败: {str(e)}")
    
    async def _run_async_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行异步查询"""
        try:
            async with self.async_driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except Exception as e:
            self.logger.error(f"异步查询执行失败: {query}, 错误: {str(e)}")
            raise DataSourceError("neo4j", f"异步查询执行失败: {str(e)}")
    
    # 实体检索方法
    
    async def entity_search(self, query: str, entity_types: Optional[List[str]] = None, 
                          top_k: int = 10) -> List[EntityResult]:
        """
        实体搜索
        
        Args:
            query: 搜索查询
            entity_types: 限制的实体类型列表
            top_k: 返回结果数量
            
        Returns:
            实体检索结果列表
        """
        try:
            # 构建查询条件
            type_condition = ""
            if entity_types:
                type_conditions = [f"n.type = '{t}'" for t in entity_types]
                type_condition = f"AND ({' OR '.join(type_conditions)})"
            
            # 构建Cypher查询
            cypher_query = f"""
            MATCH (n)
            WHERE (n.name CONTAINS $query OR n.description CONTAINS $query)
            {type_condition}
            RETURN n.name as name, n.type as type, properties(n) as properties
            LIMIT $top_k
            """
            
            parameters = {"query": query, "top_k": top_k}
            results = await self._run_async_query(cypher_query, parameters)
            
            entity_results = []
            for record in results:
                # 获取实体的关系信息
                relationships = await self._get_entity_relationships(record["name"])
                
                entity_result = EntityResult(
                    entity_name=record["name"],
                    entity_type=record["type"],
                    properties=record["properties"],
                    relationships=relationships,
                    score=self._calculate_entity_score(query, record)
                )
                entity_results.append(entity_result)
            
            self.logger.info(f"实体搜索完成: 查询='{query}', 结果数={len(entity_results)}")
            return entity_results
            
        except Exception as e:
            error_msg = f"实体搜索失败: {str(e)}"
            self.logger.error(error_msg)
            raise handle_module_error("graph", e)
    
    async def _get_entity_relationships(self, entity_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取实体的关系信息（增加数量以充分利用关系数据）"""
        try:
            cypher_query = """
            MATCH (n {name: $entity_name})-[r]-(m)
            RETURN type(r) as relationship_type, m.name as related_entity, m.type as related_type
            LIMIT $limit
            """
            
            parameters = {"entity_name": entity_name, "limit": limit}
            results = await self._run_async_query(cypher_query, parameters)
            
            return [
                {
                    "relationship_type": record["relationship_type"],
                    "related_entity": record["related_entity"],
                    "related_type": record["related_type"]
                }
                for record in results
            ]
        except Exception as e:
            self.logger.warning(f"获取实体关系失败: {entity_name}, 错误: {str(e)}")
            return []
    
    def _calculate_entity_score(self, query: str, record: Dict[str, Any]) -> float:
        """计算实体相关性评分"""
        score = 0.0
        query_lower = query.lower()
        
        # 名称匹配评分
        name = record.get("name", "").lower()
        if query_lower == name:
            score += 1.0
        elif query_lower in name:
            score += 0.8
        elif any(word in name for word in query_lower.split()):
            score += 0.6
        
        # 描述匹配评分
        properties = record.get("properties", {})
        description = properties.get("description", "").lower()
        if query_lower in description:
            score += 0.4
        
        # 实体类型权重
        entity_type = record.get("type", "")
        if entity_type in ["HERB", "FORMULA", "DISEASE"]:
            score += 0.2
        
        return min(score, 1.0)
    
    # 关系检索方法
    
    async def relationship_search(self, entity1: str, entity2: str, 
                                relationship_types: Optional[List[str]] = None) -> List[RelationshipResult]:
        """
        关系搜索
        
        Args:
            entity1: 源实体
            entity2: 目标实体
            relationship_types: 限制的关系类型列表
            
        Returns:
            关系检索结果列表
        """
        try:
            # 构建关系类型条件
            type_condition = ""
            if relationship_types:
                type_conditions = [f"type(r) = '{t}'" for t in relationship_types]
                type_condition = f"AND ({' OR '.join(type_conditions)})"
            
            # 双向查询
            cypher_query = f"""
            MATCH (a)-[r]-(b)
            WHERE (a.name = $entity1 AND b.name = $entity2) 
               OR (a.name = $entity2 AND b.name = $entity1)
            {type_condition}
            RETURN a.name as source, b.name as target, type(r) as relationship_type, 
                   properties(r) as properties
            """
            
            parameters = {"entity1": entity1, "entity2": entity2}
            results = await self._run_async_query(cypher_query, parameters)
            
            relationship_results = []
            for record in results:
                relationship_result = RelationshipResult(
                    source_entity=record["source"],
                    target_entity=record["target"],
                    relationship_type=record["relationship_type"],
                    properties=record["properties"],
                    score=1.0  # 直接关系给满分
                )
                relationship_results.append(relationship_result)
            
            self.logger.info(f"关系搜索完成: {entity1} <-> {entity2}, 结果数={len(relationship_results)}")
            return relationship_results
            
        except Exception as e:
            error_msg = f"关系搜索失败: {str(e)}"
            self.logger.error(error_msg)
            raise handle_module_error("graph", e)
    
    # 路径检索方法
    
    async def path_search(self, start_entity: str, end_entity: str, 
                         max_depth: int = 3, limit: int = 5) -> List[PathResult]:
        """
        路径搜索
        
        Args:
            start_entity: 起始实体
            end_entity: 结束实体
            max_depth: 最大路径深度
            limit: 返回路径数量限制
            
        Returns:
            路径检索结果列表
        """
        try:
            cypher_query = f"""
            MATCH path = (start {{name: $start_entity}})-[*1..{max_depth}]-(end {{name: $end_entity}})
            WITH path, length(path) as path_length
            ORDER BY path_length
            LIMIT $limit
            RETURN path, path_length
            """
            
            parameters = {
                "start_entity": start_entity,
                "end_entity": end_entity,
                "limit": limit
            }
            results = await self._run_async_query(cypher_query, parameters)
            
            path_results = []
            for record in results:
                path_data = record["path"]
                path_length = record["path_length"]
                
                # 解析路径
                path_steps = self._parse_path(path_data)
                
                # 计算路径评分（路径越短评分越高）
                score = 1.0 / (path_length + 1)
                
                path_result = PathResult(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    path=path_steps,
                    path_length=path_length,
                    score=score
                )
                path_results.append(path_result)
            
            self.logger.info(f"路径搜索完成: {start_entity} -> {end_entity}, 结果数={len(path_results)}")
            return path_results
            
        except Exception as e:
            error_msg = f"路径搜索失败: {str(e)}"
            self.logger.error(error_msg)
            raise handle_module_error("graph", e)
    
    def _parse_path(self, path_data: Any) -> List[Dict[str, Any]]:
        """解析Neo4j路径数据"""
        try:
            path_steps = []
            nodes = path_data.nodes
            relationships = path_data.relationships
            
            for i, rel in enumerate(relationships):
                source_node = nodes[i]
                target_node = nodes[i + 1]
                
                step = {
                    "source": source_node.get("name", ""),
                    "target": target_node.get("name", ""),
                    "relationship": rel.type,
                    "properties": dict(rel)
                }
                path_steps.append(step)
            
            return path_steps
        except Exception as e:
            self.logger.warning(f"路径解析失败: {str(e)}")
            return []
    
    # 高级检索功能
    
    async def semantic_entity_recognition(self, text: str) -> List[Dict[str, Any]]:
        """
        语义实体识别
        从文本中识别中医相关实体
        
        Args:
            text: 输入文本
            
        Returns:
            识别到的实体列表
        """
        try:
            entities = []
            
            # 基于图数据库的实体识别
            # 查找文本中包含的已知实体
            cypher_query = """
            MATCH (n)
            WHERE n.name IS NOT NULL 
            AND $text CONTAINS n.name
            AND size(n.name) >= 2
            RETURN n.name as entity_name, n.type as entity_type, 
                   properties(n) as properties,
                   size(n.name) as name_length
            ORDER BY name_length DESC
            LIMIT 20
            """
            
            parameters = {"text": text}
            results = await self._run_async_query(cypher_query, parameters)
            
            # 去重和评分
            found_entities = set()
            for record in results:
                entity_name = record["entity_name"]
                if entity_name not in found_entities:
                    # 计算实体在文本中的重要性
                    importance_score = self._calculate_entity_importance(text, entity_name, record)
                    
                    entities.append({
                        "name": entity_name,
                        "type": record["entity_type"],
                        "properties": record["properties"],
                        "importance_score": importance_score,
                        "positions": self._find_entity_positions(text, entity_name)
                    })
                    found_entities.add(entity_name)
            
            # 按重要性排序
            entities.sort(key=lambda x: x["importance_score"], reverse=True)
            
            self.logger.info(f"实体识别完成: 文本长度={len(text)}, 识别实体数={len(entities)}")
            return entities
            
        except Exception as e:
            self.logger.error(f"实体识别失败: {str(e)}")
            return []
    
    def _calculate_entity_importance(self, text: str, entity_name: str, record: Dict[str, Any]) -> float:
        """计算实体在文本中的重要性"""
        score = 0.0
        
        # 基础评分：实体名称长度
        score += len(entity_name) * 0.1
        
        # 出现频率评分
        count = text.count(entity_name)
        score += count * 0.3
        
        # 实体类型权重
        entity_type = record.get("entity_type", "")
        type_weights = {
            "DISEASE": 1.0,
            "SYMPTOM": 0.9,
            "HERB": 0.8,
            "FORMULA": 0.8,
            "ACUPOINT": 0.7,
            "METHOD": 0.6
        }
        score += type_weights.get(entity_type, 0.5)
        
        # 位置权重（开头的实体更重要）
        first_pos = text.find(entity_name)
        if first_pos >= 0:
            position_weight = max(0.1, 1.0 - (first_pos / len(text)))
            score += position_weight * 0.2
        
        return score
    
    def _find_entity_positions(self, text: str, entity_name: str) -> List[int]:
        """查找实体在文本中的所有位置"""
        positions = []
        start = 0
        while True:
            pos = text.find(entity_name, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    async def relationship_inference(self, entities: List[str], max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        关系推理
        基于实体列表推理可能的关系
        
        Args:
            entities: 实体列表
            max_depth: 最大推理深度
            
        Returns:
            推理出的关系列表
        """
        try:
            inferred_relations = []
            
            # 两两实体间的直接关系
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    direct_relations = await self.relationship_search(entity1, entity2)
                    for rel in direct_relations:
                        inferred_relations.append({
                            "type": "direct",
                            "source": rel.source_entity,
                            "target": rel.target_entity,
                            "relationship": rel.relationship_type,
                            "confidence": 1.0,
                            "evidence": "直接关系"
                        })
            
            # 间接关系推理
            if max_depth > 1:
                indirect_relations = await self._infer_indirect_relationships(entities, max_depth)
                inferred_relations.extend(indirect_relations)
            
            # 基于共同邻居的关系推理
            common_neighbor_relations = await self._infer_common_neighbor_relationships(entities)
            inferred_relations.extend(common_neighbor_relations)
            
            # 按置信度排序
            inferred_relations.sort(key=lambda x: x["confidence"], reverse=True)
            
            self.logger.info(f"关系推理完成: 输入实体数={len(entities)}, 推理关系数={len(inferred_relations)}")
            return inferred_relations
            
        except Exception as e:
            self.logger.error(f"关系推理失败: {str(e)}")
            return []
    
    async def _infer_indirect_relationships(self, entities: List[str], max_depth: int) -> List[Dict[str, Any]]:
        """推理间接关系"""
        indirect_relations = []
        
        try:
            for entity1 in entities:
                for entity2 in entities:
                    if entity1 != entity2:
                        # 查找间接路径
                        paths = await self.path_search(entity1, entity2, max_depth=max_depth, limit=3)
                        
                        for path in paths:
                            if path.path_length > 1:  # 只考虑间接路径
                                # 分析路径中的关系模式
                                relation_pattern = " -> ".join([step["relationship"] for step in path.path])
                                confidence = 0.8 / path.path_length  # 路径越长置信度越低
                                
                                indirect_relations.append({
                                    "type": "indirect",
                                    "source": entity1,
                                    "target": entity2,
                                    "relationship": f"间接关系({relation_pattern})",
                                    "confidence": confidence,
                                    "evidence": f"通过{path.path_length}步路径连接",
                                    "path": path.path
                                })
        
        except Exception as e:
            self.logger.warning(f"间接关系推理失败: {str(e)}")
        
        return indirect_relations
    
    async def _infer_common_neighbor_relationships(self, entities: List[str]) -> List[Dict[str, Any]]:
        """基于共同邻居推理关系"""
        common_relations = []
        
        try:
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # 查找共同邻居
                    cypher_query = """
                    MATCH (e1 {name: $entity1})-[r1]-(common)-[r2]-(e2 {name: $entity2})
                    WHERE e1 <> e2 AND e1 <> common AND e2 <> common
                    RETURN common.name as common_neighbor, 
                           type(r1) as rel1_type, type(r2) as rel2_type,
                           common.type as common_type
                    LIMIT 5
                    """
                    
                    parameters = {"entity1": entity1, "entity2": entity2}
                    results = await self._run_async_query(cypher_query, parameters)
                    
                    if results:
                        # 基于共同邻居数量计算置信度
                        confidence = min(0.7, len(results) * 0.2)
                        
                        common_neighbors = [r["common_neighbor"] for r in results]
                        evidence = f"共同邻居: {', '.join(common_neighbors[:3])}"
                        
                        common_relations.append({
                            "type": "common_neighbor",
                            "source": entity1,
                            "target": entity2,
                            "relationship": "潜在关联",
                            "confidence": confidence,
                            "evidence": evidence,
                            "common_neighbors": common_neighbors
                        })
        
        except Exception as e:
            self.logger.warning(f"共同邻居关系推理失败: {str(e)}")
        
        return common_relations
    
    async def advanced_path_analysis(self, start_entity: str, end_entity: str, 
                                   analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        高级路径分析
        
        Args:
            start_entity: 起始实体
            end_entity: 结束实体
            analysis_type: 分析类型 (shortest|comprehensive|semantic)
            
        Returns:
            路径分析结果
        """
        try:
            analysis_result = {
                "start_entity": start_entity,
                "end_entity": end_entity,
                "analysis_type": analysis_type,
                "paths": [],
                "statistics": {},
                "insights": []
            }
            
            if analysis_type == "shortest":
                # 最短路径分析
                paths = await self.path_search(start_entity, end_entity, max_depth=3, limit=1)
                analysis_result["paths"] = paths
                
                if paths:
                    shortest_path = paths[0]
                    analysis_result["statistics"]["shortest_distance"] = shortest_path.path_length
                    analysis_result["insights"].append(f"最短路径长度为{shortest_path.path_length}")
                
            elif analysis_type == "comprehensive":
                # 综合路径分析
                paths = await self.path_search(start_entity, end_entity, max_depth=4, limit=10)
                analysis_result["paths"] = paths
                
                if paths:
                    # 统计分析
                    path_lengths = [p.path_length for p in paths]
                    analysis_result["statistics"] = {
                        "total_paths": len(paths),
                        "min_length": min(path_lengths),
                        "max_length": max(path_lengths),
                        "avg_length": sum(path_lengths) / len(path_lengths)
                    }
                    
                    # 关系类型分析
                    relation_counts = {}
                    for path in paths:
                        for step in path.path:
                            rel_type = step["relationship"]
                            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
                    
                    analysis_result["statistics"]["relation_distribution"] = relation_counts
                    
                    # 生成洞察
                    most_common_rel = max(relation_counts.items(), key=lambda x: x[1])
                    analysis_result["insights"].append(f"最常见的关系类型是'{most_common_rel[0]}'")
                
            elif analysis_type == "semantic":
                # 语义路径分析
                paths = await self.path_search(start_entity, end_entity, max_depth=3, limit=5)
                analysis_result["paths"] = paths
                
                # 语义相关性分析
                semantic_scores = []
                for path in paths:
                    semantic_score = await self._calculate_path_semantic_score(path)
                    semantic_scores.append(semantic_score)
                
                analysis_result["statistics"]["semantic_scores"] = semantic_scores
                
                if semantic_scores:
                    best_semantic_path_idx = semantic_scores.index(max(semantic_scores))
                    analysis_result["insights"].append(f"语义最相关的是路径{best_semantic_path_idx + 1}")
            
            self.logger.info(f"高级路径分析完成: {start_entity} -> {end_entity}, 类型={analysis_type}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"高级路径分析失败: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_path_semantic_score(self, path: PathResult) -> float:
        """计算路径的语义相关性评分"""
        try:
            score = 0.0
            
            # 基于路径长度的基础评分
            score += 1.0 / (path.path_length + 1)
            
            # 基于关系类型的语义评分
            semantic_weights = {
                "治疗": 1.0,
                "主治": 0.9,
                "功效": 0.8,
                "包含": 0.7,
                "配伍": 0.6,
                "归经": 0.5
            }
            
            for step in path.path:
                rel_type = step["relationship"]
                weight = semantic_weights.get(rel_type, 0.3)
                score += weight * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"路径语义评分计算失败: {str(e)}")
            return 0.0
    
    async def complex_query_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        复杂查询搜索，结合实体识别和关系推理
        
        Args:
            query: 复杂查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            # 实体识别
            entities = await self._extract_entities_from_query(query)
            
            # 如果识别到实体，进行基于实体的搜索
            if entities:
                results = []
                for entity in entities:
                    entity_results = await self.entity_search(entity, top_k=5)
                    for entity_result in entity_results:
                        retrieval_result = self._entity_to_retrieval_result(entity_result, query)
                        results.append(retrieval_result)
                
                # 按评分排序并限制数量
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:top_k]
            else:
                # 如果没有识别到实体，进行全文搜索
                return await self._full_text_search(query, top_k)
                
        except Exception as e:
            error_msg = f"复杂查询搜索失败: {str(e)}"
            self.logger.error(error_msg)
            raise handle_module_error("graph", e)
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取实体（改进版：更智能的实体识别）"""
        try:
            all_entities = []
            
            # 方法1：使用规则提取器
            try:
                candidate_entities = self.entity_extractor.extract(query)
                if candidate_entities:
                    self.logger.debug(f"规则提取器找到候选实体: {candidate_entities}")
                    all_entities.extend(candidate_entities)
            except Exception as e:
                self.logger.debug(f"规则提取器失败: {e}")
            
            # 方法2：使用语义实体识别（基于图数据库）
            try:
                semantic_entities = await self.semantic_entity_recognition(query)
                if semantic_entities:
                    # 提取实体名称
                    entity_names = [e["name"] for e in semantic_entities[:5]]  # 取前5个
                    self.logger.debug(f"语义识别找到实体: {entity_names}")
                    all_entities.extend(entity_names)
            except Exception as e:
                self.logger.debug(f"语义识别失败: {e}")
            
            # 方法3：简单字符串匹配（备用）
            if not all_entities:
                self.logger.debug("尝试简单字符串匹配...")
                # 提取查询中的关键词（2-4个字符）
                import re
                # 提取中文词语
                words = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
                
                for word in words[:3]:  # 只测试前3个词
                    cypher_query = """
                    MATCH (n)
                    WHERE n.name CONTAINS $word
                    RETURN DISTINCT n.name as entity_name
                    ORDER BY size(n.name)
                    LIMIT 3
                    """
                    parameters = {"word": word}
                    results = await self._run_async_query(cypher_query, parameters)
                    
                    if results:
                        entity_names = [record["entity_name"] for record in results]
                        self.logger.debug(f"字符串匹配'{word}'找到: {entity_names}")
                        all_entities.extend(entity_names)
            
            # 去重
            unique_entities = list(dict.fromkeys(all_entities))
            
            self.logger.info(f"从查询'{query}'中提取到实体: {unique_entities}")
            return unique_entities
            
        except Exception as e:
            self.logger.warning(f"实体提取失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
    
    async def _full_text_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """全文搜索（改进版：多策略搜索）"""
        try:
            retrieval_results = []
            
            # 策略1：完整查询匹配
            cypher_query = """
            MATCH (n)
            WHERE n.name CONTAINS $query 
               OR n.description CONTAINS $query
            RETURN n.name as name, n.type as type, properties(n) as properties
            LIMIT $top_k
            """
            parameters = {"query": query, "top_k": top_k}
            results = await self._run_async_query(cypher_query, parameters)
            
            for record in results:
                content = self._build_entity_content(record)
                score = self._calculate_entity_score(query, record)
                
                retrieval_result = RetrievalResult(
                    content=content,
                    score=score,
                    source=RetrievalSource.GRAPH,
                    metadata={
                        "entity_name": record["name"],
                        "entity_type": record["type"],
                        "properties": record["properties"]
                    },
                    entities=[record["name"]]
                )
                retrieval_results.append(retrieval_result)
            
            # 策略2：如果完整查询没结果，尝试关键词搜索
            if not retrieval_results:
                self.logger.debug("完整查询无结果，尝试关键词搜索...")
                import re
                # 提取2-4字的中文词
                keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', query)
                
                for keyword in keywords[:3]:  # 最多3个关键词
                    cypher_query = """
                    MATCH (n)
                    WHERE n.name CONTAINS $keyword 
                       OR n.description CONTAINS $keyword
                    RETURN n.name as name, n.type as type, properties(n) as properties
                    LIMIT 5
                    """
                    parameters = {"keyword": keyword}
                    results = await self._run_async_query(cypher_query, parameters)
                    
                    for record in results:
                        content = self._build_entity_content(record)
                        score = self._calculate_entity_score(keyword, record) * 0.8  # 关键词搜索降低评分
                        
                        retrieval_result = RetrievalResult(
                            content=content,
                            score=score,
                            source=RetrievalSource.GRAPH,
                            metadata={
                                "entity_name": record["name"],
                                "entity_type": record["type"],
                                "properties": record["properties"],
                                "matched_keyword": keyword
                            },
                            entities=[record["name"]]
                        )
                        retrieval_results.append(retrieval_result)
                
                # 去重（基于实体名称）
                seen_entities = set()
                unique_results = []
                for result in retrieval_results:
                    entity_name = result.metadata.get("entity_name")
                    if entity_name not in seen_entities:
                        seen_entities.add(entity_name)
                        unique_results.append(result)
                retrieval_results = unique_results
            
            # 按评分排序并限制数量
            retrieval_results.sort(key=lambda x: x.score, reverse=True)
            return retrieval_results[:top_k]
            
        except Exception as e:
            self.logger.warning(f"全文搜索失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
    
    # 知识图谱推理功能
    
    async def knowledge_inference(self, query_entities: List[str], 
                                inference_type: str = "treatment") -> List[Dict[str, Any]]:
        """
        知识推理
        基于输入实体进行知识推理
        
        Args:
            query_entities: 查询实体列表
            inference_type: 推理类型 (treatment|diagnosis|compatibility)
            
        Returns:
            推理结果列表
        """
        try:
            inference_results = []
            
            if inference_type == "treatment":
                # 治疗推理：根据症状/疾病推理治疗方案
                inference_results = await self._treatment_inference(query_entities)
                
            elif inference_type == "diagnosis":
                # 诊断推理：根据症状推理可能疾病
                inference_results = await self._diagnosis_inference(query_entities)
                
            elif inference_type == "compatibility":
                # 配伍推理：分析药物配伍关系
                inference_results = await self._compatibility_inference(query_entities)
            
            self.logger.info(f"知识推理完成: 类型={inference_type}, 输入实体数={len(query_entities)}, 结果数={len(inference_results)}")
            return inference_results
            
        except Exception as e:
            self.logger.error(f"知识推理失败: {str(e)}")
            return []
    
    async def _treatment_inference(self, entities: List[str]) -> List[Dict[str, Any]]:
        """治疗推理"""
        treatment_results = []
        
        try:
            for entity in entities:
                # 查找治疗该疾病/症状的方剂和中药
                cypher_query = """
                MATCH (symptom {name: $entity})<-[:治疗|:主治]-(treatment)
                WHERE treatment.type IN ['HERB', 'FORMULA']
                RETURN treatment.name as treatment_name, 
                       treatment.type as treatment_type,
                       properties(treatment) as properties
                LIMIT 10
                """
                
                parameters = {"entity": entity}
                results = await self._run_async_query(cypher_query, parameters)
                
                for record in results:
                    confidence = 0.8  # 直接治疗关系的置信度较高
                    
                    treatment_results.append({
                        "query_entity": entity,
                        "treatment": record["treatment_name"],
                        "treatment_type": record["treatment_type"],
                        "confidence": confidence,
                        "reasoning": f"直接治疗关系",
                        "properties": record["properties"]
                    })
                
                # 间接治疗推理（通过功效）
                indirect_treatments = await self._indirect_treatment_inference(entity)
                treatment_results.extend(indirect_treatments)
        
        except Exception as e:
            self.logger.warning(f"治疗推理失败: {str(e)}")
        
        return treatment_results
    
    async def _indirect_treatment_inference(self, entity: str) -> List[Dict[str, Any]]:
        """间接治疗推理"""
        indirect_results = []
        
        try:
            # 通过功效进行间接推理
            cypher_query = """
            MATCH (symptom {name: $entity})-[:需要功效]-(effect)-[:具有功效]-(treatment)
            WHERE treatment.type IN ['HERB', 'FORMULA']
            RETURN treatment.name as treatment_name,
                   treatment.type as treatment_type,
                   effect.name as effect_name,
                   properties(treatment) as properties
            LIMIT 5
            """
            
            parameters = {"entity": entity}
            results = await self._run_async_query(cypher_query, parameters)
            
            for record in results:
                confidence = 0.6  # 间接关系置信度较低
                
                indirect_results.append({
                    "query_entity": entity,
                    "treatment": record["treatment_name"],
                    "treatment_type": record["treatment_type"],
                    "confidence": confidence,
                    "reasoning": f"通过功效'{record['effect_name']}'间接推理",
                    "properties": record["properties"]
                })
        
        except Exception as e:
            self.logger.warning(f"间接治疗推理失败: {str(e)}")
        
        return indirect_results
    
    async def _diagnosis_inference(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """诊断推理"""
        diagnosis_results = []
        
        try:
            # 基于症状组合推理疾病
            symptoms_str = "', '".join(symptoms)
            
            cypher_query = f"""
            MATCH (disease)-[:表现为|:症状包括]->(symptom)
            WHERE symptom.name IN ['{symptoms_str}']
            WITH disease, count(symptom) as matching_symptoms
            WHERE matching_symptoms >= 1
            RETURN disease.name as disease_name,
                   disease.type as disease_type,
                   matching_symptoms,
                   properties(disease) as properties
            ORDER BY matching_symptoms DESC
            LIMIT 10
            """
            
            results = await self._run_async_query(cypher_query, {})
            
            for record in results:
                # 基于匹配症状数量计算置信度
                matching_count = record["matching_symptoms"]
                confidence = min(0.9, matching_count / len(symptoms) * 0.8 + 0.2)
                
                diagnosis_results.append({
                    "symptoms": symptoms,
                    "disease": record["disease_name"],
                    "disease_type": record["disease_type"],
                    "confidence": confidence,
                    "matching_symptoms": matching_count,
                    "reasoning": f"匹配{matching_count}个症状",
                    "properties": record["properties"]
                })
        
        except Exception as e:
            self.logger.warning(f"诊断推理失败: {str(e)}")
        
        return diagnosis_results
    
    async def _compatibility_inference(self, herbs: List[str]) -> List[Dict[str, Any]]:
        """配伍推理"""
        compatibility_results = []
        
        try:
            # 分析药物间的配伍关系
            for i, herb1 in enumerate(herbs):
                for herb2 in herbs[i+1:]:
                    # 查找配伍关系
                    cypher_query = """
                    MATCH (h1 {name: $herb1})-[r]-(h2 {name: $herb2})
                    WHERE type(r) IN ['配伍', '相生', '相克', '相互作用']
                    RETURN type(r) as relationship_type,
                           properties(r) as rel_properties
                    """
                    
                    parameters = {"herb1": herb1, "herb2": herb2}
                    results = await self._run_async_query(cypher_query, parameters)
                    
                    for record in results:
                        rel_type = record["relationship_type"]
                        
                        # 根据关系类型确定配伍评级
                        compatibility_score = {
                            "配伍": 0.8,
                            "相生": 0.9,
                            "相克": 0.2,
                            "相互作用": 0.5
                        }.get(rel_type, 0.5)
                        
                        compatibility_results.append({
                            "herb1": herb1,
                            "herb2": herb2,
                            "relationship": rel_type,
                            "compatibility_score": compatibility_score,
                            "reasoning": f"存在{rel_type}关系",
                            "properties": record["rel_properties"]
                        })
        
        except Exception as e:
            self.logger.warning(f"配伍推理失败: {str(e)}")
        
        return compatibility_results
    
    # 多跳查询功能
    
    async def multi_hop_query(self, start_entities: List[str], 
                            target_types: List[str], 
                            max_hops: int = 3) -> List[Dict[str, Any]]:
        """
        多跳查询
        从起始实体出发，通过多跳关系查找目标类型的实体
        
        Args:
            start_entities: 起始实体列表
            target_types: 目标实体类型列表
            max_hops: 最大跳数
            
        Returns:
            多跳查询结果
        """
        try:
            multi_hop_results = []
            
            for start_entity in start_entities:
                for target_type in target_types:
                    # 构建多跳查询
                    cypher_query = f"""
                    MATCH path = (start {{name: $start_entity}})-[*1..{max_hops}]-(target)
                    WHERE target.type = $target_type
                    WITH path, length(path) as hop_count
                    ORDER BY hop_count
                    LIMIT 10
                    RETURN target.name as target_name,
                           target.type as target_type,
                           hop_count,
                           path,
                           properties(target) as properties
                    """
                    
                    parameters = {
                        "start_entity": start_entity,
                        "target_type": target_type
                    }
                    results = await self._run_async_query(cypher_query, parameters)
                    
                    for record in results:
                        # 计算多跳评分（跳数越少评分越高）
                        hop_count = record["hop_count"]
                        score = 1.0 / (hop_count + 1)
                        
                        # 解析路径
                        path_steps = self._parse_path(record["path"])
                        
                        multi_hop_results.append({
                            "start_entity": start_entity,
                            "target_entity": record["target_name"],
                            "target_type": record["target_type"],
                            "hop_count": hop_count,
                            "score": score,
                            "path": path_steps,
                            "properties": record["properties"]
                        })
            
            # 按评分排序
            multi_hop_results.sort(key=lambda x: x["score"], reverse=True)
            
            self.logger.info(f"多跳查询完成: 起始实体数={len(start_entities)}, 目标类型数={len(target_types)}, 结果数={len(multi_hop_results)}")
            return multi_hop_results
            
        except Exception as e:
            self.logger.error(f"多跳查询失败: {str(e)}")
            return []
    
    async def contextual_search(self, query: str, context_entities: List[str], 
                              top_k: int = 10) -> List[RetrievalResult]:
        """
        上下文搜索
        基于上下文实体进行搜索，提高搜索的相关性
        
        Args:
            query: 搜索查询
            context_entities: 上下文实体列表
            top_k: 返回结果数量
            
        Returns:
            上下文相关的检索结果
        """
        try:
            contextual_results = []
            
            # 首先进行常规搜索
            regular_results = await self.complex_query_search(query, top_k * 2)
            
            # 基于上下文实体重新评分
            for result in regular_results:
                context_score = await self._calculate_context_relevance(result, context_entities)
                
                # 调整原始评分
                adjusted_score = result.score * 0.7 + context_score * 0.3
                
                # 更新结果
                result.score = adjusted_score
                result.metadata["context_score"] = context_score
                result.metadata["context_entities"] = context_entities
                
                contextual_results.append(result)
            
            # 按调整后的评分排序
            contextual_results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"上下文搜索完成: 查询='{query}', 上下文实体数={len(context_entities)}, 结果数={len(contextual_results[:top_k])}")
            return contextual_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"上下文搜索失败: {str(e)}")
            return []
    
    async def _calculate_context_relevance(self, result: RetrievalResult, 
                                         context_entities: List[str]) -> float:
        """计算结果与上下文的相关性"""
        try:
            relevance_score = 0.0
            
            # 获取结果中的实体
            result_entities = result.entities or []
            if result.metadata.get("entity_name"):
                result_entities.append(result.metadata["entity_name"])
            
            # 计算与上下文实体的关联度
            for result_entity in result_entities:
                for context_entity in context_entities:
                    if result_entity == context_entity:
                        relevance_score += 1.0
                    else:
                        # 查找间接关联
                        relations = await self.relationship_search(result_entity, context_entity)
                        if relations:
                            relevance_score += 0.5
                        else:
                            # 查找路径关联
                            paths = await self.path_search(result_entity, context_entity, max_depth=2, limit=1)
                            if paths:
                                relevance_score += 0.3 / paths[0].path_length
            
            # 归一化评分
            max_possible_score = len(result_entities) * len(context_entities)
            if max_possible_score > 0:
                relevance_score = relevance_score / max_possible_score
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"上下文相关性计算失败: {str(e)}")
            return 0.0
    
    def _entity_to_retrieval_result(self, entity_result: EntityResult, query: str) -> RetrievalResult:
        """将实体结果转换为检索结果（优化版：更丰富的关系信息）"""
        content = self._build_entity_content({
            "name": entity_result.entity_name,
            "type": entity_result.entity_type,
            "properties": entity_result.properties
        })
        
        # 添加关系信息（大幅增加数量，充分利用丰富的关系数据）
        if entity_result.relationships:
            content += "\n\n相关关系："
            
            # 按关系类型分组
            rel_groups = {}
            for rel in entity_result.relationships[:30]:  # 增加到30个关系
                rel_type = rel.get('relationship_type', '其他')
                rel_entity = rel.get('related_entity', '')
                
                if rel_type not in rel_groups:
                    rel_groups[rel_type] = []
                rel_groups[rel_type].append(rel_entity)
            
            # 格式化输出（按类型分组更清晰）
            for rel_type, entities in rel_groups.items():
                if len(entities) == 1:
                    content += f"\n- {rel_type}：{entities[0]}"
                else:
                    # 多个实体，列出前8个
                    entities_str = "、".join(entities[:8])
                    if len(entities) > 8:
                        entities_str += f" 等{len(entities)}个"
                    content += f"\n- {rel_type}：{entities_str}"
        
        # 添加查询相关性信息
        if query:
            content += f"\n\n查询相关性：与'{query}'相关的中医实体"
        
        return RetrievalResult(
            content=content,
            score=entity_result.score,
            source=RetrievalSource.GRAPH,
            metadata={
                "entity_name": entity_result.entity_name,
                "entity_type": entity_result.entity_type,
                "properties": entity_result.properties,
                "relationships": entity_result.relationships,
                "query": query,
                "content_length": len(content)
            },
            entities=[entity_result.entity_name],
            relationships=[rel.get('relationship_type', '') for rel in entity_result.relationships]
        )
    
    def _build_entity_content(self, record: Dict[str, Any]) -> str:
        """构建实体内容描述（优化版：更丰富的内容构建）"""
        name = record.get("name", "")
        entity_type = record.get("type", "")
        properties = record.get("properties", {})
        
        # 获取中文类型名称，如果类型未知则尝试智能推断
        if entity_type == "UNKNOWN" or entity_type not in self.entity_types:
            type_name = self._infer_entity_type(name, properties)
        else:
            type_name = self.entity_types.get(entity_type, entity_type)
        
        # 构建详细内容
        content_parts = []
        content_parts.append(f"【{type_name}】{name}")
        
        # 核心信息（按优先级和重要性排序）
        info_fields = [
            ("description", "描述", 1.0),
            ("effect", "功效", 0.9),
            ("indication", "主治", 0.9),
            ("usage", "用法", 0.8),
            ("dosage", "用量", 0.8),
            ("properties", "性味", 0.7),
            ("meridian", "归经", 0.7),
            ("category", "分类", 0.6),
            ("source", "来源", 0.6),
            ("processing", "炮制", 0.5),
            ("contraindication", "禁忌", 0.8),
            ("compatibility", "配伍", 0.7),
            ("clinical_application", "临床应用", 0.8),
            ("modern_research", "现代研究", 0.6),
            ("chemical_composition", "化学成分", 0.5),
            ("pharmacological_effects", "药理作用", 0.7),
            ("therapeutic_mechanism", "治疗机制", 0.8),
            ("side_effects", "副作用", 0.7),
            ("precautions", "注意事项", 0.7),
            ("storage", "贮藏", 0.4),
        ]
        
        # 按重要性排序并添加字段
        added_fields = 0
        for prop_key, label, importance in sorted(info_fields, key=lambda x: x[2], reverse=True):
            if prop_key in properties and properties[prop_key]:
                value = properties[prop_key]
                # 确保值不为空且不只是重复实体名
                if value and value.strip() and value.strip() != name:
                    # 格式化值，确保可读性
                    formatted_value = self._format_property_value(value)
                    if formatted_value:
                        content_parts.append(f"{label}：{formatted_value}")
                        added_fields += 1
                        # 限制字段数量，优先保留重要信息
                        if added_fields >= 8:
                            break
        
        # 如果没有添加任何字段，尝试提取所有其他属性
        if added_fields == 0:
            for key, value in properties.items():
                if key not in ['name', 'type', 'id'] and value and str(value).strip():
                    # 转换key为中文（如果可能）
                    label = self._translate_property_key(key)
                    formatted_value = self._format_property_value(value)
                    if formatted_value:
                        content_parts.append(f"{label}：{formatted_value}")
                        added_fields += 1
                        if added_fields >= 5:  # 最多添加5个额外字段
                            break
        
        # 添加实体基本信息（如果属性信息不足）
        if added_fields < 3:
            content_parts.append(f"基本信息：{name}是一种{type_name}，在中医理论中具有重要地位")
        
        content = "\n".join(content_parts)
        
        # 确保内容有最小长度和质量
        if len(content) < 80:
            content += f"\n\n说明：{name}是中医临床常用的{type_name}，具有重要的药用价值。"
        
        return content
    
    def _format_property_value(self, value: Any) -> str:
        """格式化属性值，提高可读性"""
        if not value:
            return ""
        
        value_str = str(value).strip()
        
        # 如果值太长，截断并添加省略号
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        
        # 清理特殊字符
        import re
        value_str = re.sub(r'\s+', ' ', value_str)  # 合并多个空格
        value_str = re.sub(r'[^\w\s\u4e00-\u9fff，。；：！？、]', '', value_str)  # 保留中文和基本标点
        
        return value_str
    
    def _translate_property_key(self, key: str) -> str:
        """将英文属性键翻译为中文"""
        translations = {
            "description": "描述",
            "effect": "功效", 
            "indication": "主治",
            "usage": "用法",
            "dosage": "用量",
            "properties": "性味",
            "meridian": "归经",
            "category": "分类",
            "source": "来源",
            "processing": "炮制",
            "contraindication": "禁忌",
            "compatibility": "配伍",
            "clinical_application": "临床应用",
            "modern_research": "现代研究",
            "chemical_composition": "化学成分",
            "pharmacological_effects": "药理作用",
            "therapeutic_mechanism": "治疗机制",
            "side_effects": "副作用",
            "precautions": "注意事项",
            "storage": "贮藏"
        }
        
        return translations.get(key.lower(), key.replace('_', ' ').title())
    
    def _infer_entity_type(self, name: str, properties: Dict[str, Any]) -> str:
        """智能推断实体类型"""
        name_lower = name.lower()
        
        # 根据名称特征推断类型
        if any(keyword in name_lower for keyword in ['汤', '散', '丸', '膏', '丹', '饮']):
            return "方剂"
        elif any(keyword in name_lower for keyword in ['痛', '热', '寒', '虚', '实', '燥', '湿']):
            return "症状"
        elif any(keyword in name_lower for keyword in ['病', '症', '证', '候']):
            return "疾病"
        elif any(keyword in name_lower for keyword in ['穴', '俞', '门', '关', '海']):
            return "穴位"
        elif any(keyword in name_lower for keyword in ['经', '脉', '络']):
            return "经络"
        elif any(keyword in name_lower for keyword in ['脏', '腑', '心', '肝', '脾', '肺', '肾', '胃', '胆', '膀胱']):
            return "脏腑"
        else:
            # 根据常见中药名称推断
            common_herbs = [
                '川芎', '桂枝', '红花', '丹参', '黄连', '黄芩', '枳实', '龙胆',
                '甘草', '人参', '白术', '茯苓', '当归', '白芍', '熟地', '枸杞',
                '黄芪', '党参', '山药', '薏米', '山楂', '陈皮', '半夏', '厚朴'
            ]
            if name in common_herbs:
                return "中药"
            else:
                return "中医实体"
    
    # 统计和监控方法
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息"""
        try:
            stats = {}
            
            # 节点总数
            result = self._run_query("MATCH (n) RETURN count(n) as node_count")
            stats["node_count"] = result[0]["node_count"] if result else 0
            
            # 关系总数
            result = self._run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            stats["relationship_count"] = result[0]["rel_count"] if result else 0
            
            # 实体类型分布
            result = self._run_query("""
                MATCH (n) 
                WHERE n.type IS NOT NULL
                RETURN n.type as entity_type, count(n) as count 
                ORDER BY count DESC
            """)
            stats["entity_type_distribution"] = {
                record["entity_type"]: record["count"] for record in result
            }
            
            # 关系类型分布
            result = self._run_query("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, count(r) as count 
                ORDER BY count DESC
            """)
            stats["relationship_type_distribution"] = {
                record["rel_type"]: record["count"] for record in result
            }
            
            # 连通性统计
            result = self._run_query("""
                MATCH (n) 
                WHERE NOT (n)--() 
                RETURN count(n) as isolated_count
            """)
            stats["isolated_nodes"] = result[0]["isolated_count"] if result else 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            start_time = datetime.now()
            
            # 测试基本连接
            self._test_connection()
            
            # 测试简单查询
            result = self._run_query("MATCH (n) RETURN count(n) as count LIMIT 1")
            node_count = result[0]["count"] if result else 0
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "node_count": node_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # 批量操作方法
    
    def batch_entity_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[EntityResult]]:
        """批量实体搜索"""
        results = {}
        for query in queries:
            try:
                # 注意：这里使用同步方法，实际应用中可能需要异步处理
                entity_results = asyncio.run(self.entity_search(query, top_k=top_k))
                results[query] = entity_results
            except Exception as e:
                self.logger.error(f"批量搜索失败 - 查询: {query}, 错误: {str(e)}")
                results[query] = []
        
        return results
    
    # 配置和管理方法
    
    @classmethod
    def from_config(cls, config: ModuleConfig) -> 'GraphRetrievalAdapter':
        """从配置创建适配器实例"""
        graph_config = config.graph_config
        
        return cls(
            neo4j_uri=graph_config.get("neo4j_uri", "bolt://localhost:7687"),
            username=graph_config.get("username", "neo4j"),
            password=graph_config.get("password", "password"),
            database=graph_config.get("database", "neo4j"),
            dump_path=graph_config.get("dump_path"),
            timeout=graph_config.get("timeout", 20)
        )
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 如果更新了连接相关配置，重新初始化连接
        connection_keys = ["neo4j_uri", "username", "password", "database"]
        if any(key in kwargs for key in connection_keys):
            self.close()
            self._initialize_connection()
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"GraphRetrievalAdapter(uri='{self.neo4j_uri}', database='{self.database}')"