// ========================================
// Neo4j导入脚本 - 实体导入
// ========================================

// 1. 清理可能冲突的索引
DROP INDEX entity_id_index IF EXISTS;

// 2. 创建唯一约束（会自动创建索引）
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// 3. 创建其他索引（提高查询性能）
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);

// 4. 导入实体数据
LOAD CSV WITH HEADERS FROM 'file:///20250921_191838_optimized_json_dataset_entities.csv' AS row
CREATE (e:Entity {
  id: row.id,
  name: row.name,
  type: row.type,
  description: row.description,
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
});

// 5. 验证实体导入
MATCH (e:Entity) RETURN count(e) as entity_count;
