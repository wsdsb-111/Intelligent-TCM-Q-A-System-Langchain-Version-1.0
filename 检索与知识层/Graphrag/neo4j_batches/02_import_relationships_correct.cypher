// ========================================
// Neo4j导入脚本 - 正确的关系导入
// 基于实际数据中的关系类型：包含、治疗、引起、表现为
// ========================================

// 导入前准备：确保实体已导入完成
MATCH (e:Entity) RETURN count(e) as entity_count;

// 清理现有关系（如果需要重新导入）
// MATCH ()-[r]->() DELETE r;

// 分批导入关系数据（处理所有33个批次文件）

// ===== 批次 001 =====
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_001.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_001.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_001.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_001.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型（如果有的话）
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_001.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE NOT row.relationship_type IN ['包含', '治疗', '引起', '表现为']
CREATE (source)-[r:其他关系 {
  type: row.relationship_type,
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// ===== 批次 002 =====
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_002.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_002.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_002.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_002.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_002.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE NOT row.relationship_type IN ['包含', '治疗', '引起', '表现为']
CREATE (source)-[r:其他关系 {
  type: row.relationship_type,
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// ===== 继续处理其他批次... =====
// 注意：这里只显示了前2个批次的示例
// 实际使用时需要为所有33个批次重复上述模式

// ========================================
// 验证查询
// ========================================

// 查看所有关系类型及数量
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 查看具体的关系类型分布
MATCH ()-[r:包含]->() RETURN '包含' as 关系类型, count(r) as 数量
UNION ALL
MATCH ()-[r:治疗]->() RETURN '治疗' as 关系类型, count(r) as 数量
UNION ALL
MATCH ()-[r:引起]->() RETURN '引起' as 关系类型, count(r) as 数量
UNION ALL
MATCH ()-[r:表现为]->() RETURN '表现为' as 关系类型, count(r) as 数量
UNION ALL
MATCH ()-[r:其他关系]->() RETURN '其他关系' as 关系类型, count(r) as 数量
ORDER BY 数量 DESC;