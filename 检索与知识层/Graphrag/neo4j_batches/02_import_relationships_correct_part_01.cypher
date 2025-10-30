// ========================================
// Neo4j导入脚本 - 正确的关系导入 第1部分
// 批次范围: 001 - 007
// 关系类型：包含、治疗、引起、表现为
// ========================================

// 导入前检查实体数量
MATCH (e:Entity) RETURN count(e) as entity_count;

// 分批导入关系数据（批次 001 到 007）

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

// 处理其他关系类型 - 批次 001
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

// 处理其他关系类型 - 批次 002
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

// ===== 批次 003 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_003.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_003.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_003.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_003.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 003
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_003.csv' AS row
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

// ===== 批次 004 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_004.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_004.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_004.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_004.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 004
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_004.csv' AS row
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

// ===== 批次 005 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_005.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_005.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_005.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_005.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 005
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_005.csv' AS row
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

// ===== 批次 006 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_006.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_006.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_006.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_006.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 006
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_006.csv' AS row
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

// ===== 批次 007 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_007.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_007.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_007.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_007.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 007
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_007.csv' AS row
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


// ========================================
// 第1部分导入完成 - 验证查询
// ========================================

// 查看当前关系类型分布
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 查看总关系数量
MATCH ()-[r]->() RETURN count(r) as total_relationships;

// 第1部分导入完成！
