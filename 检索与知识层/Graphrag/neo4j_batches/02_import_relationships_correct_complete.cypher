// ========================================
// Neo4j导入脚本 - 正确的完整关系导入
// 基于实际数据中的关系类型：包含、治疗、引起、表现为
// 处理所有33个批次文件
// ========================================

// 导入前准备：确保实体已导入完成
MATCH (e:Entity) RETURN count(e) as entity_count;

// 可选：清理现有关系（如果需要重新导入）
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

// ===== 批次 005 完成，建议暂停10-30秒 =====

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

// ===== 批次 008 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_008.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_008.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_008.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_008.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 008
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_008.csv' AS row
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

// ===== 批次 009 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_009.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_009.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_009.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_009.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 009
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_009.csv' AS row
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

// ===== 批次 010 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_010.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_010.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_010.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_010.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 010
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_010.csv' AS row
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

// ===== 批次 010 完成，建议暂停10-30秒 =====

// ===== 批次 011 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_011.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_011.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_011.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_011.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 011
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_011.csv' AS row
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

// ===== 批次 012 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_012.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_012.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_012.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_012.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 012
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_012.csv' AS row
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

// ===== 批次 013 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_013.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_013.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_013.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_013.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 013
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_013.csv' AS row
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

// ===== 批次 014 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_014.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_014.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_014.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_014.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 014
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_014.csv' AS row
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

// ===== 批次 015 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_015.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_015.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_015.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_015.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 015
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_015.csv' AS row
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

// ===== 批次 015 完成，建议暂停10-30秒 =====

// ===== 批次 016 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_016.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_016.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_016.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_016.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 016
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_016.csv' AS row
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

// ===== 批次 017 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_017.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_017.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_017.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_017.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 017
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_017.csv' AS row
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

// ===== 批次 018 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_018.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_018.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_018.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_018.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 018
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_018.csv' AS row
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

// ===== 批次 019 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_019.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_019.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_019.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_019.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 019
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_019.csv' AS row
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

// ===== 批次 020 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_020.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_020.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_020.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_020.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 020
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_020.csv' AS row
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

// ===== 批次 020 完成，建议暂停10-30秒 =====

// ===== 批次 021 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_021.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_021.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_021.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_021.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 021
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_021.csv' AS row
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

// ===== 批次 022 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_022.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_022.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_022.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_022.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 022
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_022.csv' AS row
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

// ===== 批次 023 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_023.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_023.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_023.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_023.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 023
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_023.csv' AS row
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

// ===== 批次 024 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_024.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_024.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_024.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_024.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 024
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_024.csv' AS row
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

// ===== 批次 025 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_025.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_025.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_025.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_025.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 025
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_025.csv' AS row
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

// ===== 批次 025 完成，建议暂停10-30秒 =====

// ===== 批次 026 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_026.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_026.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_026.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_026.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 026
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_026.csv' AS row
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

// ===== 批次 027 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_027.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_027.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_027.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_027.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 027
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_027.csv' AS row
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

// ===== 批次 028 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_028.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_028.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_028.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_028.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 028
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_028.csv' AS row
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

// ===== 批次 029 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_029.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_029.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_029.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_029.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 029
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_029.csv' AS row
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

// ===== 批次 030 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_030.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_030.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_030.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_030.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 030
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_030.csv' AS row
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

// ===== 批次 030 完成，建议暂停10-30秒 =====

// ===== 批次 031 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_031.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_031.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_031.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_031.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 031
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_031.csv' AS row
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

// ===== 批次 032 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_032.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_032.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_032.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_032.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 032
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_032.csv' AS row
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

// ===== 批次 033 =====

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_033.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '包含'
CREATE (source)-[r:包含 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_033.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '治疗'
CREATE (source)-[r:治疗 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_033.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '引起'
CREATE (source)-[r:引起 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_033.csv' AS row
MATCH (source:Entity {id: row.source_id})
MATCH (target:Entity {id: row.target_id})
WHERE row.relationship_type = '表现为'
CREATE (source)-[r:表现为 {
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}]->(target);

// 处理其他关系类型 - 批次 033
LOAD CSV WITH HEADERS FROM 'file:///relationships_batch_033.csv' AS row
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
// 最终验证
// ========================================

// 验证导入结果 - 查看所有关系类型及数量
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

// 查看实体数量
MATCH (e:Entity) RETURN count(e) as entity_count;

// 查看总关系数量
MATCH ()-[r]->() RETURN count(r) as total_relationships;

// 检查是否有孤立的实体
MATCH (e:Entity)
WHERE NOT (e)-[]-()
RETURN count(e) as isolated_entities;

// 查看关系密度最高的实体
MATCH (e:Entity)-[r]-()
RETURN e.name as entity_name, count(r) as relationship_count
ORDER BY relationship_count DESC
LIMIT 10;

// 查看每种关系类型的示例
MATCH (s:Entity)-[r:包含]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 5;

MATCH (s:Entity)-[r:治疗]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 5;

MATCH (s:Entity)-[r:引起]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 5;

MATCH (s:Entity)-[r:表现为]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 5;
