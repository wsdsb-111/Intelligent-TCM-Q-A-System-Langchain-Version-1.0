// ========================================
// Neo4j导入脚本 - 正确的关系导入 第4部分
// 批次范围: 022 - 028
// 关系类型：包含、治疗、引起、表现为
// ========================================

// 导入前检查实体数量
MATCH (e:Entity) RETURN count(e) as entity_count;

// 分批导入关系数据（批次 022 到 028）

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


// ========================================
// 第4部分导入完成 - 验证查询
// ========================================

// 查看当前关系类型分布
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 查看总关系数量
MATCH ()-[r]->() RETURN count(r) as total_relationships;

// 第4部分导入完成！
