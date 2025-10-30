// ========================================
// 删除错误导入的关系
// ========================================

// 1. 查看当前的关系类型分布
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 2. 删除所有英文"RELATIONSHIP"关系
MATCH ()-[r:RELATIONSHIP]->() 
DELETE r;

// 3. 删除其他可能错误的关系类型（如果有的话）
MATCH ()-[r:关系]->() 
DELETE r;

// 4. 验证删除结果
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 5. 检查是否还有实体数据
MATCH (e:Entity) RETURN count(e) as entity_count;

// 删除所有关系
MATCH ()-[r]->() DELETE r;