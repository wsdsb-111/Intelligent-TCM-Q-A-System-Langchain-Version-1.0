// ========================================
// 最终验证脚本 - 正确的关系类型验证
// ========================================

// 验证所有关系类型及数量
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 详细的关系类型统计
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
LIMIT 3;

MATCH (s:Entity)-[r:治疗]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 3;

MATCH (s:Entity)-[r:引起]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 3;

MATCH (s:Entity)-[r:表现为]->(t:Entity)
RETURN s.name as source, type(r) as relationship, t.name as target, r.description as description
LIMIT 3;
