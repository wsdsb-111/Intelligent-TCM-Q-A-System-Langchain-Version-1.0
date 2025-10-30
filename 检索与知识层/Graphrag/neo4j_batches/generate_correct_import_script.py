#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成正确的Neo4j关系导入脚本
基于实际数据中的关系类型：包含、治疗、引起、表现为
"""

def generate_correct_import_script():
    """生成基于实际关系类型的完整导入脚本"""
    
    # 实际存在的关系类型
    relationship_types = ['包含', '治疗', '引起', '表现为']
    
    # 脚本头部
    script_content = """// ========================================
// Neo4j导入脚本 - 正确的完整关系导入
// 基于实际数据中的关系类型：包含、治疗、引起、表现为
// 处理所有33个批次文件
// ========================================

// 导入前准备：确保实体已导入完成
MATCH (e:Entity) RETURN count(e) as entity_count;

// 可选：清理现有关系（如果需要重新导入）
// MATCH ()-[r]->() DELETE r;

// 分批导入关系数据（处理所有33个批次文件）

"""

    # 为每个批次文件生成导入语句
    for batch_num in range(1, 34):  # 1到33
        batch_file = f"relationships_batch_{batch_num:03d}.csv"
        
        script_content += f"// ===== 批次 {batch_num:03d} =====\n"
        
        # 为每种实际存在的关系类型生成导入语句
        for rel_type in relationship_types:
            script_content += f"""
LOAD CSV WITH HEADERS FROM 'file:///{batch_file}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
WHERE row.relationship_type = '{rel_type}'
CREATE (source)-[r:{rel_type} {{
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}]->(target);
"""
        
        # 处理其他关系类型（如果有的话）
        script_content += f"""
// 处理其他关系类型 - 批次 {batch_num:03d}
LOAD CSV WITH HEADERS FROM 'file:///{batch_file}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
WHERE NOT row.relationship_type IN {relationship_types}
CREATE (source)-[r:其他关系 {{
  type: row.relationship_type,
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}]->(target);

"""
        
        # 每5个批次添加一个暂停建议
        if batch_num % 5 == 0:
            script_content += f"// ===== 批次 {batch_num:03d} 完成，建议暂停10-30秒 =====\n\n"
    
    # 脚本尾部 - 验证查询
    script_content += """
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
"""
    
    return script_content

def generate_batch_scripts():
    """生成分段的导入脚本"""
    
    relationship_types = ['包含', '治疗', '引起', '表现为']
    
    # 定义分段方案
    batch_ranges = [
        (1, 7),    # 第1部分：批次 001-007
        (8, 14),   # 第2部分：批次 008-014
        (15, 21),  # 第3部分：批次 015-021
        (22, 28),  # 第4部分：批次 022-028
        (29, 33),  # 第5部分：批次 029-033
    ]
    
    generated_files = []
    
    for i, (start_batch, end_batch) in enumerate(batch_ranges, 1):
        script_content = f"""// ========================================
// Neo4j导入脚本 - 正确的关系导入 第{i}部分
// 批次范围: {start_batch:03d} - {end_batch:03d}
// 关系类型：包含、治疗、引起、表现为
// ========================================

// 导入前检查实体数量
MATCH (e:Entity) RETURN count(e) as entity_count;

// 分批导入关系数据（批次 {start_batch:03d} 到 {end_batch:03d}）

"""

        # 为指定范围的批次文件生成导入语句
        for batch_num in range(start_batch, end_batch + 1):
            batch_file = f"relationships_batch_{batch_num:03d}.csv"
            
            script_content += f"// ===== 批次 {batch_num:03d} =====\n"
            
            # 为每种关系类型生成导入语句
            for rel_type in relationship_types:
                script_content += f"""
LOAD CSV WITH HEADERS FROM 'file:///{batch_file}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
WHERE row.relationship_type = '{rel_type}'
CREATE (source)-[r:{rel_type} {{
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}]->(target);
"""
            
            # 处理其他关系类型
            script_content += f"""
// 处理其他关系类型 - 批次 {batch_num:03d}
LOAD CSV WITH HEADERS FROM 'file:///{batch_file}' AS row
MATCH (source:Entity {{id: row.source_id}})
MATCH (target:Entity {{id: row.target_id}})
WHERE NOT row.relationship_type IN {relationship_types}
CREATE (source)-[r:其他关系 {{
  type: row.relationship_type,
  description: row.description,
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  source_document_id: row.source_document_id
}}]->(target);

"""
        
        # 脚本尾部 - 验证查询
        script_content += f"""
// ========================================
// 第{i}部分导入完成 - 验证查询
// ========================================

// 查看当前关系类型分布
MATCH ()-[r]->() 
RETURN type(r) as relationship_type, count(r) as count 
ORDER BY count DESC;

// 查看总关系数量
MATCH ()-[r]->() RETURN count(r) as total_relationships;

// 第{i}部分导入完成！
"""
        
        # 保存脚本
        output_file = f"02_import_relationships_correct_part_{i:02d}.cypher"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        generated_files.append(output_file)
    
    return generated_files

def main():
    """主函数"""
    print("正在生成正确的Neo4j关系导入脚本...")
    print("发现的实际关系类型：包含、治疗、引起、表现为")
    
    # 生成完整脚本
    script_content = generate_correct_import_script()
    output_file = "02_import_relationships_correct_complete.cypher"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 完整导入脚本已生成: {output_file}")
    
    # 生成分段脚本
    generated_files = generate_batch_scripts()
    
    print(f"\n✅ 分段导入脚本已生成:")
    for file in generated_files:
        print(f"   - {file}")
    
    # 生成最终验证脚本
    validation_script = """// ========================================
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
"""
    
    validation_file = "03_final_validation_correct.cypher"
    with open(validation_file, 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    print(f"✅ 验证脚本已生成: {validation_file}")
    
    print("\n🎯 问题解决方案:")
    print("你的数据中实际只有4种关系类型，而不是10种：")
    print("   1. 包含 - 方剂包含药材")
    print("   2. 治疗 - 药材/方剂治疗疾病/症状")
    print("   3. 引起 - 疾病引起症状")
    print("   4. 表现为 - 疾病的症状表现")
    
    print("\n🚀 推荐执行顺序:")
    print("1. 使用分段脚本：02_import_relationships_correct_part_01.cypher 到 part_05.cypher")
    print("2. 或者使用完整脚本：02_import_relationships_correct_complete.cypher")
    print("3. 最后执行验证脚本：03_final_validation_correct.cypher")
    
    print("\n✨ 执行完成后，你应该能看到4种关系类型，而不是之前的3种！")

if __name__ == "__main__":
    main()