#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正知识图谱中的错误关系类型

主要功能：
1. 将"证候 -症状-> 望诊/闻诊/切诊/问诊"改为"证候 -诊断方法-> 望诊/闻诊/切诊/问诊"
2. 批量查询和修正错误关系
3. 支持交互式确认和批量修正
"""

import sys
import os
import json
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Neo4j配置（从环境变量或默认值读取）
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "hx1230047")  # 神农中医知识图谱密码
}

# 诊断方法列表（四诊法）
DIAGNOSTIC_METHODS = ['望诊', '闻诊', '问诊', '切诊', '望', '闻', '问', '切']

# 证候关键词（用于识别证候类实体）
SYNDROME_KEYWORDS = ['证', '症', '病']


class KGRelationFixer:
    """知识图谱关系修正工具"""
    
    def __init__(self):
        """初始化Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
            )
            print("✅ Neo4j连接成功")
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            sys.exit(1)
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
    
    def find_misclassified_relations(self) -> List[Dict]:
        """
        查找所有错误分类的关系
        返回格式: [{"source": "瘀血证", "rel_type": "症状", "target": "望诊"}, ...]
        """
        print("\n" + "="*80)
        print("🔍 正在扫描知识图谱中的错误关系...")
        print("="*80)
        
        with self.driver.session() as session:
            # 查询所有"症状"关系指向诊断方法的情况
            cypher_query = """
            MATCH (source)-[r:症状]->(target)
            WHERE target.name IN $diagnostic_methods
            RETURN source.name AS source_name, 
                   type(r) AS rel_type, 
                   target.name AS target_name
            ORDER BY source_name, target_name
            """
            
            result = session.run(cypher_query, diagnostic_methods=DIAGNOSTIC_METHODS)
            
            wrong_relations = []
            for record in result:
                wrong_relations.append({
                    "source": record["source_name"],
                    "rel_type": record["rel_type"],
                    "target": record["target_name"]
                })
            
            print(f"\n📊 发现 {len(wrong_relations)} 个错误关系")
            return wrong_relations
    
    def display_wrong_relations(self, relations: List[Dict], limit: int = 20):
        """展示错误关系"""
        if not relations:
            print("\n✅ 没有发现错误关系！知识图谱关系类型正确。")
            return
        
        print(f"\n📋 错误关系示例（前{min(limit, len(relations))}个）：")
        print("-" * 80)
        
        for i, rel in enumerate(relations[:limit], 1):
            print(f"{i:3d}. {rel['source']} -[{rel['rel_type']}]-> {rel['target']}")
            print(f"     ❌ 应改为: {rel['source']} -[诊断方法]-> {rel['target']}")
        
        if len(relations) > limit:
            print(f"\n... 还有 {len(relations) - limit} 个类似错误")
    
    def fix_single_relation(self, source: str, target: str) -> bool:
        """
        修正单个错误关系
        
        Args:
            source: 源节点名称（如"瘀血证"）
            target: 目标节点名称（如"望诊"）
        
        Returns:
            是否修正成功
        """
        with self.driver.session() as session:
            try:
                # 删除旧关系，创建新关系
                cypher_query = """
                MATCH (source {name: $source_name})-[old_rel:症状]->(target {name: $target_name})
                CREATE (source)-[new_rel:诊断方法]->(target)
                SET new_rel = properties(old_rel)
                DELETE old_rel
                RETURN count(new_rel) AS fixed_count
                """
                
                result = session.run(
                    cypher_query, 
                    source_name=source, 
                    target_name=target
                )
                
                record = result.single()
                if record and record["fixed_count"] > 0:
                    return True
                return False
                
            except Exception as e:
                print(f"❌ 修正失败: {e}")
                return False
    
    def fix_all_relations(self, relations: List[Dict]) -> Dict[str, int]:
        """
        批量修正所有错误关系
        
        Returns:
            统计信息 {"success": 成功数, "failed": 失败数}
        """
        print("\n" + "="*80)
        print("🔧 开始批量修正错误关系...")
        print("="*80)
        
        stats = {"success": 0, "failed": 0}
        
        for i, rel in enumerate(relations, 1):
            source = rel["source"]
            target = rel["target"]
            
            print(f"\n[{i}/{len(relations)}] 修正: {source} -> {target}", end=" ... ")
            
            if self.fix_single_relation(source, target):
                print("✅")
                stats["success"] += 1
            else:
                print("❌")
                stats["failed"] += 1
        
        print("\n" + "="*80)
        print("📊 修正完成统计:")
        print("="*80)
        print(f"  ✅ 成功: {stats['success']} 个")
        print(f"  ❌ 失败: {stats['failed']} 个")
        print(f"  📈 成功率: {stats['success']/len(relations)*100:.1f}%")
        
        return stats
    
    def verify_fixes(self) -> int:
        """
        验证修正结果
        
        Returns:
            剩余错误关系数量
        """
        print("\n" + "="*80)
        print("🔍 验证修正结果...")
        print("="*80)
        
        remaining = self.find_misclassified_relations()
        
        if not remaining:
            print("\n✅ 验证通过！所有错误关系已修正。")
        else:
            print(f"\n⚠️ 仍有 {len(remaining)} 个错误关系未修正")
            self.display_wrong_relations(remaining, limit=10)
        
        return len(remaining)
    
    def show_fixed_relations_sample(self, limit: int = 10):
        """展示修正后的关系示例"""
        print("\n" + "="*80)
        print(f"📋 修正后的关系示例（前{limit}个）：")
        print("="*80)
        
        with self.driver.session() as session:
            cypher_query = """
            MATCH (source)-[r:诊断方法]->(target)
            WHERE target.name IN $diagnostic_methods
            RETURN source.name AS source_name, 
                   type(r) AS rel_type, 
                   target.name AS target_name
            ORDER BY source_name, target_name
            LIMIT $limit
            """
            
            result = session.run(
                cypher_query, 
                diagnostic_methods=DIAGNOSTIC_METHODS,
                limit=limit
            )
            
            count = 0
            for record in result:
                count += 1
                print(f"{count:3d}. {record['source_name']} -[{record['rel_type']}]-> {record['target_name']}")
            
            if count == 0:
                print("  （暂无修正后的关系）")


def interactive_mode():
    """交互式模式"""
    print("="*80)
    print("知识图谱关系修正工具")
    print("="*80)
    print("\n功能说明：")
    print("  将错误的 '证候 -症状-> 诊断方法' 修正为 '证候 -诊断方法-> 诊断方法'")
    print("  例如：'瘀血证 -症状-> 望诊' 改为 '瘀血证 -诊断方法-> 望诊'")
    
    fixer = KGRelationFixer()
    
    try:
        # 步骤1: 扫描错误关系
        wrong_relations = fixer.find_misclassified_relations()
        
        if not wrong_relations:
            print("\n✅ 知识图谱中没有发现错误关系！")
            return
        
        # 步骤2: 展示错误关系
        fixer.display_wrong_relations(wrong_relations, limit=20)
        
        # 步骤3: 询问是否修正
        print("\n" + "="*80)
        choice = input("\n是否批量修正这些错误关系？(y/n): ").strip().lower()
        
        if choice != 'y':
            print("\n❌ 取消修正操作")
            return
        
        # 步骤4: 执行修正
        stats = fixer.fix_all_relations(wrong_relations)
        
        # 步骤5: 验证修正结果
        fixer.verify_fixes()
        
        # 步骤6: 展示修正后的关系示例
        fixer.show_fixed_relations_sample(limit=10)
        
        print("\n" + "="*80)
        print("🎉 关系修正流程完成！")
        print("="*80)
        
    finally:
        fixer.close()


def auto_fix_mode():
    """自动修正模式（无需确认）"""
    print("="*80)
    print("自动修正模式")
    print("="*80)
    
    fixer = KGRelationFixer()
    
    try:
        # 扫描并自动修正
        wrong_relations = fixer.find_misclassified_relations()
        
        if not wrong_relations:
            print("\n✅ 没有需要修正的关系")
            return
        
        stats = fixer.fix_all_relations(wrong_relations)
        fixer.verify_fixes()
        
        print("\n🎉 自动修正完成！")
        
    finally:
        fixer.close()


def query_mode():
    """查询模式（仅查看，不修改）"""
    print("="*80)
    print("查询模式（仅查看错误关系）")
    print("="*80)
    
    fixer = KGRelationFixer()
    
    try:
        wrong_relations = fixer.find_misclassified_relations()
        fixer.display_wrong_relations(wrong_relations, limit=50)
        
        # 保存到JSON
        output_file = Path(__file__).parent / "results" / "wrong_relations.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(wrong_relations, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 错误关系已保存到: {output_file}")
        
    finally:
        fixer.close()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'auto':
            auto_fix_mode()
        elif mode == 'query':
            query_mode()
        else:
            print(f"❌ 未知模式: {mode}")
            print("使用方法:")
            print("  python fix_kg_relations.py         # 交互式模式")
            print("  python fix_kg_relations.py auto    # 自动修正模式")
            print("  python fix_kg_relations.py query   # 查询模式")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()

