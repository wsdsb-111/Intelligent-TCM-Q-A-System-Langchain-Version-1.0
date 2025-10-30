#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向知识图谱中添加中药信息的工具脚本
支持添加中药节点及其功效、作用等关系
"""

from neo4j import GraphDatabase
import sys
from pathlib import Path

# Neo4j连接配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "hx1230047"
NEO4J_DATABASE = "neo4j"


class HerbKnowledgeGraphManager:
    """中药知识图谱管理器"""
    
    def __init__(self, uri, username, password, database="neo4j"):
        """初始化连接"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        print(f"✅ 已连接到Neo4j: {uri}")
    
    def close(self):
        """关闭连接"""
        self.driver.close()
        print("✅ 已关闭Neo4j连接")
    
    def add_herb_with_effects(self, herb_name, effects_dict):
        """
        添加中药及其功效到知识图谱
        
        Args:
            herb_name: 中药名称
            effects_dict: 功效字典，格式如 {
                "性味": "辛，温",
                "归经": ["脾经", "胃经", "肺经"],
                "功效": ["增强血液循环", "刺激胃液分泌", "兴奋肠管", "促进消化", "健胃增进食欲"],
                "主治": ["风寒感冒", "胃寒呕吐"],
                "用法": "煎服，3-10g"
            }
        """
        with self.driver.session(database=self.database) as session:
            # 1. 创建或更新中药节点
            herb_node = session.run("""
                MERGE (h:中药 {name: $herb_name})
                SET h.类别 = '中药',
                    h.更新时间 = datetime()
                RETURN h
            """, herb_name=herb_name).single()
            
            print(f"\n✅ 已创建/更新中药节点: {herb_name}")
            
            # 2. 添加性味
            if "性味" in effects_dict:
                session.run("""
                    MATCH (h:中药 {name: $herb_name})
                    SET h.性味 = $value
                """, herb_name=herb_name, value=effects_dict["性味"])
                print(f"  ✓ 已添加性味: {effects_dict['性味']}")
            
            # 3. 添加归经
            if "归经" in effects_dict:
                for meridian in effects_dict["归经"]:
                    session.run("""
                        MATCH (h:中药 {name: $herb_name})
                        MERGE (m:经络 {name: $meridian})
                        MERGE (h)-[r:归经]->(m)
                        SET r.创建时间 = datetime()
                    """, herb_name=herb_name, meridian=meridian)
                    print(f"  ✓ 已添加归经关系: {herb_name} -> {meridian}")
            
            # 4. 添加功效节点和关系
            if "功效" in effects_dict:
                for effect in effects_dict["功效"]:
                    session.run("""
                        MATCH (h:中药 {name: $herb_name})
                        MERGE (e:功效 {name: $effect})
                        MERGE (h)-[r:具有功效]->(e)
                        SET r.创建时间 = datetime()
                    """, herb_name=herb_name, effect=effect)
                    print(f"  ✓ 已添加功效关系: {herb_name} -> {effect}")
            
            # 5. 添加主治
            if "主治" in effects_dict:
                for disease in effects_dict["主治"]:
                    session.run("""
                        MATCH (h:中药 {name: $herb_name})
                        MERGE (d:疾病 {name: $disease})
                        MERGE (h)-[r:主治]->(d)
                        SET r.创建时间 = datetime()
                    """, herb_name=herb_name, disease=disease)
                    print(f"  ✓ 已添加主治关系: {herb_name} -> {disease}")
            
            # 6. 添加用法用量
            if "用法" in effects_dict:
                session.run("""
                    MATCH (h:中药 {name: $herb_name})
                    SET h.用法用量 = $value
                """, herb_name=herb_name, value=effects_dict["用法"])
                print(f"  ✓ 已添加用法用量: {effects_dict['用法']}")
            
            # 7. 添加备注
            if "备注" in effects_dict:
                session.run("""
                    MATCH (h:中药 {name: $herb_name})
                    SET h.备注 = $value
                """, herb_name=herb_name, value=effects_dict["备注"])
                print(f"  ✓ 已添加备注")
            
            print(f"\n🎉 成功添加中药 '{herb_name}' 到知识图谱!")
    
    def query_herb(self, herb_name):
        """查询中药信息"""
        with self.driver.session(database=self.database) as session:
            # 查询中药节点及其关系
            result = session.run("""
                MATCH (h:中药 {name: $herb_name})
                OPTIONAL MATCH (h)-[r]->(target)
                RETURN h, type(r) as rel_type, target
            """, herb_name=herb_name)
            
            print(f"\n📋 知识图谱中 '{herb_name}' 的信息:")
            print("="*60)
            
            records = list(result)
            if not records:
                print(f"❌ 未找到 '{herb_name}' 的信息")
                return
            
            # 打印节点属性
            herb_node = records[0]['h']
            print(f"\n🌿 中药: {herb_name}")
            for key, value in dict(herb_node).items():
                if key != 'name':
                    print(f"  {key}: {value}")
            
            # 打印关系
            print(f"\n🔗 关系:")
            for record in records:
                if record['rel_type']:
                    target_name = dict(record['target']).get('name', str(record['target']))
                    print(f"  {record['rel_type']} -> {target_name}")
            
            print("="*60)


def add_ginger_to_kg():
    """添加'姜'到知识图谱（示例）"""
    
    # 创建管理器
    manager = HerbKnowledgeGraphManager(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    try:
        # 定义姜的信息
        ginger_info = {
            "性味": "辛，温",
            "归经": ["脾经", "胃经", "肺经"],
            "功效": [
                "增强血液循环",
                "刺激胃液分泌",
                "兴奋肠管",
                "促进消化",
                "健胃增进食欲",
                "温中散寒",
                "解表发汗"
            ],
            "主治": [
                "风寒感冒",
                "胃寒呕吐",
                "寒痰咳嗽",
                "脘腹冷痛",
                "食欲不振"
            ],
            "用法": "煎服，3-10g；或捣汁服",
            "备注": "生姜用于解表散寒、温中止呕；干姜温中散寒力强"
        }
        
        # 添加到知识图谱
        print("\n" + "="*60)
        print("开始添加'姜'到知识图谱")
        print("="*60)
        
        manager.add_herb_with_effects("姜", ginger_info)
        
        # 查询验证
        print("\n" + "="*60)
        print("验证添加结果")
        print("="*60)
        manager.query_herb("姜")
        
    finally:
        manager.close()


def add_custom_herb():
    """交互式添加自定义中药"""
    
    print("\n" + "="*60)
    print("交互式添加中药到知识图谱")
    print("="*60)
    
    herb_name = input("\n请输入中药名称: ").strip()
    if not herb_name:
        print("❌ 中药名称不能为空")
        return
    
    # 创建管理器
    manager = HerbKnowledgeGraphManager(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    try:
        herb_info = {}
        
        # 性味
        xingwei = input("性味（如'辛，温'，可选）: ").strip()
        if xingwei:
            herb_info["性味"] = xingwei
        
        # 归经
        guijing = input("归经（多个用逗号分隔，如'脾经,胃经'，可选）: ").strip()
        if guijing:
            herb_info["归经"] = [x.strip() for x in guijing.split(',')]
        
        # 功效
        gongxiao = input("功效（多个用逗号分隔，必填）: ").strip()
        if gongxiao:
            herb_info["功效"] = [x.strip() for x in gongxiao.split(',')]
        else:
            print("❌ 功效不能为空")
            return
        
        # 主治
        zhuzhi = input("主治（多个用逗号分隔，可选）: ").strip()
        if zhuzhi:
            herb_info["主治"] = [x.strip() for x in zhuzhi.split(',')]
        
        # 用法
        yongfa = input("用法用量（可选）: ").strip()
        if yongfa:
            herb_info["用法"] = yongfa
        
        # 添加到知识图谱
        print("\n" + "="*60)
        print(f"开始添加'{herb_name}'到知识图谱")
        print("="*60)
        
        manager.add_herb_with_effects(herb_name, herb_info)
        
        # 查询验证
        manager.query_herb(herb_name)
        
    finally:
        manager.close()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("中药知识图谱管理工具")
    print("="*60)
    print("\n请选择操作:")
    print("1. 添加'姜'到知识图谱（预定义数据）")
    print("2. 交互式添加自定义中药")
    print("3. 查询中药信息")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-3): ").strip()
    
    if choice == "1":
        add_ginger_to_kg()
    elif choice == "2":
        add_custom_herb()
    elif choice == "3":
        herb_name = input("\n请输入要查询的中药名称: ").strip()
        if herb_name:
            manager = HerbKnowledgeGraphManager(
                uri=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE
            )
            try:
                manager.query_herb(herb_name)
            finally:
                manager.close()
    elif choice == "0":
        print("\n👋 再见!")
    else:
        print("\n❌ 无效选项")


if __name__ == "__main__":
    main()

