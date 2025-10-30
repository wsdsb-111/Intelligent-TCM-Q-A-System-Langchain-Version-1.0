#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j 配置管理模块
统一管理Neo4j数据库连接配置
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

class Neo4jConfig:
    """Neo4j配置管理类"""
    
    def __init__(self, env_file: str = '.env'):
        """
        初始化配置
        
        Args:
            env_file: 环境变量文件路径
        """
        # 加载环境变量
        load_dotenv(env_file)
        
        # 读取配置
        self.uri = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'hx1230047')  # 神农中医知识图谱密码
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
    def get_connection_config(self) -> Dict[str, Any]:
        """
        获取连接配置字典
        
        Returns:
            包含连接信息的字典
        """
        return {
            'uri': self.uri,
            'username': self.username,
            'password': self.password,
            'database': self.database
        }
    
    def get_driver_config(self) -> Dict[str, Any]:
        """
        获取Neo4j驱动配置
        
        Returns:
            Neo4j驱动配置字典
        """
        return {
            'uri': self.uri,
            'auth': (self.username, self.password)
        }
    
    def print_config(self):
        """打印当前配置信息（隐藏密码）"""
        print("Neo4j 配置信息:")
        print(f"  URI: {self.uri}")
        print(f"  用户名: {self.username}")
        print(f"  密码: {'*' * len(self.password)}")
        print(f"  数据库: {self.database}")
    
    def validate_config(self) -> bool:
        """
        验证配置是否完整
        
        Returns:
            配置是否有效
        """
        required_fields = [self.uri, self.username, self.password]
        return all(field for field in required_fields)

def get_neo4j_config() -> Neo4jConfig:
    """
    获取Neo4j配置实例
    
    Returns:
        Neo4jConfig实例
    """
    return Neo4jConfig()

if __name__ == "__main__":
    # 测试配置
    config = get_neo4j_config()
    config.print_config()
    
    if config.validate_config():
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败，请检查环境变量")