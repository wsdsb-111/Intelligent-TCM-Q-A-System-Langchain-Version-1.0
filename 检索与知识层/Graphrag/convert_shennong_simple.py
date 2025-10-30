#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将神农数据集转换为简单的 CSV 格式
只保留 id, query, response 三个字段
"""

import json
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_simple_csv(input_file: str, output_file: str, max_records: int = None):
    """
    转换为简单CSV格式，只保留id, query, response
    
    Args:
        input_file: 输入文件路径
        output_file: 输出CSV文件路径
        max_records: 最大处理记录数
    """
    logger.info(f"开始转换: {input_file} -> {output_file}")
    
    # 读取数据
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_records and len(records) >= max_records:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if 'query' in data and 'response' in data:
                    records.append(data)
            except json.JSONDecodeError as e:
                logger.error(f"第{line_num}行解析错误: {e}")
    
    logger.info(f"加载了 {len(records)} 条记录")
    
    # 写入CSV
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # 写入表头
        writer.writerow(['id', 'query', 'response'])
        
        # 写入数据
        for idx, record in enumerate(records, 1):
            writer.writerow([
                f"SHENNONG_{idx:06d}",
                record['query'],
                record['response']
            ])
    
    logger.info(f"转换完成，保存到: {output_file}")
    logger.info(f"总记录数: {len(records)}")


def main():
    print("="*60)
    print("🔄 神农数据集简化转换工具")
    print("="*60)
    
    # 配置
    input_file = r"E:\毕业论文和设计\线上智能中医问答项目\检索与知识层\Graphrag\dataset\shennong\ChatMed_TCM-v0.2.json"
    output_file = "dataset/shennong/shennong_simple.csv"
    max_records = None  # None表示全部
    
    print(f"\n📋 配置:")
    print(f"   输入: {input_file}")
    print(f"   输出: {output_file}")
    print(f"   记录数: {'全部' if max_records is None else max_records}")
    print()
    
    try:
        convert_simple_csv(input_file, output_file, max_records)
        
        print("\n✅ 转换完成！")
        print(f"📄 CSV文件: {output_file}")
        print(f"📊 格式: id, query, response")
        print("="*60)
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

