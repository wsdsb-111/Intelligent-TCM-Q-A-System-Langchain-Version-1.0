#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从神农数据集 CSV 文件中提取实体和关系
使用 GraphRAGProcessor 进行知识图谱构建
"""

import csv
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.graphrag_processor import GraphRAGProcessor
from src.config import SimpleConfigManager
from src.models import ProcessedDocument, GraphRAGResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shennong_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShennongCSVExtractor:
    """神农 CSV 数据提取器"""
    
    def __init__(self, csv_path: str, output_dir: str = "output_extraction"):
        """
        初始化提取器
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载配置和处理器
        config_manager = SimpleConfigManager()
        self.config = config_manager.load_config()
        self.processor = GraphRAGProcessor(self.config.graphrag)
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"初始化神农CSV提取器")
        logger.info(f"CSV文件: {self.csv_path}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def load_csv(self, max_records: int = None) -> List[Dict[str, Any]]:
        """
        加载 CSV 文件
        
        Args:
            max_records: 最大加载记录数
            
        Returns:
            List[Dict]: 记录列表
        """
        logger.info(f"开始加载CSV文件: {self.csv_path}")
        
        records = []
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if max_records and len(records) >= max_records:
                    break
                
                # 确保必要字段存在
                if 'id' in row and 'query' in row and 'response' in row:
                    records.append(row)
                else:
                    logger.warning(f"跳过格式不正确的行: {row}")
        
        logger.info(f"CSV加载完成，共 {len(records)} 条记录")
        self.stats['total_records'] = len(records)
        return records
    
    def csv_to_document(self, record: Dict[str, Any]) -> ProcessedDocument:
        """
        将CSV记录转换为ProcessedDocument
        
        Args:
            record: CSV记录
            
        Returns:
            ProcessedDocument
        """
        # 组合问题和回答作为文档内容
        content = f"问题：{record['query']}\n\n回答：{record['response']}"
        
        # 使用ID作为标题
        title = f"神农问答_{record['id']}"
        
        return ProcessedDocument(
            title=title,
            content=content,
            file_path=f"shennong_csv_{record['id']}",
            file_type="csv",
            metadata={
                'record_id': record['id'],
                'original_query': record['query'],
                'original_response': record['response'],
                'source': 'shennong_simple.csv'
            }
        )
    
    def save_batch_results(self, results: List[GraphRAGResult], batch_num: int):
        """
        保存批次结果
        
        Args:
            results: 提取结果列表
            batch_num: 批次号
        """
        import json
        
        # 收集实体和关系
        all_entities = []
        all_relationships = []
        
        for result in results:
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
        
        # 保存实体 CSV
        if all_entities:
            entities_file = self.output_dir / f"entities_batch_{batch_num:03d}.csv"
            with open(entities_file, 'w', encoding='utf-8-sig', newline='') as f:
                import csv
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(['id', 'name', 'type', 'description', 'source_document_id'])
                
                for entity in all_entities:
                    writer.writerow([
                        entity.id,
                        entity.name,
                        entity.type,
                        entity.description,
                        entity.source_document_id
                    ])
            logger.info(f"实体已保存: {entities_file} ({len(all_entities)}个)")
        
        # 保存关系 CSV
        if all_relationships:
            rels_file = self.output_dir / f"relationships_batch_{batch_num:03d}.csv"
            with open(rels_file, 'w', encoding='utf-8-sig', newline='') as f:
                import csv
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(['id', 'source_entity_id', 'target_entity_id', 'type', 'description', 'source_document_id'])
                
                for rel in all_relationships:
                    writer.writerow([
                        rel.id,
                        rel.source_entity_id,
                        rel.target_entity_id,
                        rel.type,
                        rel.description,
                        rel.source_document_id
                    ])
            logger.info(f"关系已保存: {rels_file} ({len(all_relationships)}个)")
        
        # 保存统计信息
        stats_file = self.output_dir / f"stats_batch_{batch_num:03d}.json"
        batch_stats = {
            'batch_num': batch_num,
            'processed_records': len(results),
            'total_entities': len(all_entities),
            'total_relationships': len(all_relationships),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
    
    def extract_with_pause_control(self, max_records: int = None, batch_size: int = 10):
        """
        使用暂停控制进行提取
        
        Args:
            max_records: 最大处理记录数
            batch_size: 批次大小
        """
        logger.info("开始提取实体和关系")
        
        self.stats['start_time'] = datetime.now()
        
        # 重置处理器状态
        self.processor.reset_processing_state()
        
        # 加载CSV
        records = self.load_csv(max_records)
        
        if not records:
            logger.error("没有加载到任何记录")
            return
        
        logger.info(f"将处理 {len(records)} 条记录，每批 {batch_size} 条")
        
        # 分批处理
        batch_num = 1
        all_results = []
        
        for i in range(0, len(records), batch_size):
            # 检查暂停/停止
            self.processor.pause_controller.wait_if_paused()
            
            if self.processor.pause_controller.should_stop():
                logger.info("处理被用户停止")
                break
            
            # 获取当前批次
            batch_records = records[i:i + batch_size]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"处理批次 {batch_num}，记录 {i+1}-{i+len(batch_records)}/{len(records)}")
            logger.info(f"{'='*60}")
            
            # 转换为文档并提取
            batch_results = []
            
            for j, record in enumerate(batch_records):
                # 检查暂停/停止
                self.processor.pause_controller.wait_if_paused()
                
                if self.processor.pause_controller.should_stop():
                    logger.info("处理被用户停止")
                    break
                
                try:
                    # 转换为文档
                    document = self.csv_to_document(record)
                    
                    logger.info(f"处理记录 {i+j+1}/{len(records)}: {record['id']}")
                    
                    # 提取实体和关系
                    result = self.processor.extract_entities_and_relationships(document)
                    batch_results.append(result)
                    
                    # 更新统计
                    self.stats['processed_records'] += 1
                    if result.entities or result.relationships:
                        self.stats['successful_extractions'] += 1
                        self.stats['total_entities'] += len(result.entities)
                        self.stats['total_relationships'] += len(result.relationships)
                    else:
                        self.stats['failed_extractions'] += 1
                    
                    logger.info(f"✅ 提取完成: 实体{len(result.entities)}个, 关系{len(result.relationships)}个")
                    
                except Exception as e:
                    logger.error(f"❌ 处理记录 {record['id']} 失败: {e}")
                    self.stats['failed_extractions'] += 1
                
                # 短暂休息
                time.sleep(0.1)
            
            # 保存批次结果
            if batch_results:
                self.save_batch_results(batch_results, batch_num)
                all_results.extend(batch_results)
            
            batch_num += 1
            
            # 显示进度
            progress = (i + len(batch_records)) / len(records) * 100
            logger.info(f"\n📊 总体进度: {progress:.1f}% ({i+len(batch_records)}/{len(records)})")
            logger.info(f"📈 累计: 实体{self.stats['total_entities']}个, 关系{self.stats['total_relationships']}个\n")
            
            # 短暂休息
            time.sleep(0.5)
        
        self.stats['end_time'] = datetime.now()
        
        # 保存最终统计
        self.save_final_stats()
        
        logger.info("\n🎉 提取完成！")
        return all_results
    
    def save_final_stats(self):
        """保存最终统计信息"""
        import json
        
        final_stats = self.stats.copy()
        
        if final_stats['start_time'] and final_stats['end_time']:
            total_time = (final_stats['end_time'] - final_stats['start_time']).total_seconds()
            final_stats['total_time_seconds'] = total_time
            final_stats['avg_time_per_record'] = total_time / final_stats['processed_records'] if final_stats['processed_records'] > 0 else 0
        
        # 转换datetime为字符串
        final_stats['start_time'] = final_stats['start_time'].isoformat() if final_stats['start_time'] else None
        final_stats['end_time'] = final_stats['end_time'].isoformat() if final_stats['end_time'] else None
        
        stats_file = self.output_dir / "final_extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最终统计已保存: {stats_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 提取统计摘要")
        print("="*60)
        print(f"总记录数: {final_stats['total_records']}")
        print(f"已处理: {final_stats['processed_records']}")
        print(f"成功: {final_stats['successful_extractions']}")
        print(f"失败: {final_stats['failed_extractions']}")
        print(f"提取实体: {final_stats['total_entities']}个")
        print(f"提取关系: {final_stats['total_relationships']}个")
        
        if 'total_time_seconds' in final_stats:
            print(f"总耗时: {final_stats['total_time_seconds']:.2f}秒")
            print(f"平均每条: {final_stats['avg_time_per_record']:.2f}秒")
        
        print(f"\n📁 输出目录: {self.output_dir}")
        print("="*60)


def interactive_control(extractor: ShennongCSVExtractor):
    """交互式控制"""
    print("\n🎮 交互式控制面板")
    print("="*40)
    print("命令:")
    print("  p - 暂停")
    print("  r - 恢复")
    print("  s - 停止")
    print("  status - 查看状态")
    print("  stats - 查看统计")
    print("  q - 退出")
    print("="*40)
    
    def control_loop():
        while True:
            try:
                cmd = input("\n命令: ").strip().lower()
                
                if cmd == 'p':
                    extractor.processor.pause_processing()
                    print("⏸️  已暂停")
                elif cmd == 'r':
                    extractor.processor.resume_processing()
                    print("▶️  已恢复")
                elif cmd == 's':
                    extractor.processor.stop_processing()
                    print("⏹️  已停止")
                elif cmd == 'status':
                    print(f"📊 状态: {extractor.processor.get_processing_status()}")
                elif cmd == 'stats':
                    print(f"📈 统计:")
                    print(f"   已处理: {extractor.stats['processed_records']}/{extractor.stats['total_records']}")
                    print(f"   实体: {extractor.stats['total_entities']}")
                    print(f"   关系: {extractor.stats['total_relationships']}")
                elif cmd == 'q':
                    extractor.processor.stop_processing()
                    print("👋 退出")
                    break
                else:
                    print("❌ 无效命令")
            except KeyboardInterrupt:
                extractor.processor.stop_processing()
                break
    
    thread = threading.Thread(target=control_loop, daemon=True)
    thread.start()
    return thread


def main():
    """主函数"""
    print("="*60)
    print("🚀 神农数据集实体关系提取系统")
    print("="*60)
    
    # 配置
    csv_path = r"E:\毕业论文和设计\线上智能中医问答项目\检索与知识层\Graphrag\dataset\shennong\shennong_simple.csv"
    output_dir = "output_shennong_extraction"
    max_records = 50  # 限制处理数量，设置为None处理全部
    batch_size = 5
    
    print(f"\n📋 配置:")
    print(f"   CSV文件: {csv_path}")
    print(f"   输出目录: {output_dir}")
    print(f"   最大记录数: {max_records if max_records else '全部'}")
    print(f"   批次大小: {batch_size}")
    
    try:
        # 创建提取器
        extractor = ShennongCSVExtractor(csv_path, output_dir)
        
        # 启动控制面板
        control_thread = interactive_control(extractor)
        
        print("\n⏳ 开始提取...")
        
        # 开始提取
        results = extractor.extract_with_pause_control(max_records, batch_size)
        
        # 等待控制线程
        control_thread.join(timeout=1)
        
    except Exception as e:
        logger.error(f"提取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

