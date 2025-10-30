#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理神农数据集 ChatMed_TCM-v0.2.json
将问答对转换为适合GraphRAG处理的格式
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
from datetime import datetime
import time

from src.graphrag_processor import GraphRAGProcessor
from src.config import SimpleConfigManager
from src.models import ProcessedDocument, GraphRAGResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shennong_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShennongDatasetProcessor:
    """神农数据集处理器"""
    
    def __init__(self, dataset_path: str, output_dir: str = "output"):
        """
        初始化处理器
        
        Args:
            dataset_path: 数据集文件路径
            output_dir: 输出目录
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载配置和处理器
        config_manager = SimpleConfigManager()
        self.config = config_manager.load_config()
        self.processor = GraphRAGProcessor(self.config.graphrag)
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'processed_documents': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"初始化神农数据集处理器")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def load_dataset(self, max_documents: int = None) -> Generator[Dict[str, Any], None, None]:
        """
        加载数据集
        
        Args:
            max_documents: 最大处理文档数，None表示处理全部
            
        Yields:
            Dict: 包含query和response的字典
        """
        logger.info(f"开始加载数据集: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        count = 0
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_documents and count >= max_documents:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if 'query' in data and 'response' in data:
                        yield data
                        count += 1
                    else:
                        logger.warning(f"第{line_num}行数据格式不正确，跳过")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行JSON解析错误: {e}")
                    continue
        
        logger.info(f"数据集加载完成，共{count}条记录")
    
    def convert_to_document(self, data: Dict[str, Any], index: int) -> ProcessedDocument:
        """
        将数据集记录转换为ProcessedDocument
        
        Args:
            data: 包含query和response的字典
            index: 文档索引
            
        Returns:
            ProcessedDocument: 处理后的文档
        """
        # 组合query和response作为文档内容
        content = f"问题：{data['query']}\n\n回答：{data['response']}"
        
        # 创建文档标题
        title = f"神农问答_{index:06d}"
        
        # 提取前100个字符作为摘要（放在metadata中）
        summary = content[:100] + "..." if len(content) > 100 else content
        
        return ProcessedDocument(
            title=title,
            content=content,
            file_path=f"shennong_dataset_{index:06d}",  # 使用file_path代替source
            file_type="json",
            metadata={
                'original_query': data['query'],
                'original_response': data['response'],
                'dataset_index': index,
                'source_file': str(self.dataset_path),
                'summary': summary,
                'source': f"shennong_dataset_{index:06d}"
            }
        )
    
    def save_results(self, results: List[GraphRAGResult], batch_num: int):
        """
        保存提取结果
        
        Args:
            results: 提取结果列表
            batch_num: 批次号
        """
        # 保存实体到CSV
        entities_file = self.output_dir / f"entities_batch_{batch_num:03d}.csv"
        relationships_file = self.output_dir / f"relationships_batch_{batch_num:03d}.csv"
        
        # 收集所有实体和关系
        all_entities = []
        all_relationships = []
        
        for result in results:
            if result.entities:
                all_entities.extend(result.entities)
            if result.relationships:
                all_relationships.extend(result.relationships)
        
        # 保存实体
        if all_entities:
            with open(entities_file, 'w', encoding='utf-8', newline='') as f:
                f.write("id,name,type,description,source_document_id\n")
                for entity in all_entities:
                    f.write(f'"{entity.id}","{entity.name}","{entity.type}","{entity.description}","{entity.source_document_id}"\n')
            logger.info(f"实体已保存到: {entities_file}")
        
        # 保存关系
        if all_relationships:
            with open(relationships_file, 'w', encoding='utf-8', newline='') as f:
                f.write("id,source_entity_id,target_entity_id,relationship_type,description,source_document_id\n")
                for rel in all_relationships:
                    f.write(f'"{rel.id}","{rel.source_entity_id}","{rel.target_entity_id}","{rel.type}","{rel.description}","{rel.source_document_id}"\n')
            logger.info(f"关系已保存到: {relationships_file}")
        
        # 保存统计信息
        stats_file = self.output_dir / f"stats_batch_{batch_num:03d}.json"
        batch_stats = {
            'batch_num': batch_num,
            'processed_documents': len(results),
            'total_entities': len(all_entities),
            'total_relationships': len(all_relationships),
            'processing_time': sum(r.processing_time for r in results),
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'document_id': r.document_id,
                    'entity_count': len(r.entities),
                    'relationship_count': len(r.relationships),
                    'processing_time': r.processing_time,
                    'metadata': r.metadata
                }
                for r in results
            ]
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_file}")
    
    def process_batch(self, documents: List[ProcessedDocument], batch_num: int) -> List[GraphRAGResult]:
        """
        处理一批文档
        
        Args:
            documents: 文档列表
            batch_num: 批次号
            
        Returns:
            List[GraphRAGResult]: 处理结果列表
        """
        logger.info(f"开始处理批次 {batch_num}，共 {len(documents)} 个文档")
        
        results = []
        
        for i, document in enumerate(documents):
            try:
                logger.info(f"处理文档 {i+1}/{len(documents)}: {document.title}")
                
                # 提取实体和关系
                result = self.processor.extract_entities_and_relationships(document)
                results.append(result)
                
                # 更新统计
                self.stats['processed_documents'] += 1
                if result.entities or result.relationships:
                    self.stats['successful_extractions'] += 1
                    self.stats['total_entities'] += len(result.entities)
                    self.stats['total_relationships'] += len(result.relationships)
                else:
                    self.stats['failed_extractions'] += 1
                
                logger.info(f"文档处理完成: 实体{len(result.entities)}个, 关系{len(result.relationships)}个")
                
            except Exception as e:
                logger.error(f"处理文档 {document.title} 失败: {e}")
                self.stats['failed_extractions'] += 1
                
                # 创建错误结果
                error_result = GraphRAGResult(
                    document_id=document.id,
                    entities=[],
                    relationships=[],
                    processing_time=0,
                    metadata={'error': str(e), 'status': '处理失败'}
                )
                results.append(error_result)
            
            # 短暂休息
            time.sleep(0.1)
        
        # 保存批次结果
        self.save_results(results, batch_num)
        
        logger.info(f"批次 {batch_num} 处理完成")
        return results
    
    def process_with_pause_control(self, max_documents: int = None, batch_size: int = 10):
        """
        使用暂停控制功能处理数据集
        
        Args:
            max_documents: 最大处理文档数
            batch_size: 批次大小
        """
        logger.info("开始使用暂停控制功能处理数据集")
        
        self.stats['start_time'] = datetime.now()
        
        # 重置处理器状态
        self.processor.reset_processing_state()
        
        # 加载数据集
        dataset = list(self.load_dataset(max_documents))
        self.stats['total_documents'] = len(dataset)
        
        logger.info(f"总共需要处理 {len(dataset)} 个文档，每批 {batch_size} 个")
        
        # 分批处理
        batch_num = 1
        all_results = []
        
        for i in range(0, len(dataset), batch_size):
            # 检查是否应该停止
            if self.processor.pause_controller.should_stop():
                logger.info("处理被用户停止")
                break
            
            # 等待暂停恢复
            self.processor.pause_controller.wait_if_paused()
            
            # 获取当前批次
            batch_data = dataset[i:i + batch_size]
            
            # 转换为文档
            documents = []
            for j, data in enumerate(batch_data):
                doc = self.convert_to_document(data, i + j)
                documents.append(doc)
            
            # 处理批次
            batch_results = self.process_batch(documents, batch_num)
            all_results.extend(batch_results)
            
            batch_num += 1
            
            # 显示进度
            progress = (i + len(batch_data)) / len(dataset) * 100
            logger.info(f"总体进度: {progress:.1f}% ({i + len(batch_data)}/{len(dataset)})")
            
            # 短暂休息
            time.sleep(0.5)
        
        self.stats['end_time'] = datetime.now()
        
        # 保存最终统计
        self.save_final_stats(all_results)
        
        logger.info("数据集处理完成")
        return all_results
    
    def save_final_stats(self, all_results: List[GraphRAGResult]):
        """保存最终统计信息"""
        final_stats = {
            'processing_summary': self.stats.copy(),
            'final_results': {
                'total_results': len(all_results),
                'successful_results': len([r for r in all_results if r.entities or r.relationships]),
                'failed_results': len([r for r in all_results if not (r.entities or r.relationships)]),
                'total_entities': sum(len(r.entities) for r in all_results),
                'total_relationships': sum(len(r.relationships) for r in all_results),
                'total_processing_time': sum(r.processing_time for r in all_results)
            },
            'entity_types': {},
            'relationship_types': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 统计实体类型
        for result in all_results:
            for entity in result.entities:
                entity_type = entity.type
                if entity_type not in final_stats['entity_types']:
                    final_stats['entity_types'][entity_type] = 0
                final_stats['entity_types'][entity_type] += 1
        
        # 统计关系类型
        for result in all_results:
            for rel in result.relationships:
                rel_type = rel.type
                if rel_type not in final_stats['relationship_types']:
                    final_stats['relationship_types'][rel_type] = 0
                final_stats['relationship_types'][rel_type] += 1
        
        # 保存统计文件
        stats_file = self.output_dir / "final_processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最终统计信息已保存到: {stats_file}")
        
        # 打印统计摘要
        print("\n" + "="*60)
        print("🎉 神农数据集处理完成！")
        print("="*60)
        print(f"📊 处理统计:")
        print(f"   总文档数: {self.stats['total_documents']}")
        print(f"   已处理: {self.stats['processed_documents']}")
        print(f"   成功提取: {self.stats['successful_extractions']}")
        print(f"   提取失败: {self.stats['failed_extractions']}")
        print(f"   总实体数: {self.stats['total_entities']}")
        print(f"   总关系数: {self.stats['total_relationships']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            print(f"   总耗时: {total_time:.2f} 秒")
            if self.stats['processed_documents'] > 0:
                avg_time = total_time / self.stats['processed_documents']
                print(f"   平均每文档: {avg_time:.2f} 秒")
        
        print(f"\n📁 输出目录: {self.output_dir}")
        print("="*60)


def interactive_control(processor: ShennongDatasetProcessor):
    """交互式控制函数"""
    import threading
    
    print("\n🎮 交互式控制面板")
    print("="*40)
    print("命令:")
    print("  p - 暂停处理")
    print("  r - 恢复处理") 
    print("  s - 停止处理")
    print("  status - 查看状态")
    print("  stats - 查看统计")
    print("  q - 退出")
    print("="*40)
    
    def control_loop():
        while True:
            try:
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'p':
                    processor.processor.pause_processing()
                    print("⏸️  处理已暂停")
                    
                elif cmd == 'r':
                    processor.processor.resume_processing()
                    print("▶️  处理已恢复")
                    
                elif cmd == 's':
                    processor.processor.stop_processing()
                    print("⏹️  处理已停止")
                    
                elif cmd == 'status':
                    status = processor.processor.get_processing_status()
                    print(f"📊 当前状态: {status}")
                    
                elif cmd == 'stats':
                    print(f"📈 处理统计:")
                    print(f"   已处理: {processor.stats['processed_documents']}/{processor.stats['total_documents']}")
                    print(f"   成功: {processor.stats['successful_extractions']}")
                    print(f"   失败: {processor.stats['failed_extractions']}")
                    print(f"   实体: {processor.stats['total_entities']}")
                    print(f"   关系: {processor.stats['total_relationships']}")
                    
                elif cmd == 'q':
                    processor.processor.stop_processing()
                    print("👋 退出控制面板")
                    break
                    
                else:
                    print("❌ 无效命令")
                    
            except KeyboardInterrupt:
                processor.processor.stop_processing()
                print("\n⏹️  用户中断")
                break
    
    # 在后台线程中运行控制循环
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()
    
    return control_thread


def main():
    """主函数"""
    print("🚀 神农数据集 GraphRAG 处理系统")
    print("="*50)
    
    # 配置参数
    dataset_path = r"E:\毕业论文和设计\线上智能中医问答项目\检索与知识层\Graphrag\dataset\shennong\ChatMed_TCM-v0.2.json"
    output_dir = "output_shennong"
    max_documents = 100  # 限制处理文档数，用于测试
    batch_size = 5  # 批次大小
    
    try:
        # 创建处理器
        processor = ShennongDatasetProcessor(dataset_path, output_dir)
        
        # 启动交互式控制
        control_thread = interactive_control(processor)
        
        print(f"\n📋 处理配置:")
        print(f"   数据集: {dataset_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   最大文档数: {max_documents}")
        print(f"   批次大小: {batch_size}")
        print(f"\n⏳ 开始处理...")
        
        # 开始处理
        results = processor.process_with_pause_control(max_documents, batch_size)
        
        # 等待控制线程结束
        control_thread.join(timeout=1)
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
