#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 暂停控制功能使用示例
演示如何使用暂停、恢复、停止功能
"""

import time
import threading
from src.graphrag_processor import GraphRAGProcessor
from src.config import SimpleConfigManager
from src.models import ProcessedDocument

def progress_callback(current, total, status):
    """进度回调函数"""
    percentage = (current / total * 100) if total > 0 else 0
    print(f"📊 进度: {current}/{total} ({percentage:.1f}%) - {status}")

def main():
    """主函数"""
    print("🚀 GraphRAG 暂停控制功能演示")
    print("=" * 50)
    
    # 1. 加载配置
    print("1. 加载配置...")
    config_manager = SimpleConfigManager()
    config = config_manager.load_config()
    
    # 2. 创建处理器
    print("2. 创建 GraphRAG 处理器...")
    processor = GraphRAGProcessor(config.graphrag)
    
    # 3. 准备测试文档
    print("3. 准备测试文档...")
    test_documents = []
    
    # 创建多个测试文档
    for i in range(5):
        content = f"""
        患者症状：全身乏力，短气，月经不调。
        证型：肾阴阳两虚证，脾虚气滞证。
        治法：温补肾阳，健脾益气。
        方药：桂枝{i+1}、白芍{i+1}、生姜{i+1}、大枣{i+1}、甘草{i+1}。
        诊断：通过望诊、闻诊、问诊、切诊进行辨证。
        文献：出自《伤寒论》和《金匮要略》。
        名医：张仲景创立了相关理论。
        """
        
        document = ProcessedDocument(
            title=f"测试医案_{i+1}",
            content=content
        )
        test_documents.append(document)
    
    print(f"✅ 创建了 {len(test_documents)} 个测试文档")
    
    # 4. 启动批量处理（在后台线程中）
    print("4. 启动批量处理...")
    results = []
    
    def run_batch_processing():
        nonlocal results
        results = processor.batch_extract_with_pause(
            test_documents, 
            progress_callback=progress_callback
        )
    
    # 在后台线程中运行处理
    processing_thread = threading.Thread(target=run_batch_processing)
    processing_thread.start()
    
    # 5. 模拟用户交互控制
    print("\n5. 用户控制选项:")
    print("   - 输入 'p' 暂停处理")
    print("   - 输入 'r' 恢复处理")
    print("   - 输入 's' 停止处理")
    print("   - 输入 'status' 查看状态")
    print("   - 输入 'q' 退出")
    print("   - 直接回车等待处理完成")
    
    while processing_thread.is_alive():
        try:
            user_input = input("\n请输入命令: ").strip().lower()
            
            if user_input == 'p':
                processor.pause_processing()
                print("⏸️  处理已暂停")
                
            elif user_input == 'r':
                processor.resume_processing()
                print("▶️  处理已恢复")
                
            elif user_input == 's':
                processor.stop_processing()
                print("⏹️  处理已停止")
                
            elif user_input == 'status':
                status = processor.get_processing_status()
                print(f"📊 当前状态: {status}")
                
            elif user_input == 'q':
                processor.stop_processing()
                print("👋 用户退出")
                break
                
            elif user_input == '':
                # 等待处理完成
                continue
                
            else:
                print("❌ 无效命令，请重新输入")
                
        except KeyboardInterrupt:
            processor.stop_processing()
            print("\n⏹️  用户中断，处理已停止")
            break
    
    # 6. 等待处理线程结束
    print("\n6. 等待处理完成...")
    processing_thread.join(timeout=10)
    
    # 7. 显示结果
    print("\n7. 处理结果:")
    print("=" * 30)
    
    if results:
        total_entities = sum(len(result.entities) for result in results)
        total_relationships = sum(len(result.relationships) for result in results)
        successful_docs = len([r for r in results if r.entities or r.relationships])
        
        print(f"✅ 成功处理文档: {successful_docs}/{len(test_documents)}")
        print(f"📊 总实体数: {total_entities}")
        print(f"🔗 总关系数: {total_relationships}")
        
        # 显示每个文档的详细信息
        for i, result in enumerate(results):
            status = result.metadata.get('status', '完成')
            print(f"\n📄 文档 {i+1}: {result.entities[0].source_document_id if result.entities else 'N/A'}")
            print(f"   实体: {len(result.entities)} 个")
            print(f"   关系: {len(result.relationships)} 个")
            print(f"   状态: {status}")
            print(f"   耗时: {result.processing_time:.2f} 秒")
    else:
        print("❌ 没有处理结果")
    
    print("\n🎉 演示完成！")

def single_document_example():
    """单文档处理示例"""
    print("\n" + "="*50)
    print("📄 单文档处理示例")
    print("="*50)
    
    # 加载配置
    config_manager = SimpleConfigManager()
    config = config_manager.load_config()
    processor = GraphRAGProcessor(config.graphrag)
    
    # 创建测试文档
    document = ProcessedDocument(
        title="单文档测试",
        content="""
        患者症状：头痛、发热、咳嗽。
        证型：风寒感冒证。
        治法：解表散寒。
        方药：麻黄汤加减。
        诊断：通过四诊合参进行辨证。
        文献：出自《伤寒论》。
        名医：张仲景创立。
        """
    )
    
    print("开始处理单文档...")
    
    # 在后台线程中处理
    result = None
    processing_thread = threading.Thread(
        target=lambda: setattr(single_document_example, 'result', 
                              processor.extract_entities_and_relationships(document))
    )
    processing_thread.start()
    
    # 模拟用户控制
    print("输入 'p' 暂停，'r' 恢复，'s' 停止")
    while processing_thread.is_alive():
        try:
            user_input = input("命令: ").strip().lower()
            if user_input == 'p':
                processor.pause_processing()
            elif user_input == 'r':
                processor.resume_processing()
            elif user_input == 's':
                processor.stop_processing()
                break
            elif user_input == 'status':
                print(f"状态: {processor.get_processing_status()}")
        except KeyboardInterrupt:
            processor.stop_processing()
            break
    
    processing_thread.join()
    result = getattr(single_document_example, 'result', None)
    
    if result:
        print(f"\n✅ 处理完成:")
        print(f"   实体: {len(result.entities)} 个")
        print(f"   关系: {len(result.relationships)} 个")
        print(f"   状态: {result.metadata.get('status', '完成')}")
        
        # 显示部分实体
        if result.entities:
            print(f"\n📋 提取的实体示例:")
            for entity in result.entities[:5]:
                print(f"   - {entity.name} ({entity.type}): {entity.description}")

if __name__ == "__main__":
    try:
        # 运行批量处理示例
        main()
        
        # 运行单文档示例
        single_document_example()
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
