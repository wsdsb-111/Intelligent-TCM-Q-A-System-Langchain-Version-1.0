"""
向量检索系统主程序 - 演示如何使用系统构建和查询向量数据库
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from vector_retrieval_system import VectorRetrieval, TCMVectorRetriever, create_tcm_retriever
from vector_retrieval_system.config import CHROMA_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_retrieval.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主程序入口"""
    try:
        logger.info("=" * 60)
        logger.info("中医向量检索系统启动")
        logger.info("=" * 60)
        
        # 显示菜单选项
        print("\n请选择操作模式:")
        print("1. 测试现有数据库（推荐）")
        print("2. 重新构建数据库")
        print("3. 批量测试功能")
        print("4. LangChain集成演示")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            test_existing_database()
        elif choice == '2':
            rebuild_database()
        elif choice == '3':
            batch_test()
        elif choice == '4':
            demo_langchain_integration()
        elif choice == '5':
            logger.info("退出程序")
            return
        else:
            logger.info("无效选择，默认进入测试模式")
            test_existing_database()
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        raise

def test_existing_database():
    """测试现有数据库"""
    try:
        logger.info("=" * 60)
        logger.info("测试现有数据库模式")
        logger.info("=" * 60)
        
        # 初始化向量检索系统
        logger.info("初始化向量检索系统...")
        vector_retrieval = VectorRetrieval(
            persist_directory=CHROMA_CONFIG["persist_directory"],
            collection_name=CHROMA_CONFIG["collection_name"]
        )
        
        # 检查数据库状态
        stats = vector_retrieval.get_database_stats()
        if stats.get("document_count", 0) > 0:
            logger.info(f"✅ 发现现有数据库，包含{stats['document_count']:,}个文档")
            logger.info("数据库统计信息:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            # 进入交互式查询
            interactive_query(vector_retrieval)
        else:
            logger.warning("❌ 未找到现有数据库")
            choice = input("是否构建新的数据库？(y/n): ").lower().strip()
            if choice == 'y':
                rebuild_database()
            else:
                logger.info("退出程序")
                
    except Exception as e:
        logger.error(f"测试现有数据库失败: {e}")

def rebuild_database():
    """重新构建数据库"""
    try:
        logger.info("=" * 60)
        logger.info("重新构建数据库模式")
        logger.info("=" * 60)
        logger.warning("⚠️  注意：此操作将删除现有数据库并重新构建")
        
        confirm = input("确定要重新构建数据库吗？(yes/no): ").lower().strip()
        if confirm != 'yes':
            logger.info("操作已取消")
            return
        
        # 初始化向量检索系统
        logger.info("初始化向量检索系统...")
        vector_retrieval = VectorRetrieval(
            persist_directory=CHROMA_CONFIG["persist_directory"],
            collection_name=CHROMA_CONFIG["collection_name"]
        )
        
        # 重新构建数据库
        logger.info("开始重新构建向量数据库...")
        success = vector_retrieval.build_vector_database()
        if not success:
            logger.error("数据库构建失败")
            return
        
        # 显示构建结果
        stats = vector_retrieval.get_database_stats()
        logger.info("✅ 数据库构建完成!")
        logger.info("数据库统计信息:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # 进入交互式查询
        interactive_query(vector_retrieval)
        
    except Exception as e:
        logger.error(f"重新构建数据库失败: {e}")

def interactive_query(vector_retrieval: VectorRetrieval):
    """交互式查询功能"""
    logger.info("\n" + "=" * 60)
    logger.info("进入交互式查询模式")
    logger.info("输入 'quit' 或 'exit' 退出程序")
    logger.info("输入 'stats' 查看数据库统计信息")
    logger.info("输入 'export <path>' 导出数据库")
    logger.info("输入 'test' 运行预设查询测试")
    logger.info("输入 'help' 显示帮助信息")
    logger.info("=" * 60)
    
    # 预设测试查询
    test_queries = [
        "感冒了吃什么药",
        "感冒的症状",
        "感冒的治疗方法",
        "风寒感冒",
        "风热感冒",
        "感冒咳嗽",
        "感冒发烧",
        "感冒头痛",
        "感冒流鼻涕",
        "感冒鼻塞"
    ]
    
    while True:
        try:
            query = input("\n请输入查询内容: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', '退出']:
                logger.info("退出程序")
                break
            
            if query.lower() == 'stats':
                stats = vector_retrieval.get_database_stats()
                logger.info("数据库统计信息:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
                continue
            
            if query.lower().startswith('export '):
                export_path = query[7:].strip()
                if export_path:
                    success = vector_retrieval.export_database(export_path)
                    if success:
                        logger.info(f"数据库已导出到: {export_path}")
                    else:
                        logger.error("数据库导出失败")
                continue
            
            if query.lower() == 'test':
                run_preset_test(vector_retrieval, test_queries)
                continue
            
            if query.lower() == 'help':
                show_help()
                continue
            
            # 执行查询
            logger.info(f"查询: {query}")
            results = vector_retrieval.search(query, top_k=5)
            
            if not results:
                logger.info("没有找到相关结果")
                continue
            
            logger.info(f"找到{len(results)}个相关结果:")
            for i, result in enumerate(results, 1):
                if result is None:
                    logger.warning(f"结果 {i}: 数据为None，跳过")
                    continue
                    
                logger.info(f"\n结果 {i}:")
                logger.info(f"  相似度: {result.get('score', 0):.4f}")
                logger.info(f"  来源: {result.get('metadata', {}).get('source', 'unknown')}")
                logger.info(f"  诊断: {result.get('metadata', {}).get('output', 'N/A')}")
                
                # 安全地获取文本内容
                text = result.get('text', '')
                if text:
                    display_text = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  内容: {display_text}")
                else:
                    logger.info(f"  内容: [无文本内容]")
        
        except KeyboardInterrupt:
            logger.info("\n程序被用户中断")
            break
        except Exception as e:
            logger.error(f"查询出错: {e}")

def run_preset_test(vector_retrieval: VectorRetrieval, test_queries: list):
    """运行预设查询测试"""
    logger.info("\n" + "=" * 60)
    logger.info("运行预设查询测试")
    logger.info("=" * 60)
    
    successful_queries = 0
    total_results = 0
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- 测试 {i}/{len(test_queries)}: {query} ---")
        
        try:
            results = vector_retrieval.search(query, top_k=3)
            
            if results:
                successful_queries += 1
                total_results += len(results)
                
                logger.info(f"✅ 查询成功，返回 {len(results)} 个结果")
                
                # 找到第一个有效的结果
                best_result = None
                for result in results:
                    if result is not None:
                        best_result = result
                        break
                
                if best_result is not None:
                    score = best_result.get('score', 0)
                    text = best_result.get('text', '')
                    metadata = best_result.get('metadata', {})
                    output = metadata.get('output', 'N/A')
                    
                    # 截断长文本
                    display_text = text[:150] + "..." if len(text) > 150 else text
                    display_output = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
                    
                    logger.info(f"  最佳结果 (相似度: {score:.4f}):")
                    logger.info(f"    内容: {display_text}")
                    logger.info(f"    诊断: {display_output}")
                else:
                    logger.warning("⚠️ 所有结果都为None")
            else:
                logger.warning("⚠️ 没有找到相关结果")
                
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
    
    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果统计")
    logger.info("=" * 60)
    
    success_rate = successful_queries / len(test_queries)
    avg_results = total_results / successful_queries if successful_queries > 0 else 0
    
    logger.info(f"总查询数: {len(test_queries)}")
    logger.info(f"成功查询数: {successful_queries}")
    logger.info(f"成功率: {success_rate:.2%}")
    logger.info(f"总结果数: {total_results}")
    logger.info(f"平均每个查询结果数: {avg_results:.1f}")
    
    if success_rate > 0.8:
        logger.info("🎉 查询功能测试优秀！")
    elif success_rate > 0.5:
        logger.info("✅ 查询功能测试良好！")
    else:
        logger.info("⚠️ 查询功能需要改进")

def show_help():
    """显示帮助信息"""
    logger.info("\n" + "=" * 60)
    logger.info("帮助信息")
    logger.info("=" * 60)
    logger.info("可用命令:")
    logger.info("  quit/exit    - 退出程序")
    logger.info("  stats        - 查看数据库统计信息")
    logger.info("  export <路径> - 导出数据库到指定路径")
    logger.info("  test         - 运行预设查询测试")
    logger.info("  help         - 显示此帮助信息")
    logger.info("")
    logger.info("直接输入查询内容即可进行搜索")
    logger.info("例如: 感冒了吃什么药")
    logger.info("=" * 60)

def demo_langchain_integration():
    """演示LangChain集成功能"""
    logger.info("\n" + "=" * 60)
    logger.info("LangChain集成演示")
    logger.info("=" * 60)
    
    try:
        # 创建LangChain兼容的检索器
        retriever = create_tcm_retriever()
        
        # 演示检索功能
        query = "湿疹的症状和治疗方法"
        logger.info(f"查询: {query}")
        
        documents = retriever.get_relevant_documents(query, top_k=3)
        logger.info(f"检索到{len(documents)}个文档:")
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"\n文档 {i}:")
            logger.info(f"  分数: {doc.get('score', 0):.4f}")
            logger.info(f"  内容: {doc['page_content'][:150]}...")
            logger.info(f"  元数据: {doc['metadata']}")
        
    except Exception as e:
        logger.error(f"LangChain集成演示失败: {e}")

def batch_test():
    """批量测试功能"""
    logger.info("\n" + "=" * 60)
    logger.info("批量测试")
    logger.info("=" * 60)
    
    try:
        vector_retrieval = VectorRetrieval()
        
        # 测试查询列表
        test_queries = [
            "湿疹的症状",
            "血虚风燥的治疗",
            "当归的功效",
            "慢性湿疹的方剂",
            "皮肤瘙痒的中医治疗"
        ]
        
        logger.info("执行批量查询测试...")
        results = vector_retrieval.batch_search(test_queries, top_k=3)
        
        for i, (query, query_results) in enumerate(zip(test_queries, results)):
            logger.info(f"\n查询 {i+1}: {query}")
            logger.info(f"  结果数量: {len(query_results)}")
            if query_results:
                best_score = query_results[0].get('score', 0)
                logger.info(f"  最高相似度: {best_score:.4f}")
        
    except Exception as e:
        logger.error(f"批量测试失败: {e}")

if __name__ == "__main__":
    main()
