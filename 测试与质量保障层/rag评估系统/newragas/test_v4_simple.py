#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HybridRagasEvaluatorV4 简单测试脚本
测试优化的RAG流程和内存管理
"""

import asyncio
import sys
import json
import time
import io
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 修复Windows控制台编码问题
if sys.platform.startswith('win'):
    # 配置日志以支持UTF-8，但不修改sys.stdout
    class UTF8StreamHandler(logging.StreamHandler):
        def __init__(self, stream=None):
            if stream is None:
                stream = sys.stdout
            super().__init__(stream)
        
        def emit(self, record):
            try:
                msg = self.format(record)
                # 确保消息以UTF-8编码输出
                if hasattr(self.stream, 'buffer'):
                    self.stream.buffer.write(msg.encode('utf-8'))
                    self.stream.buffer.write(b'\n')
                    self.stream.buffer.flush()
                else:
                    self.stream.write(msg + '\n')
                    self.stream.flush()
            except Exception:
                self.handleError(record)
    
    # 重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[UTF8StreamHandler()]
    )

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "应用协调层"))

try:
    from hybrid_ragas_evaluator_v4 import HybridRagasEvaluatorV4
    print("✅ 成功导入 HybridRagasEvaluatorV4")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查 hybrid_ragas_evaluator_v4.py 文件是否存在")
    sys.exit(1)

# 创建结果目录
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_questions_from_dataset(dataset_path: str, num_questions: int = None, start_index: int = 0) -> list:
    """从数据集中加载指定数量的问题"""
    try:
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            print(f"❌ 数据集文件不存在: {dataset_file}")
            return []
        
        questions = []
        skipped_bad_data = 0
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_index:
                    continue
                
                if num_questions and len(questions) >= num_questions:
                    break
                
                try:
                    data = json.loads(line.strip())
                    if 'messages' in data and len(data['messages']) >= 3:
                        user_msg = data['messages'][1]
                        if (user_msg.get('role') == 'user' and 
                            user_msg.get('content') != "无"):  # 过滤掉问题内容为"无"的文档
                            question = user_msg.get('content', '')
                            questions.append({
                                'question': question,
                                'question_id': i + 1,
                                'dataset_index': i
                            })
                        else:
                            skipped_bad_data += 1
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 从数据集加载了 {len(questions)} 个问题")
        if skipped_bad_data > 0:
            print(f"   跳过了 {skipped_bad_data} 个问题内容为'无'的文档")
        if start_index > 0:
            print(f"   起始索引: {start_index}")
        if num_questions:
            print(f"   请求数量: {num_questions}")
        
        return questions
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return []

def load_existing_results(filepath: str) -> dict:
    """加载已有的测试结果文件"""
    try:
        result_file = Path(filepath)
        if not result_file.exists():
            print(f"❌ 结果文件不存在: {result_file}")
            return None
        
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        print(f"✅ 成功加载已有结果文件: {result_file}")
        print(f"   已有结果数量: {len(existing_results.get('results', []))}")
        
        return existing_results
        
    except Exception as e:
        print(f"❌ 加载已有结果失败: {e}")
        return None

def find_latest_result_file(pattern: str = "v4_完整流程_results_*.json") -> str:
    """查找最新的结果文件"""
    try:
        result_files = list(RESULTS_DIR.glob(pattern))
        if not result_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
        
    except Exception as e:
        print(f"❌ 查找结果文件失败: {e}")
        return None

def is_question_completed(question_data: dict, existing_results: dict) -> bool:
    """检查问题是否已完成"""
    if not existing_results or 'results' not in existing_results:
        return False
    
    question_id = question_data.get('question_id')
    dataset_index = question_data.get('dataset_index')
    
    for result in existing_results['results']:
        # 通过question_id或dataset_index匹配
        if (result.get('question_id') == question_id or 
            result.get('dataset_index') == dataset_index):
            # 检查状态是否为成功
            if result.get('status') == 'success':
                return True
    
    return False

def get_completed_question_ids(existing_results: dict) -> set:
    """获取已完成的问题ID集合"""
    completed_ids = set()
    
    if not existing_results or 'results' not in existing_results:
        return completed_ids
    
    for result in existing_results['results']:
        if result.get('status') == 'success':
            question_id = result.get('question_id')
            dataset_index = result.get('dataset_index')
            if question_id:
                completed_ids.add(('question_id', question_id))
            if dataset_index is not None and dataset_index >= 0:
                completed_ids.add(('dataset_index', dataset_index))
    
    return completed_ids

def save_test_results(test_name: str, results: dict, filepath: str = None):
    """保存测试结果到JSON文件（实时保存）"""
    try:
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v4_{test_name}_results_{timestamp}.json"
            filepath = RESULTS_DIR / filename
        else:
            # 确保filepath是Path对象
            filepath = Path(filepath)
        
        print(f"🔍 调试信息:")
        print(f"   - 文件路径: {filepath}")
        print(f"   - 文件路径类型: {type(filepath)}")
        print(f"   - 结果目录存在: {RESULTS_DIR.exists()}")
        print(f"   - 结果目录路径: {RESULTS_DIR}")
        
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 验证文件是否真的被创建
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"💾 测试结果已实时保存到: {filepath}")
            print(f"   文件大小: {file_size} 字节")
        else:
            print(f"❌ 文件创建失败，文件不存在: {filepath}")
        
        return str(filepath)
        
    except Exception as e:
        print(f"❌ 保存测试结果失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_test_summary(test_results: dict) -> dict:
    """生成测试总结"""
    try:
        results = test_results["results"]
        total_questions = len(results)
        successful_questions = len([r for r in results if r.get("status") == "success"])
        failed_questions = total_questions - successful_questions
        
        # 计算平均处理时间
        durations = [r.get("duration", 0) for r in results if r.get("status") == "success"]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # 计算RAGAS评估分数
        ragas_scores = []
        for r in results:
            if r.get("status") == "success" and "ragas_result" in r:
                ragas_result = r["ragas_result"]
                if ragas_result.get("status") == "success" and "evaluation_data" in ragas_result:
                    eval_data = ragas_result["evaluation_data"]
                    ragas_scores.append({
                        "context_precision": eval_data.get("context_precision", 0),
                        "context_recall": eval_data.get("context_recall", 0),
                        "faithfulness": eval_data.get("faithfulness", 0),
                        "answer_relevancy": eval_data.get("answer_relevancy", 0),
                        "overall_score": eval_data.get("overall_score", 0)
                    })
        
        # 计算平均RAGAS分数
        avg_ragas_scores = {}
        if ragas_scores:
            for key in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "overall_score"]:
                avg_ragas_scores[key] = sum(score[key] for score in ragas_scores) / len(ragas_scores)
        
        # 内存管理统计
        memory_management_stats = {
            "good": len([r for r in results if r.get("memory_management") == "good"]),
            "warning": len([r for r in results if r.get("memory_management") == "warning"])
        }
        
        return {
            "total_questions": total_questions,
            "successful_questions": successful_questions,
            "failed_questions": failed_questions,
            "success_rate": successful_questions / total_questions if total_questions > 0 else 0,
            "average_duration": avg_duration,
            "average_ragas_scores": avg_ragas_scores,
            "memory_management_stats": memory_management_stats,
            "route_type_distribution": {
                "vector": len([r for r in results if r.get("rag_result", {}).get("route_type") == "vector"]),
                "hybrid": len([r for r in results if r.get("rag_result", {}).get("route_type") == "hybrid"])
            }
        }
        
    except Exception as e:
        print(f"❌ 生成测试总结失败: {e}")
        return {}

async def simple_test(num_questions: int = 3, start_index: int = 0, dataset_path: str = None, custom_questions: list = None, resume_file: str = None):
    """完整流程测试（支持断点重续）"""
    try:
        print("🚀 启动HybridRagasEvaluatorV4完整流程测试")
        print("=" * 80)
        
        # 检查是否需要断点重续
        existing_results = None
        log_filepath = None
        
        if resume_file:
            # 使用指定的结果文件进行断点重续
            existing_results = load_existing_results(resume_file)
            if existing_results:
                log_filepath = Path(resume_file)
                print(f"🔄 断点重续模式：从 {log_filepath} 继续")
            else:
                print(f"⚠️ 无法加载指定的结果文件，将创建新文件")
                resume_file = None
        elif not custom_questions:
            # 自动查找最新的结果文件（仅当使用数据集时）
            latest_file = find_latest_result_file()
            if latest_file:
                # 检查文件是否有未完成的任务（没有end_time或summary为空）
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        temp_results = json.load(f)
                    # 检查是否已完成（有end_time和summary）
                    if not temp_results.get('end_time') or not temp_results.get('summary'):
                        existing_results = temp_results
                        log_filepath = Path(latest_file)
                        print(f"🔄 发现未完成的测试，自动从 {log_filepath} 继续")
                    else:
                        print(f"ℹ️ 发现已完成的测试文件: {latest_file}")
                        print(f"   将创建新的测试文件")
                except:
                    pass
        
        # 确定测试问题来源
        if custom_questions:
            # 使用自定义问题
            test_questions = [{"question": q, "question_id": i+1, "dataset_index": -1} for i, q in enumerate(custom_questions)]
            print(f"📝 使用自定义问题: {len(test_questions)} 个")
        elif dataset_path:
            # 从数据集加载问题
            test_questions = load_questions_from_dataset(dataset_path, num_questions, start_index)
            if not test_questions:
                print("❌ 无法从数据集加载问题，使用默认问题")
                test_questions = [
                    {"question": "我恶寒感冒，可以给我推荐一个中药吗？", "question_id": 1, "dataset_index": -1},
                    {"question": "口臭是什么原因引起的？", "question_id": 2, "dataset_index": -1},
                    {"question": "失眠多梦应该怎么调理？", "question_id": 3, "dataset_index": -1}
                ]
        else:
            # 使用默认问题
            test_questions = [
                {"question": "我恶寒感冒，可以给我推荐一个中药吗？", "question_id": 1, "dataset_index": -1},
                {"question": "口臭是什么原因引起的？", "question_id": 2, "dataset_index": -1},
                {"question": "失眠多梦应该怎么调理？", "question_id": 3, "dataset_index": -1}
            ]
            print(f"📝 使用默认问题: {len(test_questions)} 个")
        
        # 如果存在已有结果，加载它并过滤已完成的问题
        if existing_results:
            # 使用已有结果作为基础
            test_results = existing_results.copy()
            # 确保start_time存在（如果不存在则添加）
            if 'start_time' not in test_results:
                test_results['start_time'] = datetime.now().isoformat()
            
            # 获取已完成的问题ID
            completed_ids = get_completed_question_ids(existing_results)
            
            # 过滤掉已完成的问题
            original_count = len(test_questions)
            test_questions = [
                q for q in test_questions 
                if not is_question_completed(q, existing_results)
            ]
            skipped_count = original_count - len(test_questions)
            
            if skipped_count > 0:
                print(f"⏭️  跳过 {skipped_count} 个已完成的问题")
                print(f"📊 剩余 {len(test_questions)} 个问题需要处理")
            
            if len(test_questions) == 0:
                print("✅ 所有问题都已完成，无需继续处理")
                return
        else:
            # 初始化新的测试结果
            test_results = {
                "test_name": "完整流程测试",
                "start_time": datetime.now().isoformat(),
                "questions": [q["question"] for q in test_questions],  # 只保存问题文本
                "question_details": test_questions,  # 保存完整的问题详情
                "results": [],
                "summary": {}
            }
            
            # 创建新的日志文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"v4_完整流程_results_{timestamp}.json"
            log_filepath = RESULTS_DIR / log_filename
        
        print(f"📁 日志文件: {log_filepath}")
        print(f"📊 本次需要处理的问题数量: {len(test_questions)}")
        
        # 保存初始状态
        if not existing_results:
            print("💾 保存初始测试状态...")
            try:
                save_test_results("完整流程", test_results, str(log_filepath))
                print(f"✅ 初始状态保存成功: {log_filepath}")
            except Exception as e:
                print(f"❌ 初始状态保存失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("💾 更新测试状态...")
            try:
                save_test_results("完整流程", test_results, str(log_filepath))
                print(f"✅ 测试状态更新成功")
            except Exception as e:
                print(f"❌ 测试状态更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 创建评估器
        evaluator = HybridRagasEvaluatorV4()
        
        # 计算已完成的问题数量（用于显示正确的序号）
        completed_count = len(test_results.get('results', [])) if existing_results else 0
        
        for i, question_data in enumerate(test_questions, 1):
            question = question_data["question"]
            question_id = question_data["question_id"]
            dataset_index = question_data["dataset_index"]
            
            # 显示当前问题序号（包括已完成的问题）
            current_question_num = completed_count + i
            
            print(f"\n{'='*60}")
            print(f"测试问题 {current_question_num}/{completed_count + len(test_questions)}: {question}")
            if dataset_index >= 0:
                print(f"数据集索引: {dataset_index}")
            print(f"{'='*60}")
            
            # 记录问题开始时间
            question_start_time = time.time()
            
            # 显示初始内存状态
            memory_before = evaluator.get_memory_usage()
            print(f"📊 初始内存状态: GPU {memory_before['gpu_allocated']:.2f}GB, CPU {memory_before['cpu_memory']:.2f}GB")
            
            # 执行完整评估流程（RAG处理 + RAGAS评估）
            print("🔄 执行完整评估流程...")
            result = await evaluator.full_evaluation_pipeline(question)
            
            # 记录问题结束时间
            question_end_time = time.time()
            question_duration = question_end_time - question_start_time
            
            # 初始化问题结果
            question_result = {
                "question_id": question_id,
                "dataset_index": dataset_index,
                "question": question,
                "start_time": datetime.fromtimestamp(question_start_time).isoformat(),
                "end_time": datetime.fromtimestamp(question_end_time).isoformat(),
                "duration": question_duration,
                "memory_before": memory_before,
                "status": "failed"
            }
            
            # 显示结果
            if result.get("status") == "success":
                rag_result = result["rag_result"]
                ragas_result = result["ragas_result"]
                
                print(f"✅ 路由类型: {rag_result['route_type']}")
                print(f"✅ 置信度: {rag_result['confidence']:.2f}")
                print(f"✅ 答案长度: {len(rag_result['answer'])} 字符")
                print(f"✅ 生成文档数量: {len(rag_result['contexts'])}")
                print(f"✅ 评估文档数量: {len(rag_result['evaluation_contexts'])}")
                print(f"✅ 总处理时间: {result['total_processing_time']:.2f}秒")
                print()
                
                # 显示RAGAS评估结果
                if ragas_result.get("status") == "success":
                    eval_data = ragas_result["evaluation_data"]
                    print(f"📊 RAGAS评估结果:")
                    print(f"  - 上下文精确度: {eval_data.get('context_precision', 0):.2f}")
                    print(f"  - 上下文召回率: {eval_data.get('context_recall', 0):.2f}")
                    print(f"  - 忠实度: {eval_data.get('faithfulness', 0):.2f}")
                    print(f"  - 答案相关性: {eval_data.get('answer_relevancy', 0):.2f}")
                    print(f"  - 总体分数: {eval_data.get('overall_score', 0):.2f}")
                    print()
                else:
                    print(f"❌ RAGAS评估失败: {ragas_result.get('error', '未知错误')}")
                    print()
                
                # 显示答案预览
                answer_preview = rag_result['answer'][:200] + "..." if len(rag_result['answer']) > 200 else rag_result['answer']
                print(f"📝 答案预览: {answer_preview}")
                
                # 显示Ground Truth预览
                if 'ground_truth' in ragas_result:
                    ground_truth_preview = ragas_result['ground_truth'][:200] + "..." if len(ragas_result['ground_truth']) > 200 else ragas_result['ground_truth']
                    print(f"🎯 Ground Truth预览: {ground_truth_preview}")
                print()
                
                # 显示生成文档预览
                print(f"📚 生成文档预览:")
                for j, context in enumerate(rag_result['contexts'][:3], 1):
                    context_preview = context[:80] + "..." if len(context) > 80 else context
                    print(f"  {j}. {context_preview}")
                print()
                
                # 更新问题结果
                question_result.update({
                    "status": "success",
                    "rag_result": rag_result,
                    "ragas_result": ragas_result,
                    "answer_preview": answer_preview,
                    "ground_truth_preview": ground_truth_preview if 'ground_truth' in ragas_result else "",
                    "context_previews": [context[:80] + "..." if len(context) > 80 else context for context in rag_result['contexts'][:3]]
                })
                
            else:
                print(f"❌ 评估失败: {result.get('error', '未知错误')}")
                question_result["error"] = result.get('error', '未知错误')
                test_results["results"].append(question_result)
                
                # 即使失败也要实时保存
                print(f"💾 保存问题 {current_question_num} 的测试结果（失败）...")
                try:
                    save_test_results("完整流程", test_results, str(log_filepath))
                    print(f"✅ 问题 {current_question_num} 结果保存成功（失败）")
                except Exception as e:
                    print(f"❌ 问题 {current_question_num} 结果保存失败: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            # 显示最终内存状态
            memory_after = evaluator.get_memory_usage()
            print(f"📊 最终内存状态: GPU {memory_after['gpu_allocated']:.2f}GB, CPU {memory_after['cpu_memory']:.2f}GB")
            
            # 显示组件状态
            component_status = evaluator.get_component_status()
            component_status_str = [f'{k}:{v["state"]}' for k, v in component_status.items()]
            print(f"🔧 组件状态: {component_status_str}")
            
            # 分析内存变化
            gpu_change = memory_after['gpu_allocated'] - memory_before['gpu_allocated']
            cpu_change = memory_after['cpu_memory'] - memory_before['cpu_memory']
            
            print(f"📈 内存变化: GPU {gpu_change:+.2f}GB, CPU {cpu_change:+.2f}GB")
            
            if abs(gpu_change) < 0.1 and abs(cpu_change) < 0.1:
                print("✅ 内存管理良好，组件已正确卸载")
                memory_management = "good"
            else:
                print("⚠️ 内存管理可能存在问题，组件未完全卸载")
                memory_management = "warning"
            
            # 更新问题结果
            question_result.update({
                "memory_after": memory_after,
                "memory_change": {
                    "gpu_change": gpu_change,
                    "cpu_change": cpu_change
                },
                "component_status": component_status,
                "memory_management": memory_management
            })
            
            # 添加到结果列表
            test_results["results"].append(question_result)
            
            # 实时保存当前进度
            print(f"💾 保存问题 {current_question_num} 的测试结果...")
            try:
                save_test_results("完整流程", test_results, str(log_filepath))
                print(f"✅ 问题 {current_question_num} 结果保存成功")
            except Exception as e:
                print(f"❌ 问题 {current_question_num} 结果保存失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成测试总结
        test_results["end_time"] = datetime.now().isoformat()
        test_results["summary"] = generate_test_summary(test_results)
        
        # 最终保存测试结果
        print(f"💾 保存最终测试结果...")
        save_test_results("完整流程", test_results, str(log_filepath))
        
        print(f"\n{'='*80}")
        print("✅ 完整流程测试完成")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_memory_management():
    """测试内存管理"""
    try:
        print("\n" + "="*80)
        print("🧪 测试内存管理")
        print("="*80)
        
        # 初始化内存管理测试结果
        memory_test_results = {
            "test_name": "内存管理测试",
            "start_time": datetime.now().isoformat(),
            "question": "我恶寒感冒，可以给我推荐一个中药吗？",
            "stages": []
        }
        
        # 立即创建并保存初始日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        memory_log_filename = f"v4_内存管理_results_{timestamp}.json"
        memory_log_filepath = RESULTS_DIR / memory_log_filename
        print(f"📁 创建内存管理日志文件: {memory_log_filepath}")
        
        # 保存初始状态
        save_test_results("内存管理", memory_test_results, str(memory_log_filepath))
        
        evaluator = HybridRagasEvaluatorV4()
        
        # 测试问题
        question = "我恶寒感冒，可以给我推荐一个中药吗？"
        
        print(f"测试问题: {question}")
        
        # 显示初始状态
        memory_before = evaluator.get_memory_usage()
        component_status_before = evaluator.get_component_status()
        
        print(f"\n📊 初始状态:")
        print(f"  GPU显存: {memory_before['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_before['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_before.items()]
        print(f"  组件状态: {component_status_str}")
        
        # 记录初始状态
        memory_test_results["stages"].append({
            "stage": "初始状态",
            "timestamp": datetime.now().isoformat(),
            "memory": memory_before,
            "component_status": component_status_before
        })
        
        # 实时保存初始状态
        print("💾 保存初始状态...")
        save_test_results("内存管理", memory_test_results, str(memory_log_filepath))
        
        # 执行RAG处理
        print(f"\n🔄 执行RAG处理...")
        rag_start_time = time.time()
        rag_result = await evaluator.process_question(question)
        rag_end_time = time.time()
        
        # 显示RAG处理后的状态
        memory_after_rag = evaluator.get_memory_usage()
        component_status_after_rag = evaluator.get_component_status()
        
        print(f"\n📊 RAG处理后的状态:")
        print(f"  GPU显存: {memory_after_rag['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_after_rag['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_after_rag.items()]
        print(f"  组件状态: {component_status_str}")
        
        # 记录RAG处理后的状态
        memory_test_results["stages"].append({
            "stage": "RAG处理后",
            "timestamp": datetime.now().isoformat(),
            "memory": memory_after_rag,
            "component_status": component_status_after_rag,
            "rag_result": {
                "status": rag_result.get("status"),
                "processing_time": rag_result.get("processing_time", 0),
                "answer_length": len(rag_result.get("answer", "")) if rag_result.get("status") == "success" else 0
            }
        })
        
        # 实时保存RAG处理后的状态
        print("💾 保存RAG处理后的状态...")
        save_test_results("内存管理", memory_test_results, str(memory_log_filepath))
        
        if rag_result.get("status") == "success":
            print(f"✅ RAG处理成功")
            print(f"  答案长度: {len(rag_result['answer'])} 字符")
            print(f"  处理时间: {rag_result['processing_time']:.2f}秒")
        else:
            print(f"❌ RAG处理失败: {rag_result.get('error', '未知错误')}")
            memory_test_results["error"] = rag_result.get('error', '未知错误')
            save_test_results("内存管理", memory_test_results)
            return
        
        # 等待一段时间，观察内存是否被正确释放
        print(f"\n⏳ 等待5秒，观察内存释放...")
        await asyncio.sleep(5)
        
        # 显示最终状态
        memory_final = evaluator.get_memory_usage()
        component_status_final = evaluator.get_component_status()
        
        print(f"\n📊 最终状态:")
        print(f"  GPU显存: {memory_final['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_final['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_final.items()]
        print(f"  组件状态: {component_status_str}")
        
        # 记录最终状态
        memory_test_results["stages"].append({
            "stage": "最终状态",
            "timestamp": datetime.now().isoformat(),
            "memory": memory_final,
            "component_status": component_status_final
        })
        
        # 实时保存最终状态
        print("💾 保存最终状态...")
        save_test_results("内存管理", memory_test_results, str(memory_log_filepath))
        
        # 分析内存变化
        gpu_change = memory_final['gpu_allocated'] - memory_before['gpu_allocated']
        cpu_change = memory_final['cpu_memory'] - memory_before['cpu_memory']
        
        print(f"\n📈 内存变化分析:")
        print(f"  GPU显存变化: {gpu_change:+.2f}GB")
        print(f"  CPU内存变化: {cpu_change:+.2f}GB")
        
        memory_management_status = "good" if abs(gpu_change) < 0.1 and abs(cpu_change) < 0.1 else "warning"
        
        if memory_management_status == "good":
            print("✅ 内存管理良好，组件已正确卸载")
        else:
            print("⚠️ 内存管理可能存在问题，组件未完全卸载")
        
        # 完成测试结果
        memory_test_results.update({
            "end_time": datetime.now().isoformat(),
            "memory_analysis": {
                "gpu_change": gpu_change,
                "cpu_change": cpu_change,
                "memory_management_status": memory_management_status
            },
            "summary": {
                "total_duration": time.time() - rag_start_time,
                "memory_management_status": memory_management_status,
                "gpu_memory_leak": abs(gpu_change) >= 0.1,
                "cpu_memory_leak": abs(cpu_change) >= 0.1
            }
        })
        
        # 最终保存内存管理测试结果
        print("💾 保存最终内存管理测试结果...")
        save_test_results("内存管理", memory_test_results, str(memory_log_filepath))
        
        print(f"\n{'='*80}")
        print("✅ 内存管理测试完成")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
        import traceback
        traceback.print_exc()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HybridRagasEvaluatorV4 测试脚本')
    
    # 测试模式选择
    parser.add_argument('--mode', choices=['full', 'memory'], default='full',
                       help='测试模式: full=完整流程测试, memory=内存管理测试')
    
    # 问题数量控制
    parser.add_argument('-n', '--num-questions', type=int, default=3,
                       help='测试问题数量 (默认: 3)')
    
    # 起始索引
    parser.add_argument('-s', '--start-index', type=int, default=0,
                       help='数据集起始索引 (默认: 0)')
    
    # 数据集路径
    parser.add_argument('-d', '--dataset', type=str, 
                       default=str(project_root / "测试与质量保障层" / "testdataset" / "eval_dataset_100.jsonl"),
                       help='数据集文件路径')
    
    # 自定义问题
    parser.add_argument('-q', '--questions', nargs='+', 
                       help='自定义测试问题列表')
    
    # 断点重续
    parser.add_argument('-r', '--resume', type=str,
                       help='断点重续：指定要续传的结果文件路径')
    
    # 是否跳过内存管理测试
    parser.add_argument('--skip-memory', action='store_true',
                       help='跳过内存管理测试')
    
    return parser.parse_args()

if __name__ == "__main__":
    print("🚀 开始执行HybridRagasEvaluatorV4测试脚本...")
    
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        print(f"📋 测试配置:")
        print(f"   模式: {args.mode}")
        print(f"   问题数量: {args.num_questions}")
        print(f"   起始索引: {args.start_index}")
        print(f"   数据集: {args.dataset}")
        if args.questions:
            print(f"   自定义问题: {len(args.questions)} 个")
        if args.resume:
            print(f"   断点重续: {args.resume}")
        print()
        
        if args.mode == 'full':
            # 运行完整流程测试
            asyncio.run(simple_test(
                num_questions=args.num_questions,
                start_index=args.start_index,
                dataset_path=args.dataset,
                custom_questions=args.questions,
                resume_file=args.resume
            ))
        
        if not args.skip_memory:
            # 运行内存管理测试
            asyncio.run(test_memory_management())
        
    except Exception as e:
        print(f"❌ 主函数执行失败: {e}")
        import traceback
        traceback.print_exc()
