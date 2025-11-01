#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产RAG流程完整测试
测试大项目的完整RAG流程：问题输入->智能路由分类->向量/混合检索召回文档->重排序和查询扩展->选出生成文档->传递给模型根据文档生成回答->输出回答

注意：此测试需要先启动FastAPI服务
启动命令：python 部署与基础设施层/启动服务.py
"""

import asyncio
import sys
import time
import requests
from pathlib import Path
from typing import Dict, Any, List
import json


def _extract_stage_times(meta: Dict[str, Any]) -> Dict[str, float]:
    """从metadata中提取三个阶段用时（检索/增强/生成），单位秒。
    兼容多种字段命名，缺省为0.0。
    """
    meta = meta or {}
    # 常见字段名
    retrieval_keys = ["retrieval_time", "retrieval_time_sec", "retrieval_seconds"]
    enhance_keys = ["enhancement_time", "enhancement_time_sec", "enhance_time", "enhance_seconds"]
    gen_keys = ["generation_time", "generation_time_sec", "generate_time", "generation_seconds"]

    def pick(keys):
        for k in keys:
            v = meta.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        # 兼容 stages 字段
        stages = meta.get("stages") or meta.get("timings") or {}
        for k in keys:
            v = stages.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    return {
        "retrieval_time_sec": pick(retrieval_keys),
        "enhancement_time_sec": pick(enhance_keys),
        "generation_time_sec": pick(gen_keys),
    }


class ProductionRAGFlowTester:
    """生产RAG流程测试器"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results = []
    
    def check_service(self) -> bool:
        """检查服务是否已启动"""
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ FastAPI服务已启动")
                return True
            else:
                print(f"⚠️  服务响应异常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到FastAPI服务: {e}")
            print(f"   请确保服务已启动: python 部署与基础设施层/启动服务.py")
            return False
    
    def test_vector_only_flow(self):
        """测试纯向量检索流程"""
        print("=" * 80)
        print("测试1: 纯向量检索流程（ENTITY_DRIVEN）")
        print("=" * 80)
        
        # 测试问题：应该被路由为纯向量检索（包含明确实体）
        question = "请推荐适合经常口臭的中药"
        
        print(f"问题: {question}")
        
        try:
            # 调用API并计时（客户端整体时长）
            t0 = time.time()
            response = requests.post(
                f"{self.api_base_url}/api/v1/query",
                json={
                    "query": question,
                    "temperature": 0.5,
                    "max_new_tokens": 512
                },
                timeout=180
            )
            t1 = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 验证结果
                assert result.get("success"), "查询应该成功"
                assert result.get("answer"), "应该有答案"
                
                # 验证路由决策
                routing_decision = result.get("metadata", {}).get("routing_decision")
                print(f"\n✅ 路由决策: {routing_decision}")
                
                if routing_decision == "vector_only":
                    print("✅ 正确：使用了纯向量检索")
                else:
                    print(f"⚠️  预期 vector_only，实际 {routing_decision}")
                
                # 显示检索结果数量
                meta = result.get("metadata", {}) or {}
                num_results = meta.get("num_retrieval_results", 0)
                print(f"📚 检索结果数量: {num_results}")
                
                # 验证检索数量
                num_results = len(result.get("retrieval_results", []))
                assert num_results == 3, f"预期3个检索文档，实际 {num_results}"
                # 验证生成文档选择 (假设retrieval_results即用于生成的)
                assert num_results == 3, "生成应使用3个文档"
                # 检查扩展/重排序 (假设metadata有字段)
                meta = result.get("metadata", {})
                print(f"Debug: metadata = {meta}")  # 添加调试打印
                if not meta.get("query_expanded", False):
                    print("⚠️ 查询扩展未启用（预期启用）")
                else:
                    print("✅ 查询扩展已启用")
                if not meta.get("results_reranked", False):
                    print("⚠️ 重排序未启用（预期启用）")
                else:
                    print("✅ 重排序已启用")
                
                # 显示答案
                answer = result.get("answer", "")
                print(f"\n📝 答案（前200字符）:")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
                # 显示时间
                total_time = meta.get("total_time", 0)
                client_time = t1 - t0
                stages = _extract_stage_times(meta)
                print(f"\n⏱️  接口reported总耗时: {total_time:.2f}秒 | 客户端测量: {client_time:.2f}秒")
                print(f"   阶段用时: 检索={stages['retrieval_time_sec']:.2f}s | 增强={stages['enhancement_time_sec']:.2f}s | 生成={stages['generation_time_sec']:.2f}s")

                # 采样部分检索文档
                retrieval_samples = []
                try:
                    raw_retrieval = meta.get("retrieval_results") or []
                    for item in raw_retrieval[:3]:
                        if isinstance(item, dict):
                            retrieval_samples.append(item.get("content") or item.get("text") or str(item)[:200])
                        else:
                            retrieval_samples.append(str(item)[:200])
                except Exception:
                    pass
                
                self.results.append({
                    "test": "纯向量检索流程",
                    "success": True,
                    "routing_decision": routing_decision,
                    "num_retrieval_results": num_results,
                    "api_total_time_sec": total_time,
                    "client_response_time_sec": client_time,
                    "retrieval_time_sec": stages.get("retrieval_time_sec", 0.0),
                    "enhancement_time_sec": stages.get("enhancement_time_sec", 0.0),
                    "generation_time_sec": stages.get("generation_time_sec", 0.0),
                    "answer_preview": (answer[:200] + ("..." if len(answer) > 200 else "")),
                    "retrieval_samples": retrieval_samples
                })
                
                return True
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_hybrid_flow(self):
        """测试混合检索流程"""
        print("\n" + "=" * 80)
        print("测试2: 混合检索流程（COMPLEX_REASONING）")
        print("=" * 80)
        
        # 测试问题：应该被路由为混合检索（复杂推理）
        question = "人参和黄芪的配伍关系是什么？"
        
        print(f"问题: {question}")
        
        try:
            # 调用API并计时（客户端整体时长）
            t0 = time.time()
            response = requests.post(
                f"{self.api_base_url}/api/v1/query",
                json={
                    "query": question,
                    "temperature": 0.5,
                    "max_new_tokens": 512
                },
                timeout=180
            )
            t1 = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 验证结果
                assert result.get("success"), "查询应该成功"
                assert result.get("answer"), "应该有答案"
                
                # 验证路由决策
                routing_decision = result.get("metadata", {}).get("routing_decision")
                print(f"\n✅ 路由决策: {routing_decision}")
                
                if routing_decision == "hybrid":
                    print("✅ 正确：使用了混合检索")
                else:
                    print(f"⚠️  预期 hybrid，实际 {routing_decision}")
                
                meta = result.get("metadata", {}) or {}
                num_results = meta.get("num_retrieval_results", 0)
                print(f"📚 检索结果数量: {num_results}")
                
                # 显示答案
                answer = result.get("answer", "")
                print(f"\n📝 答案（前200字符）:")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
                # 显示时间
                total_time = meta.get("total_time", 0)
                client_time = t1 - t0
                stages = _extract_stage_times(meta)
                print(f"\n⏱️  接口reported总耗时: {total_time:.2f}秒 | 客户端测量: {client_time:.2f}秒")
                print(f"   阶段用时: 检索={stages['retrieval_time_sec']:.2f}s | 增强={stages['enhancement_time_sec']:.2f}s | 生成={stages['generation_time_sec']:.2f}s")

                # 采样部分检索文档
                retrieval_samples = []
                try:
                    raw_retrieval = meta.get("retrieval_results") or []
                    for item in raw_retrieval[:3]:
                        if isinstance(item, dict):
                            retrieval_samples.append(item.get("content") or item.get("text") or str(item)[:200])
                        else:
                            retrieval_samples.append(str(item)[:200])
                except Exception:
                    pass
                
                self.results.append({
                    "test": "混合检索流程",
                    "success": True,
                    "routing_decision": routing_decision,
                    "num_retrieval_results": num_results,
                    "api_total_time_sec": total_time,
                    "client_response_time_sec": client_time,
                    "retrieval_time_sec": stages.get("retrieval_time_sec", 0.0),
                    "enhancement_time_sec": stages.get("enhancement_time_sec", 0.0),
                    "generation_time_sec": stages.get("generation_time_sec", 0.0),
                    "answer_preview": (answer[:200] + ("..." if len(answer) > 200 else "")),
                    "retrieval_samples": retrieval_samples
                })
                
                return True
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_complex_reasoning(self):
        """测试复杂推理查询"""
        print("\n" + "=" * 80)
        print("测试3: 复杂推理查询（应该路由到混合检索）")
        print("=" * 80)
        
        # 测试问题：没有明确实体，应该是混合检索
        question = "如何治疗失眠多梦？"
        
        print(f"问题: {question}")
        
        try:
            # 调用API并计时（客户端整体时长）
            t0 = time.time()
            response = requests.post(
                f"{self.api_base_url}/api/v1/query",
                json={
                    "query": question,
                    "temperature": 0.5,
                    "max_new_tokens": 512
                },
                timeout=180
            )
            t1 = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 验证结果
                assert result.get("success"), "查询应该成功"
                assert result.get("answer"), "应该有答案"
                
                # 验证路由决策
                routing_decision = result.get("metadata", {}).get("routing_decision")
                print(f"\n✅ 路由决策: {routing_decision}")
                
                if routing_decision == "hybrid":
                    print("✅ 正确：使用了混合检索")
                else:
                    print(f"⚠️  预期 hybrid，实际 {routing_decision}")
                
                # 显示答案
                answer = result.get("answer", "")
                print(f"\n📝 答案（前200字符）:")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
                meta = result.get("metadata", {}) or {}
                total_time = meta.get("total_time", 0)
                client_time = t1 - t0
                stages = _extract_stage_times(meta)
                print(f"\n⏱️  接口reported总耗时: {total_time:.2f}秒 | 客户端测量: {client_time:.2f}秒")
                print(f"   阶段用时: 检索={stages['retrieval_time_sec']:.2f}s | 增强={stages['enhancement_time_sec']:.2f}s | 生成={stages['generation_time_sec']:.2f}s")

                # 采样部分检索文档
                retrieval_samples = []
                try:
                    raw_retrieval = meta.get("retrieval_results") or []
                    for item in raw_retrieval[:3]:
                        if isinstance(item, dict):
                            retrieval_samples.append(item.get("content") or item.get("text") or str(item)[:200])
                        else:
                            retrieval_samples.append(str(item)[:200])
                except Exception:
                    pass
                
                self.results.append({
                    "test": "复杂推理查询",
                    "success": True,
                    "routing_decision": routing_decision,
                    "api_total_time_sec": total_time,
                    "client_response_time_sec": client_time,
                    "retrieval_time_sec": stages.get("retrieval_time_sec", 0.0),
                    "enhancement_time_sec": stages.get("enhancement_time_sec", 0.0),
                    "generation_time_sec": stages.get("generation_time_sec", 0.0),
                    "answer_preview": (answer[:200] + ("..." if len(answer) > 200 else "")),
                    "retrieval_samples": retrieval_samples
                })
                
                return True
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_hybrid_document_selection(self):
        print("\n" + "=" * 80)
        print("测试4: 混合检索文档选择")
        print("=" * 80)
        
        question = "人参和黄芪的配伍关系是什么？"
        
        print(f"问题: {question}")
        
        try:
            # 调用API并计时（客户端整体时长）
            t0 = time.time()
            response = requests.post(
                f"{self.api_base_url}/api/v1/query",
                json={
                    "query": question,
                    "temperature": 0.5,
                    "max_new_tokens": 512
                },
                timeout=180
            )
            t1 = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # 验证结果
                assert result.get("success"), "查询应该成功"
                assert result.get("answer"), "应该有答案"
                
                # 验证路由决策
                routing_decision = result.get("metadata", {}).get("routing_decision")
                print(f"\n✅ 路由决策: {routing_decision}")
                
                if routing_decision == "hybrid":
                    print("✅ 正确：使用了混合检索")
                else:
                    print(f"⚠️  预期 hybrid，实际 {routing_decision}")
                
                meta = result.get("metadata", {}) or {}
                num_results = meta.get("num_retrieval_results", 0)
                print(f"📚 检索结果数量: {num_results}")
                
                # 验证检索
                retrieval_results = result.get("retrieval_results", [])
                
                # 调试：打印前几个结果的详细信息
                print(f"\n🔍 调试信息：")
                print(f"检索结果总数: {len(retrieval_results)}")
                if retrieval_results:
                    print(f"第一个结果类型: {type(retrieval_results[0])}")
                    print(f"第一个结果内容: {retrieval_results[0]}")
                    print(f"前3个结果的source字段: {[r.get('source') if isinstance(r, dict) else 'N/A' for r in retrieval_results[:3]]}")
                    print(f"所有结果的source字段: {[r.get('source') if isinstance(r, dict) else 'N/A' for r in retrieval_results]}")
                
                vector_docs = [r for r in retrieval_results if isinstance(r, dict) and r.get("source") == "vector"]
                graph_docs = [r for r in retrieval_results if isinstance(r, dict) and r.get("source") == "graph"]
                
                if len(retrieval_results) != 10:
                    print(f"⚠️ 预期总检索10（5向量+5图谱），实际 {len(retrieval_results)}")
                else:
                    print(f"✅ 总检索数量正确: 10")
                if len(vector_docs) != 5:
                    print(f"⚠️ 预期5个向量文档，实际 {len(vector_docs)}")
                    print(f"   所有source值: {set(r.get('source') if isinstance(r, dict) else None for r in retrieval_results)}")
                else:
                    print(f"✅ 向量文档数量正确: 5")
                if len(graph_docs) != 5:
                    print(f"⚠️ 预期5个图谱文档，实际 {len(graph_docs)}")
                    print(f"   所有source值: {set(r.get('source') if isinstance(r, dict) else None for r in retrieval_results)}")
                assert len(graph_docs) == 5, f"预期5个图谱文档，实际 {len(graph_docs)}"
                # 验证生成选择 (假设metadata有selected_docs或类似)
                selected = meta.get("selected_for_generation", retrieval_results)
                selected_vector = [r for r in selected if r.get("source") == "vector"]
                selected_graph = [r for r in selected if r.get("source") == "graph"]
                assert len(selected_vector) == 3, "生成应使用3个向量文档"
                assert len(selected_graph) == 5, "生成应使用5个图谱文档"
                
                # 显示答案
                answer = result.get("answer", "")
                print(f"\n📝 答案（前200字符）:")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
                # 显示时间
                total_time = meta.get("total_time", 0)
                client_time = t1 - t0
                stages = _extract_stage_times(meta)
                print(f"\n⏱️  接口reported总耗时: {total_time:.2f}秒 | 客户端测量: {client_time:.2f}秒")
                print(f"   阶段用时: 检索={stages['retrieval_time_sec']:.2f}s | 增强={stages['enhancement_time_sec']:.2f}s | 生成={stages['generation_time_sec']:.2f}s")

                # 采样部分检索文档
                retrieval_samples = []
                try:
                    raw_retrieval = meta.get("retrieval_results") or []
                    for item in raw_retrieval[:3]:
                        if isinstance(item, dict):
                            retrieval_samples.append(item.get("content") or item.get("text") or str(item)[:200])
                        else:
                            retrieval_samples.append(str(item)[:200])
                except Exception:
                    pass
                
                self.results.append({
                    "test": "混合检索文档选择",
                    "success": True,
                    "routing_decision": routing_decision,
                    "num_retrieval_results": num_results,
                    "api_total_time_sec": total_time,
                    "client_response_time_sec": client_time,
                    "retrieval_time_sec": stages.get("retrieval_time_sec", 0.0),
                    "enhancement_time_sec": stages.get("enhancement_time_sec", 0.0),
                    "generation_time_sec": stages.get("generation_time_sec", 0.0),
                    "answer_preview": (answer[:200] + ("..." if len(answer) > 200 else "")),
                    "retrieval_samples": retrieval_samples
                })
                
                return True
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get("success"))
        
        print(f"总测试数: {total_tests}")
        print(f"成功数: {successful_tests}")
        print(f"失败数: {total_tests - successful_tests}")
        
        if self.results:
            avg_api_time = sum(r.get("api_total_time_sec", 0) for r in self.results) / len(self.results)
            avg_client_time = sum(r.get("client_response_time_sec", 0) for r in self.results) / len(self.results)
            print(f"\n平均耗时: 接口reported {avg_api_time:.2f}秒 | 客户端测量 {avg_client_time:.2f}秒")
        
        print("\n详细结果:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.get("success") else "❌"
            routing = result.get("routing_decision", "unknown")
            api_time = result.get("api_total_time_sec", 0)
            client_time = result.get("client_response_time_sec", 0)
            print(f"{status} 测试{i}: {result.get('test')} | 路由: {routing} | 接口: {api_time:.2f}s | 客户端: {client_time:.2f}s")
        
        print("=" * 80)

    def save_json_report(self, output_path: Path = None):
        """保存JSON报告（包含检索样本与时长）"""
        try:
            if output_path is None:
                output_path = Path(__file__).parent / "production_rag_report.json"
            report = {
                "api_base_url": self.api_base_url,
                "summary": {
                    "total_tests": len(self.results),
                    "successful": sum(1 for r in self.results if r.get("success")),
                },
                "results": self.results
            }
            # 确保目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 JSON报告已生成: {output_path}")
            # 冗余备份到 reports 目录，便于查找
            try:
                reports_dir = Path(__file__).parent / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                backup_path = reports_dir / "production_rag_report.json"
                with open(backup_path, "w", encoding="utf-8") as bf:
                    json.dump(report, bf, ensure_ascii=False, indent=2)
                print(f"📄 备份报告: {backup_path}")
            except Exception as _:
                pass
        except Exception as e:
            print(f"生成JSON报告失败: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 生产RAG流程完整测试")
    print("=" * 80)
    print()
    print("测试目标：")
    print("1. 验证智能路由分类功能（二元路由：vector_only / hybrid）")
    print("2. 验证纯向量检索流程（ENTITY_DRIVEN）")
    print("3. 验证混合检索流程（COMPLEX_REASONING）")
    print("4. 验证检索文档数量和生成文档数量")
    print("5. 验证完整的端到端流程")
    print()
    
    tester = ProductionRAGFlowTester()
    
    # 检查服务是否已启动
    if not tester.check_service():
        print("\n❌ 测试终止：FastAPI服务未启动")
        print("\n请先启动服务：")
        print("  python 部署与基础设施层/启动服务.py")
        return
    
    # 运行测试
    import os
    print(f"当前工作目录: {os.getcwd()}")
    try:
        # 测试1: 纯向量检索
        tester.test_vector_only_flow()
        
        # 测试2: 混合检索
        tester.test_hybrid_flow()
        
        # 测试3: 复杂推理
        tester.test_complex_reasoning()

        # 测试4: 混合检索文档选择
        tester.test_hybrid_document_selection()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印总结
        tester.print_summary()
        # 生成JSON报告（多重回退）
        try:
            tester.save_json_report()
        except Exception as e:
            print(f"首次保存报告失败，尝试写入当前工作目录: {e}")
            try:
                from pathlib import Path
                alt_path = Path(os.getcwd()) / "production_rag_report.json"
                tester.save_json_report(output_path=alt_path)
            except Exception as e2:
                print(f"写入当前目录仍失败: {e2}")
    
    print("\n✅ 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

