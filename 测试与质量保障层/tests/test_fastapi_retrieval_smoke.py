#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 检索能力冒烟测试
目标：验证服务是否能通过 /api/v1/query 进行向量检索与混合检索并召回文档

运行方式：
  1) 先启动服务：python 部署与基础设施层/启动服务.py
  2) 再运行本测试：python 测试与质量保障层/tests/test_fastapi_retrieval_smoke.py
"""

import requests

API_BASE = "http://localhost:8000"


def call_query(question: str):
    url = f"{API_BASE}/api/v1/query"
    payload = {
        "query": question,
        "temperature": 0.5,
        "max_new_tokens": 256
    }
    # 提高超时，避免混合检索初始化/图查询较慢导致读超时
    resp = requests.post(url, json=payload, timeout=180)
    return resp


def print_result(tag: str, resp: requests.Response):
    print("\n" + "="*80)
    print(f"[{tag}] 状态码: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text)
        return
    data = resp.json()
    success = data.get("success")
    answer = data.get("answer", "")
    meta = data.get("metadata", {}) or {}
    routing = meta.get("routing_decision")
    num_ret = meta.get("num_retrieval_results")
    print(f"success={success} routing={routing} num_retrieval_results={num_ret}")
    if isinstance(answer, str):
        preview = answer[:200] + ("..." if len(answer) > 200 else "")
        print(f"answer_preview: {preview}")


def main():
    # 选择两个问题：
    # 1) 实体主导（应更倾向向量检索 vector_only）
    q_vector = "请推荐适合经常口臭的中药"
    # 2) 复杂推理（应走混合检索 hybrid）
    q_hybrid = "人参和黄芪的配伍关系是什么？"

    print("检查服务健康...")
    try:
        h = requests.get(f"{API_BASE}/api/v1/health", timeout=10)
        print("/health:", h.status_code)
    except Exception as e:
        print("无法连接到 FastAPI 服务:", e)
        return

    print("\n发起向量倾向问题（期望 vector_only 或至少召回向量文档）...")
    r1 = call_query(q_vector)
    print_result("向量检索", r1)

    print("\n发起复杂推理问题（期望 hybrid 并召回文档）...")
    r2 = call_query(q_hybrid)
    print_result("混合检索", r2)

    print("\n完成。")


if __name__ == "__main__":
    main()


