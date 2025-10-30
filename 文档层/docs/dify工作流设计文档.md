Dify 智能问答系统工作流设计文档（V2.0）
（基于混合模式：Dify 节点直接实现 + FastAPI 转发，含智能路由混合判断与参数动态配置）
一、项目概述
1.1 目标
通过 Dify 工作流实现 “用户查询→智能路由（模型 + 关键词混合判断）→检索增强→回答生成” 的完整流程，结合混合模式（轻量逻辑本地化、重度依赖组件转发）和懒加载机制（动态释放显存），解决本地设备显存限制问题。同时支持在 Dify 平台直接调整本地微调模型的提示词模板和生成参数（temperature、max_tokens 等），提升系统灵活性。
1.2 核心升级点
智能路由由 “Qwen-Flash 模型 + 关键词规则” 混合实现，提升分类准确率。
支持在 Dify 界面配置本地微调模型的提示词模板（含动态拼接）和生成参数（temperature、max_tokens 等）。
二、整体架构
用户层输入查询 → 传递至 Dify 工作流层；Dify 工作流层执行三个核心动作：1. 调用本地关键词库进行关键词规则判断；2. 调用阿里云 Qwen-Flash 模型进行模型分类；3. 智能路由节点综合规则与模型结果，输出 “向量检索” 或 “混合检索” 结果；Dify 工作流层通过 HTTP 请求调用 FastAPI 服务；FastAPI 服务管理本地资源层（FAISS 向量数据库、Neo4j 知识图谱、重排序模型、本地微调模型）；最终 Dify 工作流层将回答结果输出至用户层。
三、Dify 工作流设计（核心步骤与节点拆分）
3.1 流程总览
用户查询 → 智能路由（关键词规则 + Qwen-Flash 模型） → 加载检索组件 → 检索与知识召回 → 查询扩展与重排序 → 关键词增强 → 卸载检索组件 → 加载本地模型生成组件 → 回答生成（提示词 / 参数由 Dify 配置） → 卸载生成组件 → 输出回答
3.2 节点拆分与实现方式
接受用户查询：核心功能是获取用户输入并传递至下一节点；实现方式为 Dify 直接实现；依赖资源是 Dify 内置变量（inputs.query）。
智能路由（关键词规则）：核心功能是基于关键词库初步分类查询；实现方式为 Dify 直接实现；依赖资源是本地关键词库（硬编码 / 轻量文件）。
智能路由（模型判断）：核心功能是用 Qwen-Flash 模型二次分类；实现方式为 Dify 直接实现（调用云 API）；依赖资源是阿里云 Qwen-Flash API。
路由结果融合：核心功能是结合规则与模型结果输出最终路由；实现方式为 Dify 直接实现；依赖资源是规则结果 + 模型结果。
加载检索组件：核心功能是加载 FAISS、Neo4j、重排序 / 扩展模型；实现方式为 Dify 调用 FastAPI 接口；依赖资源是 FastAPI 的组件管理模块。
检索与知识召回：核心功能是基于路由结果召回文档；实现方式为 Dify 调用 FastAPI 接口；依赖资源是 FAISS、Neo4j。
查询扩展与重排序：核心功能是优化召回文档质量；实现方式为 Dify 调用 FastAPI 接口；依赖资源是重排序模型、查询扩展模型。
关键词增强：核心功能是从向量检索文档中提取关键词；实现方式为 Dify 直接实现；依赖资源是轻量分词库（如 jieba）。
卸载检索组件：核心功能是释放 FAISS、重排序等组件显存；实现方式为 Dify 调用 FastAPI 接口；依赖资源是 FastAPI 的组件管理模块。
配置生成参数与提示词：核心功能是定义本地模型的生成参数和提示词模板；实现方式为 Dify 直接实现；依赖资源是 Dify 环境变量 / 节点变量。
加载生成组件：核心功能是加载本地微调模型（Qwen3-1.7B+LoRA）；实现方式为 Dify 调用 FastAPI 接口；依赖资源是 FastAPI 的组件管理模块。
回答生成：核心功能是基于最终文档生成回答；实现方式为 Dify 调用 FastAPI 接口；依赖资源是本地微调模型（应用 Dify 配置的参数）。
卸载生成组件：核心功能是释放生成模型显存；实现方式为 Dify 调用 FastAPI 接口；依赖资源是 FastAPI 的组件管理模块。
输出回答：核心功能是返回结果给用户；实现方式为 Dify 直接实现；依赖资源是 Dify 内置输出节点。
四、智能路由（模型 + 关键词混合判断）设计
4.1 核心逻辑
智能路由采用 “规则优先，模型兜底” 的混合策略：
先用关键词规则快速匹配（如 “图像”“图谱” 等关键词直接判定为混合检索）；
规则无法匹配时，调用 Qwen-Flash 模型进行语义分类；
最终融合两者结果，输出 “vector_search”（向量检索）或 “hybrid_search”（混合检索）。
4.2 阿里云 Qwen-Flash 模型配置
API Key：sk-6157e39178ac439bb00c43ba6b094501
模型名称：qwen-flash
Base URL：https://dashscope.aliyuncs.com/compatible-mode/v1
Dify 配置：通过 “Custom OpenAI-compatible” 集成（进入 Dify 后台→Settings→Model Providers→Add Model Provider，选择 “Custom OpenAI-compatible”，填写 API Base URL、API Key，添加模型名称 “qwen-flash”）。
4.3 节点实现示例
关键词规则判断节点（Dify 代码节点）：定义关键词库（可在 Dify 环境变量中配置，方便修改）：KEYWORD_RULES = {"hybrid_search": ["图像", "图谱", "关系", "原因", "为什么"], "vector_search": ["定义", "简介", "是什么", "概述"]}；实现规则判断函数：def rule_based_route (query: str) -> str：将查询转为小写，先检查混合检索关键词，存在则返回 “hybrid_search”；再检查向量检索关键词，存在则返回 “vector_search”；规则无法匹配时返回空；输出规则结果：outputs.rule_route = rule_based_route (inputs.query)。
模型判断节点（调用 Qwen-Flash，Dify 代码节点）：仅当规则无法匹配（inputs.rule_route == ""）时调用模型；定义模型 Prompt：“请将用户查询分类为以下两种类型之一：- vector_search（适合纯文本定义、简介类查询）- hybrid_search（适合涉及图像、关系、原因的查询）。用户查询：{inputs.query} 仅返回分类结果，无需额外说明。”；调用 Qwen-Flash 模型：outputs.model_route = llm.invoke (prompt).strip ()；若规则已匹配（inputs.rule_route != ""），则 outputs.model_route =""。
路由结果融合节点（Dify 代码节点）：若规则有结果（inputs.rule_route != ""），优先使用规则结果：outputs.final_route = inputs.rule_route；若规则无结果，使用模型结果（默认向量检索兜底）：outputs.final_route = inputs.model_route if inputs.model_route in ["vector_search", "hybrid_search"] else "vector_search"。
五、Dify 动态配置提示词与生成参数（本地微调模型）
5.1 提示词模板配置（Dify 端）
全局提示词模板（Dify 环境变量）：
进入 Dify 应用→Settings→Environment Variables，添加变量：
变量名：GENERATION_PROMPT_TEMPLATE
变量值：“基于以下文档，用简洁准确的语言回答问题（参考文档中的关键信息，不要编造）。文档：{docs}\n 问题：{query}\n 回答：”
提示词拼接节点（Dify 代码节点）：
从环境变量获取模板：prompt_template = env.GENERATION_PROMPT_TEMPLATE；
替换占位符（docs 为最终文档，query 为用户输入）：full_prompt = prompt_template.format (docs="\n".join ([doc ["content"] for doc in inputs.reranked_docs]), query=inputs.query)；
输出完整提示词：outputs.full_prompt = full_prompt（传递给生成节点）。
5.2 生成参数配置（Dify 端）
参数定义（Dify 环境变量）：
进入 Dify 应用→Settings→Environment Variables，添加变量：
TEMPERATURE：0.6（控制随机性）
MAX_TOKENS：800（最大生成长度）
TOP_P：0.9（核采样参数）
参数处理节点（Dify 代码节点）：
准备基础生成参数：generation_params = {"temperature": float (env.TEMPERATURE), "max_tokens": int (env.MAX_TOKENS), "top_p": float (env.TOP_P)}；
动态调整（可选）：若文档数量 > 5，增加 max_tokens：if len (inputs.reranked_docs) > 5: generation_params ["max_tokens"] = 1200；
输出参数：outputs.generation_params = generation_params（传递给生成节点）。
六、FastAPI 服务设计（适配动态参数）
6.1 生成接口升级（接收 Dify 传递的提示词与参数）
导入必要模块：from pydantic import BaseModel；from typing import Dict；定义请求模型：class GenerateRequest (BaseModel)：包含 full_prompt（Dify 拼接的完整提示词）、generation_params（Dify 传递的生成参数）；实现生成接口：@app.post ("/generate")，函数内步骤：
解析提示词和参数：prompt = request.full_prompt；params = request.generation_params；
调用本地微调模型（应用参数）：inputs = tokenizer (prompt, return_tensors="pt").to (model.device)；outputs = model.generate (**inputs, temperature=params.get ("temperature", 0.7), max_new_tokens=params.get ("max_tokens", 512), top_p=params.get ("top_p", 0.9), do_sample=True)；
返回生成结果：generated_text = tokenizer.decode (outputs [0], skip_special_tokens=True)；return {"text": generated_text}。
6.2 Dify 调用生成接口（传递提示词与参数）
Dify 代码节点实现：导入 requests 模块；尝试调用 FastAPI 接口：response = requests.post ("http://localhost:8000/generate", json={"full_prompt": inputs.full_prompt,"generation_params": inputs.generation_params}, timeout=30)；输出回答：outputs.answer = response.json ()["text"]；异常处理：except Exception as e: outputs.error = f"生成失败: {str (e)}"。
七、部署与测试注意事项
智能路由优先级验证：测试含关键词的查询是否被规则正确捕获（如含 “图像” 的查询是否直接判定为混合检索），无关键词的查询是否由模型正确分类。
参数生效测试：在 Dify 中修改 TEMPERATURE（如设为 0.1），验证回答是否更稳定；修改 MAX_TOKENS，检查生成长度是否变化。
懒加载与参数兼容性：确保生成参数（如 max_tokens）不超过本地模型的最大上下文长度（避免显存溢出），可在 FastAPI 中添加参数校验逻辑。
网络可达性：Dify 容器需能访问 FastAPI 服务（建议同网段，用宿主机 IP 如 “http://192.168.x.x:8000”）；确保 Dify 能访问阿里云 API（https://dashscope.aliyuncs.com）。
八、总结
本版本文档重点优化了智能路由的混合判断逻辑（关键词规则 + Qwen-Flash 模型），提升了分类准确性；同时实现了本地微调模型的提示词和生成参数在 Dify 上的动态配置，无需修改代码即可灵活调整生成效果。
通过 “Dify 配置层 + FastAPI 执行层” 的分离设计，既保留了本地资源的高效利用，又兼顾了系统的可操作性和扩展性，完美适配本地设备显存限制的场景。