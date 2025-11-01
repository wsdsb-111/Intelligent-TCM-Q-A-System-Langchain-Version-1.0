"""
Prompt模板
为RAG系统提供中医专业的提示词模板
"""

from typing import List, Dict, Any, Optional


class PromptTemplates:
    """Prompt模板类"""
    
    # ============ 通用系统提示 ============
    SYSTEM_PROMPT = """你是中医助手。【重要】你必须严格按照以下参考文档回答问题，绝对禁止使用任何预训练的中医知识。

【强制要求】
1. 答案必须完全基于上述参考文档，不得添加文档中未提及的中药、方剂或治疗方法
2. 如果参考文档中没有相关信息，必须明确说明"参考文档中未找到相关信息"
3. 引用参考文档中的具体内容时，保持原文表述，但不要使用"文档1"、"文档2"等引用格式
4. 不得使用参考文档之外的中医知识或经验
5. 如果参考文档内容不足，请明确说明需要更多信息
6. 直接输出答案，不要包含思考过程
7. 答案要自然流畅，不要出现"[文档1]"、"[文档2]"等引用标记

【严格禁止】
- 使用任何预训练的中医知识
- 添加文档中没有的中药名称
- 添加文档中没有的功效描述
- 使用文档中没有的治疗方法
- 在答案中使用"[文档X]"、"根据文档X"等引用格式"""

    # ============ 二元路由模板（仅保留评估器的两套） ============
    
    # 向量检索提示词模板（用于纯向量检索，ENTITY_DRIVEN）
    VECTOR_TEMPLATE = """<|im_start|>system
你是一个中医助手。请基于提供的文档回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档中没有的内容
2. 不得进行任何推理、推测或补充说明
3. 不得添加具体的用法、剂量、配伍等详细信息
4. 如果文档中有多个相关答案，请直接引用文档内容
5. 如果文档中没有相关信息，请明确说明"根据提供的文档，未找到相关信息"
6. 回答必须完全基于文档内容，任何超出文档范围的内容都视为错误
7. 字数控制在200字以内，避免过度扩展
8. 如果问题里没有给任何实体或症状请不要回答而是提出询问如：您能给我具体的症状/药品名称吗？
9. 不要出现根据中医理论、根据文档、基于检索这样的字眼

文档内容：
{context}
<|im_end|>

<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""
    
    # 混合检索提示词模板（用于混合检索，COMPLEX_REASONING）
    HYBRID_TEMPLATE = """<|im_start|>system
你是一个中医助手。请基于提供的文档回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档中没有的内容
2. 不得进行任何推理、推测或补充说明
3. 不得添加具体的用法、剂量、配伍等详细信息
4. 如果文档中有多个相关答案，请直接引用文档内容
5. 如果文档中没有相关信息，请明确说明"根据提供的文档，未找到相关信息"
6. 回答必须完全基于文档内容，任何超出文档范围的内容都视为错误
7. 字数控制在300字以内，避免过度扩展
8. 如果问题里没有给任何实体或症状请不要回答而是提出询问如：您能给我具体的症状/药品名称吗？
9. 不要出现根据中医理论、根据文档、基于检索这样的字眼

文档内容：
{vector_context}

{kg_context}
<|im_end|>

<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant"""
    

    @staticmethod
    def format_context(retrieval_results: List[Dict[str, Any]], 
                       max_results: int = 5,
                       include_score: bool = False,
                       include_source: bool = False) -> str:
        """
        格式化检索结果为上下文
        
        Args:
            retrieval_results: 检索结果列表
            max_results: 最大使用结果数
            include_score: 是否包含评分
            include_source: 是否包含来源
            
        Returns:
            格式化的上下文字符串
        """
        if not retrieval_results:
            return "暂无相关参考资料。"
        
        context_parts = []
        
        for i, result in enumerate(retrieval_results[:max_results], 1):
            content = result.get('content', '')
            
            # 构建条目
            parts = [f"{i}. {content}"]
            
            # 添加评分信息
            if include_score:
                score = result.get('fused_score', result.get('score', 0))
                parts.append(f"   (相关度: {score:.2f})")
            
            # 添加来源信息
            if include_source:
                sources = result.get('contributing_sources', [])
                if sources:
                    sources_str = ', '.join(sources)
                    parts.append(f"   (来源: {sources_str})")
            
            context_parts.append('\n'.join(parts))
        
        return '\n\n'.join(context_parts)
    
    @staticmethod
    def build_rag_prompt(query: str,
                        retrieval_results: List[Dict[str, Any]],
                        template: str = None,
                        max_context_results: int = 5) -> str:
        """
        构建完整的RAG提示词
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果
            template: 使用的模板（默认RAG_TEMPLATE）
            max_context_results: 最大上下文结果数
            
        Returns:
            完整的提示词
        """
        # 格式化上下文
        context = PromptTemplates.format_context(
            retrieval_results,
            max_results=max_context_results,
            include_score=False,
            include_source=False
        )
        
        # 使用评估器向量模板作为默认RAG模板
        return PromptTemplates.VECTOR_TEMPLATE.format(context=context, query=query)
    
    @staticmethod
    def build_direct_prompt(query: str) -> str:
        """
        构建直接问答提示词（无检索上下文）
        
        Args:
            query: 用户问题
            
        Returns:
            完整的提示词
        """
        return PromptTemplates.DIRECT_TEMPLATE.format(query=query)
    
    @staticmethod
    def build_vector_prompt(query: str, retrieval_results: List[Dict[str, Any]], max_context_results: int = 5) -> str:
        """
        构建向量检索模式的提示词
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果
            max_context_results: 最大上下文结果数
            
        Returns:
            完整的提示词
        """
        context = PromptTemplates.format_context(retrieval_results, max_results=max_context_results)
        return PromptTemplates.VECTOR_TEMPLATE.format(context=context, query=query)
    
    @staticmethod
    def build_kg_prompt(query: str, retrieval_results: List[Dict[str, Any]], max_context_results: int = 5) -> str:
        # 已不使用独立KG模板，回退到VECTOR模板（保持接口兼容）
        context = PromptTemplates.format_context(retrieval_results, max_results=max_context_results)
        return PromptTemplates.VECTOR_TEMPLATE.format(context=context, query=query)
    
    @staticmethod
    def build_hybrid_prompt(query: str, 
                           vector_results: List[Dict[str, Any]], 
                           kg_results: List[Dict[str, Any]],
                           max_context_results: int = 3) -> str:
        """
        构建混合检索模式的提示词
        
        Args:
            query: 用户问题
            vector_results: 向量检索结果
            kg_results: 知识图谱检索结果
            max_context_results: 每种检索方式的最大结果数
            
        Returns:
            完整的提示词
        """
        vector_context = PromptTemplates.format_context(vector_results, max_results=max_context_results)
        kg_context = PromptTemplates.format_context(kg_results, max_results=max_context_results)
        return PromptTemplates.HYBRID_TEMPLATE.format(
            vector_context=vector_context,
            kg_context=kg_context,
            query=query
        )
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量（粗略估计）
        中文：1字符约1.5-2个token
        
        Args:
            text: 输入文本
            
        Returns:
            估计的token数
        """
        # 简单估算：中文字符数 * 1.8
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        # 中文按1.8倍，英文按1.3倍估算
        estimated_tokens = int(chinese_chars * 1.8 + other_chars * 1.3)
        
        return estimated_tokens
    
    @staticmethod
    def truncate_context_by_tokens(retrieval_results: List[Dict[str, Any]],
                                   max_tokens: int = 1500) -> List[Dict[str, Any]]:
        """
        根据token限制截断上下文
        
        Args:
            retrieval_results: 检索结果列表
            max_tokens: 最大token数
            
        Returns:
            截断后的结果列表
        """
        truncated_results = []
        total_tokens = 0
        
        for result in retrieval_results:
            content = result.get('content', '')
            tokens = PromptTemplates.estimate_tokens(content)
            
            if total_tokens + tokens <= max_tokens:
                truncated_results.append(result)
                total_tokens += tokens
            else:
                # 如果还有空间，截断当前内容
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # 至少保留100个token
                    # 简单截断：按字符比例
                    ratio = remaining_tokens / tokens
                    truncate_length = int(len(content) * ratio)
                    truncated_content = content[:truncate_length] + "..."
                    
                    truncated_result = result.copy()
                    truncated_result['content'] = truncated_content
                    truncated_results.append(truncated_result)
                
                break
        
        return truncated_results
    
    @staticmethod
    def format_entities_and_relationships(result: Dict[str, Any]) -> str:
        """
        格式化实体和关系信息
        
        Args:
            result: 检索结果
            
        Returns:
            格式化的实体关系字符串
        """
        parts = []
        
        entities = result.get('entities', [])
        if entities:
            parts.append(f"实体: {', '.join(entities)}")
        
        relationships = result.get('relationships', [])
        if relationships:
            parts.append(f"关系: {', '.join(relationships)}")
        
        return ' | '.join(parts) if parts else ""


# 其余历史模板已移除，保留评估器的两套模板（VECTOR/HYBRID）

