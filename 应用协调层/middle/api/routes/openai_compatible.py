"""
OpenAI兼容API端点
为Dify提供OpenAI格式的API接口，支持API Key认证
"""

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
import json
import secrets
import os
import time
import traceback

from ..schemas.openai_schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, ChatCompletionChoice, Usage
)
from ...services.model_service import get_model_service
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/v1", tags=["OpenAI兼容"])

# API Key配置（可以从配置文件或环境变量读取）
API_KEY = os.getenv("OPENAI_API_KEY", "sk-qwen3-1.7b-local-dev-key-12345")
API_KEY_HEADER = "x-api-key"  # 或使用 Authorization header


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None)
) -> str:
    """
    验证API Key
    
    支持两种方式：
    1. x-api-key header
    2. Authorization: Bearer <key>
    """
    api_key = None
    
    # 优先使用 x-api-key header
    if x_api_key:
        api_key = x_api_key
    # 其次使用 Authorization header
    elif authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Please provide a valid API key using 'x-api-key' header or 'Authorization: Bearer <key>' header."
        )
    
    return api_key


@router.post("/chat/completions", 
            response_model=ChatCompletionResponse,
            summary="OpenAI兼容聊天完成接口",
            description="OpenAI格式的聊天完成接口，用于Dify集成")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """
    OpenAI兼容的聊天完成接口
    
    接收OpenAI格式的请求，转发到本地模型服务
    提示词和生成参数由Dify工作流节点配置，这里只负责转发
    """
    try:
        logger.info(f"收到OpenAI兼容请求: model={request.model}, messages={len(request.messages)}条")
        
        # 获取模型服务
        model_service = get_model_service()
        if not model_service._initialized:
            raise HTTPException(
                status_code=503,
                detail="Model service not initialized"
            )
        
        # 将OpenAI格式的消息转换为本地模型需要的格式
        # 提取最后一条用户消息作为query（提示词由Dify配置）
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400,
                detail="No user message found in messages"
            )
        
        # 使用最后一条用户消息
        query = user_messages[-1].content
        
        # 提取系统提示词（如果有）
        system_messages = [msg for msg in request.messages if msg.role == "system"]
        system_prompt = system_messages[-1].content if system_messages else None
        
        # 构建完整的提示词（包含所有消息，提示词格式由Dify工作流配置）
        full_prompt = ""
        for msg in request.messages:
            # 安全获取role值（支持字符串或枚举）
            role_value = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            if role_value == "system":
                full_prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif role_value == "user":
                full_prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif role_value == "assistant":
                full_prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        
        # 如果不是流式输出或最后一条不是assistant消息，添加assistant开始标记
        last_role = request.messages[-1].role.value if hasattr(request.messages[-1].role, 'value') else str(request.messages[-1].role)
        if not request.stream or last_role != "assistant":
            full_prompt += "<|im_start|>assistant\n"
        
        # 使用完整提示词作为query（提示词内容由Dify配置）
        query = full_prompt if full_prompt else query
        
        # 获取生成参数（从请求中读取，如果Dify已配置）
        max_tokens = request.max_tokens if request.max_tokens else 512
        temperature = request.temperature if request.temperature is not None else 0.7
        top_p = request.top_p if request.top_p is not None else 0.95
        
        # 流式输出
        if request.stream:
            return StreamingResponse(
                _stream_generate(model_service, query, max_tokens, temperature, top_p),
                media_type="text/event-stream"
            )
        
        # 非流式输出
        logger.debug(f"调用模型生成: query_length={len(query)}, max_tokens={max_tokens}, temperature={temperature}")
        try:
            result = model_service.generate(
                query=query,
                system_prompt=None,  # 已在query中包含
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as gen_error:
            logger.error(f"模型生成失败: {gen_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model generation failed: {str(gen_error)}"
            )
        
        answer = result.get("answer", "")
        if not answer:
            logger.warning("模型返回空答案")
            answer = "抱歉，模型未能生成有效回答。"
        
        # 获取metadata（如果存在）
        metadata = result.get("metadata", {})
        input_tokens = metadata.get("input_tokens", 0)
        generated_tokens = metadata.get("generated_tokens", len(answer.split()))  # 简单估算
        
        # 构建OpenAI格式的响应
        try:
            response = ChatCompletionResponse(
                id=f"chatcmpl-{secrets.token_hex(12)}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=answer
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=input_tokens,
                    completion_tokens=generated_tokens,
                    total_tokens=input_tokens + generated_tokens
                )
            )
        except Exception as resp_error:
            logger.error(f"构建响应失败: {resp_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build response: {str(resp_error)}"
            )
        
        logger.info(f"OpenAI兼容请求完成: answer_length={len(answer)}")
        return response
    
    except HTTPException as http_err:
        # 重新抛出HTTP异常（保持原有状态码）
        raise http_err
    except Exception as e:
        # 记录详细错误信息
        error_detail = traceback.format_exc()
        logger.error(f"OpenAI兼容请求错误: {e}\n{error_detail}")
        
        # 返回更友好的错误信息
        error_message = str(e)
        if "not initialized" in error_message.lower():
            raise HTTPException(
                status_code=503,
                detail="Model service not initialized. Please check if the model is loaded."
            )
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {error_message}"
            )


async def _stream_generate(model_service, query: str, max_tokens: int, temperature: float, top_p: float):
    """
    流式生成响应
    """
    import time
    import asyncio
    
    # 注意：当前model_service.generate是同步的，这里简化为一次性返回
    # 如果需要真正的流式输出，需要修改model_service支持流式生成
    try:
        result = model_service.generate(
            query=query,
            system_prompt=None,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        answer = result.get("answer", "")
        
        # 模拟流式输出（按字符分块）
        chunk_size = 10
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i+chunk_size]
            delta = {
                "role": "assistant",
                "content": chunk
            }
            
            chunk_data = {
                "id": f"chatcmpl-{secrets.token_hex(12)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen3-1.7b-finetuned",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None if i + chunk_size < len(answer) else "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)  # 小延迟模拟流式效果
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"流式生成错误: {e}", exc_info=True)
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


@router.get("/models",
           summary="列出可用模型",
           description="返回可用的模型列表（OpenAI兼容）")
async def list_models(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    列出可用模型
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-1.7b-finetuned",
                "object": "model",
                "created": 1700000000,
                "owned_by": "local"
            }
        ]
    }

