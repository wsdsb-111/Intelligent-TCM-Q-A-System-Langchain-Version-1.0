"""
OpenAI兼容API数据模型
定义OpenAI格式的请求和响应Pydantic模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum


class ChatRole(str, Enum):
    """聊天角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """聊天消息"""
    role: ChatRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """OpenAI聊天完成请求"""
    model: str = Field("qwen3-1.7b-finetuned", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="消息列表", min_items=1)
    temperature: Optional[float] = Field(0.7, description="生成温度", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, description="nucleus sampling参数", ge=0.0, le=1.0)
    n: Optional[int] = Field(1, description="生成数量", ge=1, le=10)
    stream: Optional[bool] = Field(False, description="是否流式输出")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止序列")
    max_tokens: Optional[int] = Field(512, description="最大生成token数", ge=1, le=4096)
    presence_penalty: Optional[float] = Field(0.0, description="存在惩罚", ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, description="频率惩罚", ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="用户标识")


class ChatCompletionChoice(BaseModel):
    """聊天完成选择"""
    index: int = Field(..., description="选择索引")
    message: Optional[ChatMessage] = Field(None, description="消息（非流式）")
    delta: Optional[Dict[str, Any]] = Field(None, description="增量更新（流式）")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class Usage(BaseModel):
    """Token使用统计"""
    prompt_tokens: int = Field(0, description="提示词token数")
    completion_tokens: int = Field(0, description="完成token数")
    total_tokens: int = Field(0, description="总token数")


class ChatCompletionResponse(BaseModel):
    """OpenAI聊天完成响应"""
    id: str = Field(..., description="响应ID")
    object: str = Field("chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[ChatCompletionChoice] = Field(..., description="选择列表")
    usage: Optional[Usage] = Field(None, description="Token使用统计")

