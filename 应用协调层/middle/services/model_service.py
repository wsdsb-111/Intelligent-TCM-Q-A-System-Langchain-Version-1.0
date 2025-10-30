"""
模型服务
负责加载和管理Qwen1.5-1.8B + LoRA微调模型
"""

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import time

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelService:
    """
    模型服务类
    单例模式，负责模型的加载、推理和生命周期管理
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式，确保只加载一次模型"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化模型服务"""
        if self._initialized:
            return
            
        self.base_model = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "average_generation_time": 0.0
        }
        
        logger.info("模型服务初始化（延迟加载）")
    
    def load_model(self, 
                   base_model_path: str,
                   adapter_path: str,
                   device: str = "auto",
                   torch_dtype = torch.float16) -> bool:
        """
        加载基础模型和LoRA适配器
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA适配器路径
            device: 设备类型 (auto/cuda/cpu)
            torch_dtype: 模型精度
            
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info("=" * 60)
            logger.info("开始加载中医问答微调模型")
            logger.info("=" * 60)
            
            # 检测设备
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            if self.device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                logger.info("使用CPU模式")
            
            logger.info("-" * 60)
            
            # 验证路径
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"适配器路径不存在: {adapter_path}")
            
            # 加载基础模型
            logger.info("[1/3] 正在加载基础模型...")
            # 创建临时卸载目录用于显存不足时卸载部分层
            import tempfile
            temp_offload_dir = tempfile.mkdtemp()
            
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            # 如果使用auto device_map，添加offload_dir参数以支持CPU卸载
            if self.device == "cuda":
                load_kwargs["offload_folder"] = temp_offload_dir
                load_kwargs["offload_buffers"] = True
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **load_kwargs
            )
            
            if self.device == "cpu":
                self.base_model = self.base_model.to("cpu")
            
            if self.device == "cuda":
                logger.info(f"   ✓ 模型已加载到GPU，当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            else:
                logger.info("   ✓ 模型已加载到CPU")
            
            # 加载tokenizer
            logger.info("[2/3] 正在加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            logger.info("   ✓ Tokenizer加载完成")
            
            # 加载LoRA适配器
            logger.info("[3/3] 正在加载LoRA适配器（中医微调权重）...")
            
            # 尝试使用临时卸载目录加载LoRA适配器
            try:
                self.model = PeftModel.from_pretrained(
                    self.base_model, 
                    adapter_path,
                    offload_dir=temp_offload_dir
                )
            except Exception as e:
                # 如果offload_dir失败，尝试不使用offload_dir
                logger.warning(f"使用offload_dir失败，尝试直接加载: {e}")
                try:
                    self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
                except Exception as e2:
                    logger.error(f"LoRA适配器加载完全失败: {e2}")
                    raise
            
            if self.device == "cuda":
                logger.info("   ✓ LoRA适配器加载完成")
                logger.info(f"   总显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                logger.info(f"   剩余显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}GB")
            else:
                logger.info("   ✓ LoRA适配器加载完成")
            
            logger.info("=" * 60)
            logger.info("✓ 模型加载完成！")
            logger.info("=" * 60)
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            return False
    
    def generate(self,
                query: str,
                system_prompt: str = None,
                max_new_tokens: int = 512,
                temperature: float = 0.5,  # 稍微调高默认温度
                top_p: float = 0.6,        # 保持较低的默认top_p
                repetition_penalty: float = 1.1,
                mode: str = "default",  # "vector", "kg", "hybrid", "default"
                **kwargs) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户问题
            system_prompt: 系统提示（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            repetition_penalty: 重复惩罚
            mode: 生成模式 ("vector", "kg", "hybrid", "default")
            
        Returns:
            Dict包含answer和metadata
        """
        # 根据模式设置生成参数
        # 统一使用评估系统的生成参数（更快更稳）：
        # max_new_tokens 512, temperature 0.1, top_p 0.4, num_beams 3, do_sample False,
        # repetition_penalty 1.3, length_penalty 1.0, min_new_tokens 20, no_repeat_ngram_size 5,
        # early_stopping True, use_cache True
        generation_params = {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.4,
            "num_beams": 3,
            "do_sample": False,
            "repetition_penalty": 1.3,
            "length_penalty": 1.0,
            "min_new_tokens": 20,
            "no_repeat_ngram_size": 5,
            "early_stopping": True,
            "use_cache": True
        }
        else:
            # 默认模式：使用传入的参数
            generation_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": True
            }
        
        # 合并kwargs
        generation_params.update(kwargs)
        
        if not self._initialized or self.model is None:
            raise RuntimeError("模型未初始化，请先调用load_model()")
        
        start_time = time.time()
        
        try:
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            # 应用chat模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_length = model_inputs.input_ids.shape[1]
            
            # 生成
            with torch.no_grad():
                # 设置pad_token_id和eos_token_id
                if self.tokenizer.pad_token is None:
                    generation_params["pad_token_id"] = self.tokenizer.eos_token_id
                else:
                    generation_params["pad_token_id"] = self.tokenizer.pad_token_id
                generation_params["eos_token_id"] = self.tokenizer.eos_token_id
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_params
                )
            
            # 解码输出（只保留新生成的部分）
            generated_ids = [
                output_ids[input_length:] for output_ids in generated_ids
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 过滤<think>标签及其内容
            response = self._filter_think_tags(response)
            
            generation_time = time.time() - start_time
            tokens_generated = generated_ids[0].shape[0]
            
            # 更新统计
            self._update_stats(True, generation_time, tokens_generated)
            
            # 返回结果
            result = {
                "answer": response,
                "metadata": {
                    "generation_time": round(generation_time, 2),
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": round(tokens_generated / generation_time, 2),
                    "model": "qwen1.5-1.8b-tcm",
                    "mode": mode,
                    "temperature": generation_params.get("temperature", temperature),
                    "top_p": generation_params.get("top_p", top_p)
                }
            }
            
            # 添加GPU信息（如果使用GPU）
            if self.device == "cuda":
                result["metadata"]["gpu_memory_used"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            
            return result
            
        except Exception as e:
            self._update_stats(False, 0, 0)
            logger.error(f"生成回答失败: {e}", exc_info=True)
            raise
    
    @staticmethod
    def _filter_think_tags(text: str) -> str:
        """
        过滤<think>标签及其内容
        
        Args:
            text: 原始文本
            
        Returns:
            过滤后的文本
        """
        # 移除<think>...</think>标签及其内容（支持多行）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除可能遗留的单个标签
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        
        # 移除可能的思考性开头语句
        think_patterns = [
            r'^让我.*?[。\n]',
            r'^首先.*?[。\n]', 
            r'^根据.*?分析.*?[。\n]',
            r'^需要.*?思考.*?[。\n]'
        ]
        for pattern in think_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # 清理多余的空行和空格
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _update_stats(self, success: bool, generation_time: float, tokens: int):
        """更新统计信息"""
        self._stats["total_requests"] += 1
        if success:
            self._stats["successful_requests"] += 1
            self._stats["total_tokens_generated"] += tokens
            
            # 更新平均生成时间
            prev_avg = self._stats["average_generation_time"]
            count = self._stats["successful_requests"]
            self._stats["average_generation_time"] = (prev_avg * (count - 1) + generation_time) / count
        else:
            self._stats["failed_requests"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["initialized"] = self._initialized
        stats["device"] = self.device
        
        if self.device == "cuda" and self._initialized:
            stats["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            stats["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        
        return stats
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
    
    def unload_model(self):
        """卸载模型，释放内存"""
        if self.model:
            del self.model
            self.model = None
        if self.base_model:
            del self.base_model
            self.base_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("模型已卸载")


# 全局单例
_model_service = None

def get_model_service() -> ModelService:
    """获取模型服务单例"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

