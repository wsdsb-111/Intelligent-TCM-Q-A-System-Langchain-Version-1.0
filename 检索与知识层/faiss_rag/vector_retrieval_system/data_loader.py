"""
数据加载器模块 - 支持CSV/JSON/Arrow格式数据读取和处理
"""

import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Union, Iterator
import logging
from .config import DATASET_CONFIG, TEXT_CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    """统一数据加载器，支持多种格式的中医数据集"""
    
    def __init__(self):
        self.csv_path = DATASET_CONFIG["csv_path"]
        self.json_path = DATASET_CONFIG["json_path"] 
        self.arrow_path = DATASET_CONFIG["arrow_path"]
        
    def load_csv_data(self, file_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """加载CSV格式的中医数据集"""
        if file_path is None:
            file_path = self.csv_path
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载CSV数据，共{len(df)}条记录")
            
            # 转换为标准格式
            data = []
            for _, row in df.iterrows():
                data.append({
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),
                    "source": "csv"
                })
            return data
            
        except Exception as e:
            logger.error(f"加载CSV数据失败: {e}")
            return []
    
    def load_json_data(self, file_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """加载JSON格式的中医数据集"""
        if file_path is None:
            file_path = self.json_path
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"成功加载JSON数据，共{len(data)}条记录")
            
            # 标准化数据格式
            standardized_data = []
            for item in data:
                standardized_data.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                    "source": "json"
                })
            return standardized_data
            
        except Exception as e:
            logger.error(f"加载JSON数据失败: {e}")
            return []
    
    def load_arrow_data(self, directory_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """加载Arrow格式的中医数据集"""
        if directory_path is None:
            directory_path = self.arrow_path
            
        try:
            data = []
            arrow_files = list(Path(directory_path).glob("*.arrow"))
            
            for arrow_file in arrow_files:
                logger.info(f"正在加载Arrow文件: {arrow_file}")
                
                # 读取Arrow文件
                table = pa.ipc.open_file(arrow_file).read_all()
                df = table.to_pandas()
                
                # 转换为标准格式
                for _, row in df.iterrows():
                    data.append({
                        "instruction": row.get("instruction", ""),
                        "input": row.get("input", ""),
                        "output": row.get("output", ""),
                        "source": "arrow"
                    })
            
            logger.info(f"成功加载Arrow数据，共{len(data)}条记录")
            return data
            
        except Exception as e:
            logger.error(f"加载Arrow数据失败: {e}")
            return []
    
    def load_all_data(self) -> List[Dict[str, Any]]:
        """加载所有可用的数据集"""
        all_data = []
        
        # 加载CSV数据
        csv_data = self.load_csv_data()
        all_data.extend(csv_data)
        
        # 加载JSON数据
        json_data = self.load_json_data()
        all_data.extend(json_data)
        
        # 加载Arrow数据
        arrow_data = self.load_arrow_data()
        all_data.extend(arrow_data)
        
        logger.info(f"总共加载了{len(all_data)}条中医数据记录")
        return all_data
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 清理文本
        text = text.strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # 去除多余空格
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """文本分块处理"""
        if chunk_size is None:
            chunk_size = TEXT_CONFIG["chunk_size"]
        if overlap is None:
            overlap = TEXT_CONFIG["chunk_overlap"]
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句号、问号、感叹号处分割
            if end < len(text):
                for i in range(end, max(start + chunk_size // 2, start), -1):
                    if text[i] in ['。', '？', '！', '.', '?', '!']:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if len(chunk) >= TEXT_CONFIG["min_chunk_size"]:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def prepare_for_embedding(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为向量化准备数据"""
        processed_data = []
        
        for item in data:
            # 合并instruction、input和output作为主要文本内容（若instruction/input为空，output通常含实际文本）
            main_text = f"{item.get('instruction','')} {item.get('input','')} {item.get('output','')}".strip()
            main_text = self.preprocess_text(main_text)
            
            if not main_text:
                continue
            
            # 分块处理
            chunks = self.chunk_text(main_text)

            # 如果由于 min_chunk_size 或分割逻辑导致没有生成任何块，但 main_text 非空，保底返回整个文本
            if not chunks and main_text:
                chunks = [main_text]
            
            for i, chunk in enumerate(chunks):
                processed_data.append({
                    "text": chunk,
                    "output": item.get("output", ""),
                    "source": item.get("source", "unknown"),
                    "original_id": f"{item.get('source', 'unknown')}_{len(processed_data)}",
                    "chunk_id": i,
                    "metadata": {
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                        "source": item.get("source", "unknown")
                    }
                })
        
        logger.info(f"预处理完成，生成{len(processed_data)}个文本块")
        return processed_data
