# 768维向量数据库构建指南

## 概述

本指南介绍如何使用GTE Base模型（768维）构建向量数据库，确保与现有512维数据库格式完全一致，实现无缝衔接。

## 文件说明

### 核心脚本

1. **`构建768维向量数据库.py`** - 主构建脚本
   - 使用GTE Base模型生成768维向量
   - 输出格式与512维数据库完全一致
   - 支持GPU加速和批量处理

2. **`测试768维数据库兼容性.py`** - 兼容性测试脚本
   - 验证文件结构和格式
   - 对比两个数据库的一致性
   - 功能测试和性能验证

3. **`768维数据库使用示例.py`** - 使用示例
   - 展示如何无缝切换数据库
   - 对比测试两个数据库
   - 提供最佳实践建议

## 快速开始

### 步骤1: 构建768维数据库

```bash
cd 检索与知识层\faiss_rag
python 构建768维向量数据库.py
```

### 步骤2: 验证兼容性

```bash
python 测试768维数据库兼容性.py
```

### 步骤3: 测试使用

```bash
python 768维数据库使用示例.py
```

## 数据库结构

构建完成后，768维数据库将包含以下文件：

```
向量数据库_768维/
├── faiss.index      # Faiss索引文件
├── metadata.pkl     # 元数据文件
└── documents.json   # 文档内容文件
```

## 无缝切换方法

### 方法1: 修改FaissManager初始化

```python
# 使用512维数据库
faiss_manager = FaissManager(
    persist_directory="向量数据库_简单查询",
    dimension=512
)

# 切换到768维数据库
faiss_manager = FaissManager(
    persist_directory="向量数据库_768维",
    dimension=768
)
```

### 方法2: 配置文件切换

```python
# 配置选择
DATABASE_CONFIG = {
    "512维": {
        "path": "向量数据库_简单查询",
        "dimension": 512,
        "model": "nlp_gte_sentence-embedding_chinese-small"
    },
    "768维": {
        "path": "向量数据库_768维", 
        "dimension": 768,
        "model": "nlp_gte_sentence-embedding_chinese-base"
    }
}

# 动态切换
selected_db = DATABASE_CONFIG["768维"]
faiss_manager = FaissManager(
    persist_directory=selected_db["path"],
    dimension=selected_db["dimension"]
)
```

## 性能对比

| 特性 | 512维数据库 | 768维数据库 |
|------|-------------|-------------|
| 模型 | GTE Small | GTE Base |
| 向量维度 | 512 | 768 |
| 文件大小 | 较小 | 较大 |
| 计算速度 | 较快 | 较慢 |
| 语义理解 | 良好 | 更好 |
| 内存占用 | 较低 | 较高 |

## 注意事项

1. **模型路径**: 确保GTE Base模型已正确下载到指定路径
2. **内存需求**: 768维模型需要更多内存，建议至少8GB RAM
3. **GPU支持**: 支持GPU加速，但CPU模式也能正常工作
4. **格式兼容**: 输出格式与512维数据库完全一致，可以无缝切换

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型文件完整性

2. **内存不足**
   - 减少BATCH_SIZE
   - 使用CPU模式
   - 增加系统内存

3. **GPU问题**
   - 检查CUDA安装
   - 使用CPU模式作为备选

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用小数据集测试
MAX_SAMPLES = 1000  # 限制处理数量
```

## 最佳实践

1. **生产环境**: 建议使用768维模型获得更好的语义理解
2. **开发测试**: 可以使用512维模型快速验证
3. **资源受限**: 根据硬件条件选择合适的模型
4. **性能优化**: 使用GPU加速和批量处理

## 更新日志

- **v1.0**: 初始版本，支持768维数据库构建
- **v1.1**: 添加兼容性测试和性能对比
- **v1.2**: 优化内存使用和错误处理

## 技术支持

如有问题，请检查：
1. 模型文件完整性
2. 依赖包版本
3. 系统资源充足性
4. 日志文件错误信息
