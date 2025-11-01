"""
API Schemas模块
导出所有Pydantic数据模型
"""

# 为了避免循环导入，使用importlib.util从文件直接导入
# 注意：不能使用 from ..schemas import，因为会形成循环导入
import importlib.util
import sys
from pathlib import Path

# 获取schemas.py的路径（父目录的同级文件）
schemas_py_path = Path(__file__).parent.parent / "schemas.py"

# 使用importlib.util从文件路径直接导入，避免循环导入
_schemas_module_name = "middle_api_schemas_core"
if _schemas_module_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(_schemas_module_name, schemas_py_path)
    _schemas_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_schemas_module)
    sys.modules[_schemas_module_name] = _schemas_module
else:
    _schemas_module = sys.modules[_schemas_module_name]

# 重新导出标准API schemas
QueryRequest = _schemas_module.QueryRequest
QueryResponse = _schemas_module.QueryResponse
RetrievalRequest = _schemas_module.RetrievalRequest
RetrievalResponse = _schemas_module.RetrievalResponse
RetrievalResultSchema = _schemas_module.RetrievalResultSchema
QueryMetadata = _schemas_module.QueryMetadata
HealthStatus = _schemas_module.HealthStatus
ErrorResponse = _schemas_module.ErrorResponse
MultimodalRequest = _schemas_module.MultimodalRequest
MultimodalResponse = _schemas_module.MultimodalResponse
BatchQueryRequest = _schemas_module.BatchQueryRequest
BatchQueryResponse = _schemas_module.BatchQueryResponse

# 导出Dify节点schemas
from .dify_schemas import (
    RouterType,
    RetrievalConfigModel,
    DifyRetrievalRequest, DifyRetrievalResponse,
    DifyExpandRerankRequest, DifyExpandRerankResponse,
    DifyGenerateAnswerRequest, DifyGenerateAnswerResponse,
    DocumentSchema,
    GenerationParams
)

__all__ = [
    # 标准API schemas
    "QueryRequest", "QueryResponse",
    "RetrievalRequest", "RetrievalResponse",
    "RetrievalResultSchema",
    "QueryMetadata",
    "HealthStatus",
    "ErrorResponse",
    # Dify节点schemas
    "RouterType",
    "RetrievalConfigModel",
    "DifyRetrievalRequest", "DifyRetrievalResponse",
    "DifyExpandRerankRequest", "DifyExpandRerankResponse",
    "DifyGenerateAnswerRequest", "DifyGenerateAnswerResponse",
    "DocumentSchema",
    "GenerationParams"
]

