# Dify查询扩展与重排序Python节点代码

## 重要提示

**需要在Dify节点中添加第三个输入变量：`query`**
- 变量名：`query`
- 变量类型：String
- 变量值：`{{#start.query#}}`（从开始节点获取用户查询）

---

## Python代码

```python
import json
import requests

def main(content_vector: str, content_hybrid: str, query: str) -> dict:
    """
    调用查询扩展与重排序API
    
    Args:
        content_vector: 纯向量检索节点的HTTP响应（JSON字符串）
        content_hybrid: 混合检索节点的HTTP响应（JSON字符串）
        query: 用户查询（从{{#start.query#}}传入）
    
    Returns:
        处理后的文档结果
    """
    try:
        # 1. 解析HTTP响应，提取documents
        documents = None
        
        # 优先使用content_hybrid（混合检索），如果没有则使用content_vector（纯向量）
        content_to_use = content_hybrid if content_hybrid else content_vector
        
        if content_to_use:
            # 解析HTTP节点响应
            response_data = json.loads(content_to_use)
            
            # 判断输入格式：可能是完整的HTTP响应（有body字段）或直接是body的JSON
            if "body" in response_data:
                # 格式1：完整的HTTP响应 {"body": "{\"documents\":[...]}"}
                body_str = response_data["body"]
                if isinstance(body_str, str):
                    body_data = json.loads(body_str)
                else:
                    body_data = body_str
            else:
                # 格式2：直接是body的JSON {"documents":[...]}
                body_data = response_data
            
            # 提取documents
            if "documents" in body_data and body_data.get("documents"):
                documents = body_data["documents"]
        
        # 如果没有documents，返回错误
        if not documents:
            return {
                "Processed_content": json.dumps({
                    "success": False,
                    "error": "无法从检索结果中提取documents"
                }, ensure_ascii=False)
            }
        
        # 2. 调用查询扩展与重排序API
        api_url = "http://host.docker.internal:8000/api/dify/expand_and_rerank"
        headers = {
            "Content-Type": "application/json"
        }
        
        request_body = {
            "query": query,
            "documents": documents,
            "parallel": True
        }
        
        # 3. 发送请求
        response = requests.post(
            api_url,
            json=request_body,
            headers=headers,
            timeout=30
        )
        
        # 4. 处理响应
        if response.status_code == 200:
            result = response.json()
            return {
                "Processed_content": json.dumps(result, ensure_ascii=False)
            }
        else:
            return {
                "Processed_content": json.dumps({
                    "success": False,
                    "error": f"API调用失败: {response.status_code}",
                    "detail": response.text
                }, ensure_ascii=False)
            }
    
    except json.JSONDecodeError as e:
        return {
            "Processed_content": json.dumps({
                "success": False,
                "error": f"JSON解析失败: {str(e)}"
            }, ensure_ascii=False)
        }
    except requests.exceptions.RequestException as e:
        return {
            "Processed_content": json.dumps({
                "success": False,
                "error": f"HTTP请求失败: {str(e)}"
            }, ensure_ascii=False)
        }
    except Exception as e:
        return {
            "Processed_content": json.dumps({
                "success": False,
                "error": f"处理失败: {str(e)}"
            }, ensure_ascii=False)
        }
```

## 输入变量配置

在Dify节点中添加以下输入变量：

1. **content_vector**（已设置）
   - 类型：String
   - 值：`{{#HTTP纯向量检索节点.body#}}` 或 `{{#HTTP纯向量检索节点#}}`

2. **content_hybrid**（已设置）
   - 类型：String
   - 值：`{{#HTTP混合检索节点.body#}}` 或 `{{#HTTP混合检索节点#}}`

3. **query**（需要添加）
   - 类型：String
   - 值：`{{#start.query#}}`

## 输出变量

- **Processed_content**（已设置）
  - 类型：String
  - 内容：JSON格式的查询扩展和重排序结果

## 响应格式

成功时返回：
```json
{
  "success": true,
  "expanded_queries": ["扩展查询1", "扩展查询2", "扩展查询3"],
  "reranked_documents": [
    {
      "content": "文档内容",
      "source": "vector",
      "fused_score": 0.95,
      ...
    },
    ...
  ]
}
```
