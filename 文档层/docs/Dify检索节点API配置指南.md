# Dify检索节点API配置指南

本文档说明如何在Dify工作流中配置检索相关的HTTP请求节点。

## 1. 纯向量检索节点

### API地址
```
POST http://host.docker.internal:8000/api/dify/retrieve_documents
```

### 请求头（HEADERS）
| 键 | 值 |
|---|---|
| Content-Type | application/json |

### Body（JSON）
```json
{
  "query": "{{#start.query#}}",
  "router_type": "vector_only"
}
```

### 说明
- 召回3个向量文档
- 用于生成3个文档
- `query` 使用Dify变量 `{{#start.query#}}` 传递用户查询

---

## 2. 混合检索节点

### API地址
```
POST http://host.docker.internal:8000/api/dify/retrieve_documents
```

### 请求头（HEADERS）
| 键 | 值 |
|---|---|
| Content-Type | application/json |

### Body（JSON）
```json
{
  "query": "{{#start.query#}}",
  "router_type": "hybrid"
}
```

### 说明
- 召回5向量+5图谱（共10个文档）
- 用于生成3向量+5图谱（共8个文档）
- `query` 使用Dify变量 `{{#start.query#}}` 传递用户查询

---

## 3. 查询扩展与重排序节点

### API地址
```
POST http://host.docker.internal:8000/api/dify/expand_and_rerank
```

### 请求头（HEADERS）
| 键 | 值 |
|---|---|
| Content-Type | application/json |

### Body（JSON）
```json
{
  "query": "{{#start.query#}}",
  "documents": {{#检索节点.documents#}},
  "parallel": true
}
```

### 说明
- 使用text2vec查询扩展，生成相关查询
- 使用bge-reranker重排序，优化文档相关性
- `documents` 使用前一个检索节点的输出 `{{#检索节点.documents#}}`
- `parallel` 设置为 `true` 表示并行执行扩展和重排序

---

## 注意事项

1. **Docker容器访问**：如果Dify在Docker容器中运行，使用 `host.docker.internal` 访问宿主机服务
2. **本地访问**：如果Dify和FastAPI在同一台机器且不在Docker中，可使用 `localhost` 或 `127.0.0.1`
3. **变量引用**：`{{#start.query#}}` 是Dify变量语法，会自动替换为用户输入
4. **超时设置**：建议在Dify HTTP节点中设置超时为30秒或更长


























