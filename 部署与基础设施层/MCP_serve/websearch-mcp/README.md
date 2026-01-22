# WebSearch MCP 服务

该目录实现了教程版的网页搜索 MCP 服务，按照 Dify v1.6+ 原生支持的流程开发，提供独立的 `web_search` 工具用于实时网页检索。

## 目录结构

```
websearch-mcp/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastMCP 服务入口
│   └── search_client.py     # 博查搜索 API 封装
├── Dockerfile               # 容器镜像构建
├── docker-compose.yml       # 服务编排（加入 Dify 网络）
├── requirements.txt         # 依赖清单（包含 fastmcp==0.3.2）
├── sample.env               # 环境变量示例（启动前需复制为 .env）
└── README.md
```

## 使用步骤

1. **准备环境变量**
   ```bash
   cd 部署与基础设施层/MCP_serve/websearch-mcp
   cp sample.env .env
   # 编辑 .env，填入真实的 BOCHA_API_KEY 等配置
   ```

2. **构建并启动服务**
   ```bash
   docker compose up -d --build
   ```
   - 若 Dify 使用的 Docker 网络名称不同，请修改 `docker-compose.yml` 中的 `dify_network`。

3. **验证服务状态**
   ```bash
   curl http://127.0.0.1:8062/mcp/tools
   ```
   返回结果中应包含 `web_search` 工具定义。

4. **在 Dify 中接入**
   - 控制台 → 工具 → MCP → 添加服务
   - 传输方式：`Streamable HTTP`
   - 服务 URL：`http://websearch-mcp:8062/mcp`（或 `http://宿主机IP:8062/mcp`）
   - 验证成功后，将 `web_search` 工具添加到需要网页检索的智能体。

## 说明

- 该服务仅负责网页搜索，与主项目功能完全解耦。
- 若需扩展更多工具，可在 `app/main.py` 中继续声明新的 `@server.tool`。
- 运行日志可通过 `docker logs -f websearch-mcp` 查看。


