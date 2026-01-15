# Agentic RAG 使用指南

## 架构说明

本项目实现了 Agentic RAG（智能检索增强生成），包含三个核心组件：

1. **`retriever.py`** - 向量检索器，封装 FAISS 索引加载和搜索逻辑
2. **`app.py`** - FastAPI 检索服务，提供 HTTP 接口
3. **`main.py`** - Agent 主程序，集成检索工具，实现智能对话

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
```

### 3. 确保索引文件存在

确保以下文件存在：
- `my_history.index` - FAISS 索引文件
- `chunks_metadata.json` - 元数据文件

如果还没有，先运行 `indexer.py` 生成索引。

### 4. 启动检索服务

**方法1：直接运行**
```bash
python app.py
```

**方法2：使用启动脚本**
```bash
chmod +x start_retriever.sh
./start_retriever.sh
```

**方法3：使用 uvicorn**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

服务启动后，访问 http://localhost:8000/docs 查看 API 文档。

### 5. 启动 Agent

在另一个终端窗口：

```bash
python main.py
```

## 使用示例

### 启动 Agent 后，可以这样提问：

```
你: 去年我去过哪里？

Digital Twin: [会自动调用 search_memory 工具，查找去年的记录，然后回答]

你: 我记得之前写过关于情绪管理的笔记

Digital Twin: [会自动搜索相关笔记，并展示内容]

你: 上个月我有什么重要的决定？

Digital Twin: [会自动搜索上个月的记录]
```

## API 接口说明

### POST /retrieve

检索接口，供 Agent 调用。

**请求体：**
```json
{
  "query": "搜索查询文本",
  "max_results": 5,
  "date_filter": "last_year"  // 可选，支持多种格式：
  // - 具体日期: "YYYY-MM-DD" (如 "2024-06-02")
  // - 年月: "YYYY-MM" (如 "2024-03")
  // - 年份: "YYYY" (如 "2024")
  // - 相对时间: "last_year", "last_month", "last_week"
  // - N个月前: "N_months_ago" (如 "3_months_ago")
  // - N天前: "N_days_ago" (如 "30_days_ago")
}
```

**响应：**
```json
{
  "results": [
    {
      "id": "notion_1",
      "source": "notion",
      "date": "2024-03-27",
      "content": "...",
      "distance": 0.123
    }
  ],
  "query": "搜索查询文本",
  "total_results": 1
}
```

### GET /health

健康检查接口，返回服务状态和缓存统计信息。

**响应：**
```json
{
  "status": "healthy",
  "retriever_loaded": true,
  "index_path": "my_history.index",
  "metadata_path": "chunks_metadata.json",
  "cache_stats": {
    "cache_size": 42,
    "max_cache_size": 1000,
    "cache_hits": 150,
    "cache_misses": 50,
    "hit_rate": 75.0
  }
}
```

### GET /stats

获取检索服务详细统计信息，包括索引大小、缓存状态等。

## 配置说明

### 环境变量

- `AI_BUILDER_TOKEN` - AI Builder API 密钥（必需）
- `INDEX_PATH` - FAISS 索引文件路径（默认：`my_history.index`）
- `METADATA_PATH` - 元数据文件路径（默认：`chunks_metadata.json`）
- `RETRIEVER_URL` - 检索服务 URL（默认：`http://localhost:8000`）
- `PORT` - FastAPI 服务端口（默认：8000）

## 工作原理

1. **检索服务（app.py）**：
   - 启动时加载 FAISS 索引和元数据到内存
   - 提供 `/retrieve` 接口，接收查询文本
   - 调用 AI Builder API 生成查询文本的 embedding
   - 使用 FAISS 进行相似度搜索
   - 支持日期过滤（相对时间和绝对时间）

2. **Agent（main.py）**：
   - 每次对话时动态生成 System Prompt（包含当前日期）
   - 调用 AI Builder API，传入工具定义
   - 当模型决定调用 `search_memory` 工具时：
     - 解析工具调用参数
     - 调用检索服务的 `/retrieve` 接口
     - 将结果返回给模型
     - 模型基于检索结果生成最终回复

3. **时间感知**：
   - System Prompt 中包含当前日期
   - 当用户问"去年"时，Agent 知道当前年份，自动计算"去年"是哪一年
   - 支持多种时间过滤格式：
     - 相对时间：`last_year`, `last_month`, `last_week`
     - N个月前：`3_months_ago`, `6_months_ago` 等
     - N天前：`30_days_ago`, `90_days_ago` 等
     - 具体日期：`YYYY-MM-DD`
     - 年月：`YYYY-MM`（如 `2024-03`）
     - 年份：`YYYY`（如 `2024`）

4. **性能优化**：
   - **Embedding 缓存**：自动缓存查询文本的 embedding，减少重复 API 调用
   - **结果去重**：自动去除重复的检索结果
   - **智能排序**：按相似度从高到低排序结果

## 故障排除

### 检索服务无法启动

1. 检查 `AI_BUILDER_TOKEN` 是否设置
2. 检查索引文件和元数据文件是否存在
3. 检查端口 8000 是否被占用

### Agent 无法调用检索工具

1. 确保检索服务正在运行（访问 http://localhost:8000/health）
2. 检查 `RETRIEVER_URL` 环境变量是否正确
3. 查看 Agent 终端的错误信息

### 搜索结果为空

1. 检查索引文件是否是最新的
2. 尝试更通用的查询词
3. 检查日期过滤条件是否正确

## 已完成的优化

- [x] **检索结果缓存**：Embedding 缓存机制，显著减少重复 API 调用
- [x] **复杂时间查询**：支持 "N_months_ago"、"YYYY-MM"、"YYYY" 等多种格式
- [x] **结果去重和排序**：自动去除重复结果，按相似度排序
- [x] **检索日志和监控**：详细的日志记录和 `/stats` 监控端点

## 下一步优化

- [ ] 支持多轮对话的上下文理解
- [ ] 添加检索结果的相关性评分
- [ ] 支持多索引切换
- [ ] 添加单元测试
