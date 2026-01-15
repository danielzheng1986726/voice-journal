# 日志功能使用指南

## 日志配置

项目已配置完整的日志系统，方便定位错误和调试。

### 日志文件位置

- **应用日志**: `logs/app.log` - FastAPI 服务的所有日志
- **Agent 日志**: `logs/agent.log` - Agent 对话和工具调用的日志
- **检索器日志**: 输出到 `logs/app.log`（通过应用日志器）

### 日志级别

- **DEBUG**: 详细的调试信息（向量维度、API 调用详情等）
- **INFO**: 一般信息（请求、响应、工具调用等）
- **ERROR**: 错误信息（异常、失败等）
- **WARNING**: 警告信息

### 日志输出

- **控制台**: INFO 级别及以上的日志（实时查看）
- **文件**: DEBUG 级别及以上的日志（完整记录）

## 日志内容

### 1. 应用启动日志

```
2026-01-13 14:30:00 - vector_indexer - INFO - ============================================================
2026-01-13 14:30:00 - vector_indexer - INFO - 🚀 正在启动检索服务...
2026-01-13 14:30:00 - vector_indexer - INFO -    索引文件: my_history.index
2026-01-13 14:30:00 - vector_indexer - INFO -    元数据文件: chunks_metadata.json
2026-01-13 14:30:00 - vector_indexer.retriever - INFO - 📖 正在加载索引: my_history.index
2026-01-13 14:30:00 - vector_indexer.retriever - INFO - ✅ 索引加载完成，向量数量: 2165, 维度: 1536
```

### 2. HTTP 请求日志

```
2026-01-13 14:35:00 - vector_indexer - INFO - 🌐 POST /retrieve - Client: 127.0.0.1
2026-01-13 14:35:00 - vector_indexer - INFO - 📥 检索请求: query='完成课程PPT', max_results=5, date_filter=2024-06-02
2026-01-13 14:35:01 - vector_indexer - INFO - ✅ 检索完成: 找到 1 条结果，耗时 0.85秒
2026-01-13 14:35:01 - vector_indexer - INFO - ✅ POST /retrieve - Status: 200 - 耗时: 0.852秒
```

### 3. Agent 对话日志

```
2026-01-13 14:40:00 - vector_indexer.agent - INFO - 💬 开始对话: message='2024年6月2日我在做什么让我感到开心？' (长度: 25)
2026-01-13 14:40:00 - vector_indexer.agent - DEBUG - 📝 System Prompt 已生成，当前日期: 2026-01-13
2026-01-13 14:40:01 - vector_indexer.agent - DEBUG - 🤖 调用 LLM API: model=supermind-agent-v1, messages_count=2
2026-01-13 14:40:02 - vector_indexer.agent - INFO - 🔧 检测到 1 个工具调用
2026-01-13 14:40:02 - vector_indexer.agent - INFO - 🔧 工具调用: search_memory
2026-01-13 14:40:02 - vector_indexer.agent - INFO - 🔍 调用检索服务: query='完成课程PPT', max_results=5, date_filter=2024-06-02
2026-01-13 14:40:03 - vector_indexer.agent - INFO - ✅ 检索成功: 找到 1 条结果
2026-01-13 14:40:04 - vector_indexer.agent - INFO - ✅ 对话完成: 响应长度=256
```

### 4. 错误日志

```
2026-01-13 14:45:00 - vector_indexer - ERROR - ❌ 检索失败 (耗时 0.12秒): API调用失败: Connection timeout
2026-01-13 14:45:00 - vector_indexer - ERROR - Traceback (most recent call last):
  File "/path/to/app.py", line 145, in retrieve
    ...
```

## 查看日志

### 实时查看日志

```bash
# 查看应用日志
tail -f logs/app.log

# 查看 Agent 日志
tail -f logs/agent.log

# 查看最近的错误
grep ERROR logs/app.log | tail -20

# 查看特定时间的日志
grep "2026-01-13 14:30" logs/app.log
```

### 搜索日志

```bash
# 搜索工具调用
grep "工具调用" logs/agent.log

# 搜索检索请求
grep "检索请求" logs/app.log

# 搜索错误
grep -i error logs/*.log

# 搜索特定查询
grep "完成课程PPT" logs/app.log
```

## 日志配置

### 修改日志级别

在代码中修改日志级别：

```python
# 在 app.py 或 main.py 中
logger.setLevel(logging.DEBUG)  # 显示所有日志
logger.setLevel(logging.INFO)   # 只显示 INFO 及以上
logger.setLevel(logging.ERROR)  # 只显示错误
```

### 修改日志目录

通过环境变量设置：

```bash
export LOG_DIR="/path/to/custom/logs"
python app.py
```

### 日志轮转（可选）

可以配置日志轮转，避免日志文件过大：

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

## 常见问题排查

### 1. 检索服务无响应

查看日志：
```bash
grep "检索" logs/app.log | tail -20
```

### 2. Agent 工具调用失败

查看日志：
```bash
grep "工具调用" logs/agent.log | tail -20
```

### 3. API 调用失败

查看日志：
```bash
grep "API" logs/*.log | grep ERROR
```

### 4. 性能问题

查看耗时日志：
```bash
grep "耗时" logs/app.log | tail -20
```

## 最佳实践

1. **开发时**: 使用 DEBUG 级别，查看详细信息
2. **生产时**: 使用 INFO 级别，减少日志量
3. **定期清理**: 定期清理旧日志文件，避免占用过多空间
4. **监控错误**: 设置监控，及时发现 ERROR 级别的日志
