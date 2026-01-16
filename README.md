# Voice Journal & Digital Twin

整合的语音日记和智能记忆系统 - 通过语音转写快速记录想法，并通过 RAG 技术实现智能对话和记忆检索。

## 🎯 项目概述

这是一个集成了语音记录管理和智能对话功能的完整系统：

- **🎤 Voice Journal**: 语音转写记录管理，支持录音、编辑、查看
- **🤖 Digital Twin**: 基于 RAG 的智能对话系统，可以回忆过去、查找记录、分析模式
- **📊 向量索引**: 自动构建和维护 FAISS 向量索引，支持增量更新

## ✨ 功能特性

### Voice 记录管理
- 🎤 **语音转写记录**: 支持通过 API 或脚本快速添加语音转写文本
- 📝 **记录管理**: Web 界面查看、编辑、复制所有记录
- 🔄 **自动同步**: 新记录自动同步到 RAG 系统并触发索引重建
- 📊 **记录统计**: 显示记录总数和最近记录

### RAG 智能对话
- 💬 **智能问答**: 基于向量检索的对话系统，可以回答关于历史记录的问题
- 🔍 **记忆检索**: 支持按日期、关键词、内容检索历史记录
- 📅 **日期过滤**: 支持"最近两天"、"上个月"等自然语言日期查询
- 🧠 **上下文理解**: 支持多轮对话，理解上下文和代词引用

### 索引管理
- 🔄 **增量索引**: 新记录自动触发增量索引更新
- 🔨 **全量重建**: 支持手动或定时全量索引重建
- 📈 **进度显示**: 实时显示索引重建进度和状态
- ⚙️ **自动维护**: 定时检查并重建索引（每30分钟）

## 📁 项目结构

```
vector_indexer/
├── app.py                    # FastAPI Web 应用（整合所有功能）
├── main.py                   # RAG 对话逻辑（ReAct Agent）
├── retriever.py              # 向量检索器
├── indexer.py                # 全量索引构建工具
├── incremental_indexer.py    # 增量索引更新工具
├── process_voice.py          # 语音记录处理脚本
├── voice_records.json        # Voice 记录存储文件
├── all_chunks.json           # RAG 系统数据源（包含所有记录）
├── my_history.index          # FAISS 向量索引文件
├── chunks_metadata.json      # 索引元数据文件
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
└── [文档]
    ├── QUICKSTART.md         # 快速启动指南
    ├── WEB_QUICKSTART.md     # Web 界面使用指南
    ├── AGENTIC_RAG_GUIDE.md  # RAG 系统指南
    └── ...
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd vector_indexer
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API key
# AI_BUILDER_TOKEN=your_api_key_here
```

或使用环境变量：

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
```

### 3. 构建初始索引（首次使用）

```bash
# 确保 all_chunks.json 文件存在
python indexer.py
```

### 4. 启动 Web 服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 访问 Web 界面

- 🎤 **录音页面**: http://localhost:8000/
- 📝 **记录列表**: http://localhost:8000/records
- 🤖 **智能对话**: http://localhost:8000/chat
- ⚙️ **设置页面**: http://localhost:8000/settings
- 📚 **API 文档**: http://localhost:8000/docs

## 📖 使用指南

### 添加语音记录

#### 方法 1: Web 界面
1. 访问 http://localhost:8000/
2. 点击录音按钮，使用浏览器语音识别
3. 或直接在文本框中输入内容
4. 点击保存

#### 方法 2: API 调用
```bash
# POST 请求
curl -X POST http://localhost:8000/api/voice \
  -H "Content-Type: application/json" \
  -d '{"content": "今天和朋友吃饭，聊到他准备换工作"}'

# GET 请求（更简单）
curl "http://localhost:8000/api/voice/add?content=这是一条测试记录"
```

#### 方法 3: 命令行脚本
```bash
python process_voice.py "今天和朋友吃饭，聊到他准备换工作"
```

### 智能对话

访问 http://localhost:8000/chat

- "最近两天有什么记录？"
- "2024年6月2日我在做什么让我感到开心？"
- "我和朋友一起做了什么？"
- "上个月我提到了哪些人？"

### 索引管理

#### 手动重建索引
1. 访问 http://localhost:8000/settings
2. 点击"手动重建索引"按钮
3. 查看实时进度

#### 自动索引更新
- 新记录添加后自动触发增量索引
- 定时任务每30分钟检查并重建（作为兜底）

## 🔧 API 端点

### Voice 记录 API

- `GET /api/records` - 获取所有记录
- `POST /api/voice` - 添加语音记录（JSON）
- `GET /api/voice/add?content=文本` - 添加语音记录（GET，适合快捷指令）
- `PUT /api/voice/{record_id}` - 更新记录

### RAG 对话 API

- `POST /chat` - 智能对话（支持对话历史）
- `POST /retrieve` - 向量检索

### 索引管理 API

- `GET /api/index-status` - 获取索引重建状态
- `POST /api/rebuild-index` - 手动触发索引重建

详细 API 文档：http://localhost:8000/docs

## 📊 数据格式

### Voice 记录格式

```json
{
  "id": "voice_20260115_1208",
  "source": "voice",
  "date": "2026-01-15",
  "time": "12:08",
  "content": "测试：这是一条测试记录"
}
```

### RAG Chunk 格式

```json
{
  "id": "voice_20260115_1208",
  "source": "voice",
  "date": "2026-01-15",
  "content": "测试：这是一条测试记录"
}
```

## 🔐 环境变量

### 必需变量

- `AI_BUILDER_TOKEN`: AI Builder API Token（必需）

### 可选变量

- `PORT`: Web 服务端口（默认: 8000）
- `INDEX_PATH`: 索引文件路径（默认: `my_history.index`）
- `METADATA_PATH`: 元数据文件路径（默认: `chunks_metadata.json`）
- `LOG_DIR`: 日志目录（默认: `logs`）

## 🌐 云端部署

项目已准备好部署到 AI Builder Space 或其他云平台。

### 部署步骤

1. 确保代码已推送到 GitHub 公共仓库
2. 使用 Dockerfile 构建镜像
3. 设置环境变量 `AI_BUILDER_TOKEN`
4. 部署后访问公网 URL

详细部署指南请参考 `DEPLOYMENT.md`（如果存在）。

## 🛠️ 开发

### 本地开发

```bash
# 启动开发服务器（支持热重载）
uvicorn app:app --reload --port 8000

# 运行测试
python test_agent.py
```

### 日志

- 应用日志: `logs/app.log`
- Agent 日志: `logs/agent.log`
- 日志自动轮转（单个文件最大 10MB，保留 5 个备份）

## 📚 文档

- [快速启动指南](QUICKSTART.md)
- [Web 界面使用指南](WEB_QUICKSTART.md)
- [RAG 系统指南](AGENTIC_RAG_GUIDE.md)
- [性能审计报告](PERFORMANCE_AUDIT.md)
- [项目总结](PROJECT_SUMMARY.md)

## 🔄 更新日志

### 最新版本（整合版）

- ✅ 整合 voice_journal 和 vector_indexer 功能
- ✅ 统一 Web 界面（录音、记录、对话、设置）
- ✅ 自动索引同步和重建
- ✅ 支持增量索引更新
- ✅ 索引重建进度显示
- ✅ 改进的 RAG 检索逻辑（支持 voice 记录）

## ⚠️ 注意事项

1. **API Key 安全**: 
   - 不要将 API key 提交到 Git 仓库
   - 使用 `.env` 文件管理（已添加到 `.gitignore`）

2. **索引文件大小**: 
   - `my_history.index` 可能较大（取决于数据量）
   - `all_chunks.json` 会随着记录增加而增长

3. **首次使用**: 
   - 需要先运行 `python indexer.py` 构建初始索引
   - 之后新记录会自动触发增量索引

## 📝 许可证

MIT

## 🙏 致谢

- 使用 [FAISS](https://github.com/facebookresearch/faiss) 进行向量检索
- 使用 [FastAPI](https://fastapi.tiangolo.com/) 构建 Web 服务
- 使用 [OpenAI API](https://platform.openai.com/) 进行 Embeddings 和对话
