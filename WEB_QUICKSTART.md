# Web 界面快速启动指南

## 🚀 快速启动（3步）

### 1. 设置环境变量

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
```

**或者** 在 `~/.zshrc` 或 `~/.bashrc` 中添加（永久设置）：

```bash
echo 'export AI_BUILDER_TOKEN="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 2. 启动 Web 服务

**方法1：使用启动脚本（推荐）**

```bash
chmod +x start_retriever.sh
./start_retriever.sh
```

**方法2：直接运行 Python**

```bash
python app.py
```

**方法3：使用 uvicorn**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. 打开浏览器访问

访问：**http://localhost:8000**

你会看到一个漂亮的聊天界面，可以直接与 Digital Twin 守护者对话！

## 📋 完整启动流程

### 前置条件检查

1. **确保索引文件存在**
   ```bash
   ls -lh my_history.index chunks_metadata.json
   ```
   如果不存在，先运行 `python indexer.py` 生成索引

2. **检查 Python 依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **检查环境变量**
   ```bash
   echo $AI_BUILDER_TOKEN
   ```

### 启动服务

```bash
# 进入项目目录
cd /Users/xiaodongzheng/vector_indexer

# 设置环境变量（如果还没设置）
export AI_BUILDER_TOKEN='your_api_key_here'

# 启动服务
python app.py
```

### 访问 Web 界面

- **主界面（聊天）**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **统计信息**: http://localhost:8000/stats

## 🎯 使用示例

### 在 Web 界面中提问

1. 打开 http://localhost:8000
2. 在输入框中输入问题，例如：
   - "2024年6月2日我在做什么让我感到开心？"
   - "去年我去过哪里？"
   - "我给「内心的小孩」起的名字叫什么？"
3. 点击"发送"或按 Enter 键
4. AI 会自动检索记忆库并回答

### 对话历史功能

- ✅ 对话会自动保存到浏览器本地存储
- ✅ 刷新页面后对话历史不会丢失
- ✅ 点击"清空"按钮可以开始新对话
- ✅ 支持多轮对话，AI 能理解上下文

## 🔧 常见问题

### 1. 端口被占用

如果 8000 端口被占用，可以修改端口：

```bash
export PORT=8001
python app.py
```

或直接指定：

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

### 2. 索引文件不存在

```bash
# 先运行索引生成脚本
python indexer.py
```

### 3. API Key 未设置

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
```

### 4. 依赖未安装

```bash
pip install -r requirements.txt
```

## 📱 命令行版本（可选）

如果你更喜欢命令行界面，可以运行：

```bash
python main.py
```

## 🛑 停止服务

在运行服务的终端中按 `Ctrl + C` 停止服务。

## 📊 监控和调试

### 查看日志

日志文件保存在 `logs/` 目录：
- `logs/app.log` - Web 服务日志
- `logs/agent.log` - Agent 日志

### 查看服务状态

访问 http://localhost:8000/health 查看服务健康状态

访问 http://localhost:8000/stats 查看详细统计信息

## 🎨 Web 界面功能

- ✅ 美观的聊天界面
- ✅ 对话历史自动保存
- ✅ 支持多轮对话
- ✅ 实时响应状态显示
- ✅ 清空对话功能
- ✅ 响应式设计（支持移动端）

## 💡 提示

1. **首次使用**：确保已经运行过 `python indexer.py` 生成索引文件
2. **API Key**：确保设置了正确的 `AI_BUILDER_TOKEN` 环境变量
3. **对话历史**：Web 界面的对话历史保存在浏览器本地，不同浏览器/设备之间不共享
4. **性能**：首次启动会加载索引文件，可能需要几秒钟

## 🚀 一键启动脚本

创建一个 `start_web.sh` 文件：

```bash
#!/bin/bash
cd /Users/xiaodongzheng/vector_indexer
export AI_BUILDER_TOKEN='your_api_key_here'
python app.py
```

然后：

```bash
chmod +x start_web.sh
./start_web.sh
```
