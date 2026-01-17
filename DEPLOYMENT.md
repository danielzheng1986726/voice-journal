# 部署指南

## 📋 部署前准备

### 必需的环境变量

1. **AI_BUILDER_TOKEN**（必需）
   - AI Builder API Token，用于 RAG 对话和扫描功能
   - 获取方式：访问 https://space.ai-builders.com

2. **PORT**（可选，默认 8000）
   - Web 服务端口，云平台会自动设置

### Firebase 配置

项目使用 Firebase Admin SDK，支持两种配置方式：

1. **默认凭证**（推荐用于云端部署）
   - 使用 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量
   - 或使用服务账号 JSON 文件路径

2. **服务账号文件**
   - 将 `firebase-service-account.json` 上传到服务器
   - 设置 `GOOGLE_APPLICATION_CREDENTIALS` 指向该文件

## 🚀 部署步骤

### 方式 1: Docker 部署（推荐）

项目已包含 `Dockerfile`，支持直接构建镜像部署。

#### 1. 构建 Docker 镜像

```bash
docker build -t voice-journal:latest .
```

#### 2. 运行容器

```bash
docker run -d \
  -p 8000:8000 \
  -e AI_BUILDER_TOKEN='your_api_key_here' \
  -e PORT=8000 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/firebase-service-account.json \
  -v $(pwd)/firebase-service-account.json:/app/firebase-service-account.json:ro \
  -v $(pwd)/data:/app/data \
  --name voice-journal \
  voice-journal:latest
```

**数据持久化说明：**
- 建议挂载数据目录以持久化数据：
  - `voice_records.json` - 语音记录
  - `conversations.json` - 会话数据
  - `all_chunks.json` - RAG 数据源
  - `my_history.index` - FAISS 索引
  - `chunks_metadata.json` - 索引元数据

### 方式 2: 云平台部署

#### Koyeb

1. 登录 [Koyeb](https://www.koyeb.com/)
2. 创建新应用，选择 "GitHub" 作为源
3. 选择仓库：`danielzheng1986726/voice-journal`
4. 设置环境变量：
   - `AI_BUILDER_TOKEN`: 你的 API key
   - `GOOGLE_APPLICATION_CREDENTIALS`: `/app/firebase-service-account.json`（如果使用文件）
5. 部署

#### Railway

1. 登录 [Railway](https://railway.app/)
2. 创建新项目，选择 "Deploy from GitHub repo"
3. 选择仓库
4. 在 Variables 中设置环境变量：
   - `AI_BUILDER_TOKEN`
   - `PORT`（Railway 会自动设置）
5. 部署

#### Render

1. 登录 [Render](https://render.com/)
2. 创建新 Web Service
3. 连接 GitHub 仓库
4. 设置环境变量：
   - `AI_BUILDER_TOKEN`
5. 部署

#### AI Builder Space

1. 登录 AI Builder Space
2. 创建新服务
3. 连接 GitHub 仓库
4. 设置环境变量：
   - `AI_BUILDER_TOKEN`
5. 部署

### 方式 3: 直接部署（VPS/服务器）

#### 1. 克隆仓库

```bash
git clone https://github.com/danielzheng1986726/voice-journal.git
cd voice-journal
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 设置环境变量

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
export PORT=8000
```

或创建 `.env` 文件：

```bash
AI_BUILDER_TOKEN=your_api_key_here
PORT=8000
```

#### 4. 配置 Firebase

如果有 `firebase-service-account.json` 文件：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-service-account.json
```

#### 5. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
```

#### 6. 使用 systemd 管理（可选）

创建 `/etc/systemd/system/voice-journal.service`：

```ini
[Unit]
Description=Voice Journal Web Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/voice-journal
Environment="AI_BUILDER_TOKEN=your_api_key_here"
Environment="PORT=8000"
ExecStart=/usr/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl enable voice-journal
sudo systemctl start voice-journal
```

## 🔐 安全配置

### Firebase 认证

1. **获取 Firebase 服务账号密钥**
   - 访问 Firebase Console
   - 项目设置 > 服务账号
   - 生成新的私钥
   - 下载 JSON 文件

2. **配置服务账号**
   - 将 JSON 文件上传到服务器安全位置
   - 设置 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量
   - 或使用环境变量直接配置（更安全）

### 数据隔离

- ✅ 已实现用户数据隔离（按 `user_id` 过滤）
- ✅ Admin 页面已添加 Firebase 认证保护
- ✅ 所有 API 端点都需要认证

## 📊 数据持久化

### 重要文件

以下文件需要持久化存储：

- `voice_records.json` - 用户语音记录
- `conversations.json` - 会话数据
- `all_chunks.json` - RAG 数据源
- `my_history.index` - FAISS 向量索引
- `chunks_metadata.json` - 索引元数据
- `scan_results.json` - 扫描结果

### 数据备份建议

1. **定期备份数据文件**
2. **使用云存储**（如 AWS S3、Google Cloud Storage）
3. **数据库迁移**（未来可考虑迁移到 PostgreSQL/MongoDB）

## 🔧 部署后检查

### 1. 健康检查

```bash
curl http://your-domain/api/records
```

应该返回 401（未认证）或 200（已认证）

### 2. 测试认证

访问主页，应该能看到 Firebase 登录界面

### 3. 测试 RAG 功能

1. 登录后创建会话
2. 发送一条消息
3. 等待索引更新（约 1-2 分钟）
4. 测试查询功能

### 4. 检查日志

```bash
# Docker
docker logs voice-journal

# systemd
journalctl -u voice-journal -f
```

## 🐛 常见问题

### 1. 索引文件缺失

首次部署时，索引文件不存在是正常的。系统会：
- 自动检测并优雅降级
- 新记录会自动触发索引构建
- 可以在 Admin 页面手动触发索引重建

### 2. Firebase 认证失败

检查：
- `GOOGLE_APPLICATION_CREDENTIALS` 是否正确设置
- 服务账号 JSON 文件是否存在
- Firebase 项目配置是否正确

### 3. 端口冲突

确保 `PORT` 环境变量与云平台配置一致

### 4. 数据丢失

确保数据目录已正确挂载（Docker）或配置持久化存储

## 📝 部署清单

- [ ] 环境变量已配置（`AI_BUILDER_TOKEN`）
- [ ] Firebase 服务账号已配置
- [ ] 数据目录已配置持久化
- [ ] 健康检查通过
- [ ] 认证功能正常
- [ ] RAG 功能正常
- [ ] 日志正常输出

## 🔗 相关链接

- GitHub 仓库: https://github.com/danielzheng1986726/voice-journal
- AI Builder Space: https://space.ai-builders.com
- Firebase Console: https://console.firebase.google.com
