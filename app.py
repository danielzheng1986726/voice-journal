#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音记录 Web 应用
功能：显示最近的语音记录，并提供复制功能
"""

import json
import os
import logging
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Journal")

RECORDS_FILE = Path(__file__).parent / "voice_records.json"

# RAG 相关配置（使用相对路径，所有文件都在 vector_indexer 目录下）
VECTOR_INDEXER_DIR = Path(__file__).parent  # 当前目录就是 vector_indexer
INDEX_PATH = VECTOR_INDEXER_DIR / "my_history.index"
METADATA_PATH = VECTOR_INDEXER_DIR / "chunks_metadata.json"
FLAG_FILE = VECTOR_INDEXER_DIR / ".need_reindex"
INDEX_STATUS_FILE = VECTOR_INDEXER_DIR / ".index_status.json"  # 索引重建状态文件

# 设置 RAG 模块的环境变量（在导入前设置）
os.environ.setdefault("INDEX_PATH", str(INDEX_PATH))
os.environ.setdefault("METADATA_PATH", str(METADATA_PATH))

# 确保 AI_BUILDER_TOKEN 已设置（从 .env 文件加载，覆盖现有值）
# 注意：必须在导入 rag_main 之前设置，因为 rag_main 在导入时会读取环境变量
load_dotenv(override=True)
ai_builder_token = os.getenv("AI_BUILDER_TOKEN")
if ai_builder_token:
    # 强制设置环境变量（确保 rag_main.py 能读取到）
    os.environ["AI_BUILDER_TOKEN"] = ai_builder_token
    logger.info(f"✅ AI_BUILDER_TOKEN 已设置（长度: {len(ai_builder_token)}）")
else:
    logger.warning("⚠️  AI_BUILDER_TOKEN 未设置，RAG 功能将不可用")

# 导入 RAG 模块（vector_indexer 使用 main.py 而不是 rag_main.py）
RAG_AVAILABLE = False
chat_with_agent = None
try:
    from main import chat_with_agent
    RAG_AVAILABLE = True
    logger.info("✅ RAG 模块加载成功")
    logger.info(f"   索引路径: {INDEX_PATH}")
    logger.info(f"   元数据路径: {METADATA_PATH}")
except Exception as e:
    logger.warning(f"⚠️  RAG 模块加载失败（索引文件可能不存在）: {e}")
    logger.info("   这是正常的，云端演示版可以在没有索引文件的情况下运行")
    logger.info("   录音功能正常工作，RAG 聊天功能将返回友好提示")
    RAG_AVAILABLE = False
    chat_with_agent = None

# 初始化定时任务调度器
scheduler = BackgroundScheduler()
scheduler.start()
logger.info("✅ 定时任务调度器已启动")

# 对话历史存储（简单的内存存储，实际应用可以使用 Redis 等）
conversation_histories = {}

def sync_to_rag_system(voice_record):
    """
    将voice记录同步到RAG系统的all_chunks.json
    
    参数:
        voice_record: dict，格式为 {id, source, date, time, content}
    
    转换为RAG格式: {id, source, date, content}（去掉time字段）
    """
    logger.info(f"开始同步记录到RAG系统: {voice_record.get('id')}")
    
    # all_chunks.json的路径（放在vector_indexer目录中，与indexer.py一致）
    rag_file = VECTOR_INDEXER_DIR / "all_chunks.json"
    
    try:
        # 读取现有chunks
        if rag_file.exists():
            logger.debug(f"读取文件: {rag_file}")
            with open(rag_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.debug(f"读取到 {len(chunks)} 条现有记录")
        else:
            logger.warning(f"文件不存在，创建新文件: {rag_file}")
            chunks = []
    except Exception as e:
        logger.error(f"读取all_chunks.json失败: {e}", exc_info=True)
        chunks = []
    
    # 转换格式（去掉time字段）
    rag_chunk = {
        "id": voice_record["id"],
        "source": voice_record["source"],
        "date": voice_record["date"],
        "content": voice_record["content"]
    }
    
    # 检查是否已存在（避免重复）
    existing_ids = [c.get('id') for c in chunks]
    if rag_chunk["id"] in existing_ids:
        logger.warning(f"记录已存在，跳过: {rag_chunk['id']}")
        return
    
    # 追加新记录
    chunks.append(rag_chunk)
    logger.debug(f"添加记录后，总数: {len(chunks)}")
    
    # 保存回文件
    try:
        logger.debug(f"保存文件: {rag_file}")
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.debug(f"文件保存成功，大小: {rag_file.stat().st_size} 字节")
        
        # 设置"需要重建索引"的标记文件（放在vector_indexer目录中）
        flag_file = FLAG_FILE
        with open(flag_file, 'w') as f:
            f.write("1")
        logger.debug(f"标记文件已创建: {flag_file}")
        
        # 实时触发增量索引（异步执行，不阻塞）
        # 只在 RAG 可用时触发索引重建
        if RAG_AVAILABLE:
            try:
                scheduler.add_job(
                    incremental_rebuild_index,
                    id=f'incremental_index_{voice_record["id"]}',
                    name=f'增量索引-{voice_record["id"]}',
                    replace_existing=True
                )
                logger.info(f"✅ 已触发实时增量索引: {voice_record['id']}")
            except Exception as e:
                logger.warning(f"⚠️  触发增量索引失败: {e}，将在下次定时任务时重建")
        else:
            logger.info(f"ℹ️  RAG 功能不可用，跳过索引重建: {voice_record['id']}")
            
        logger.info(f"✓ 已同步到RAG系统: {voice_record['id']}")
        
    except Exception as e:
        logger.error(f"保存all_chunks.json失败: {e}", exc_info=True)
        raise

class VoiceRecordRequest(BaseModel):
    """语音记录请求模型"""
    content: str

def generate_id():
    """生成唯一 ID，格式：voice_YYYYMMDD_HHMM"""
    now = datetime.now()
    return f"voice_{now.strftime('%Y%m%d_%H%M')}"

def create_record(content: str):
    """创建一条记录"""
    now = datetime.now()
    return {
        "id": generate_id(),
        "source": "voice",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "content": content
    }

def load_records():
    """加载记录"""
    if RECORDS_FILE.exists():
        try:
            with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_records(records):
    """保存记录到文件"""
    with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

@app.get("/", response_class=HTMLResponse)
async def index():
    """ChatGPT 风格的数字记忆助手界面"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Memory - 你的数字记忆助手</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100vh;
            display: flex;
            background: #343541;
            color: #ececf1;
        }
        
        .sidebar {
            width: 260px;
            background: #202123;
            display: flex;
            flex-direction: column;
            padding: 10px;
        }
        
        .new-chat-btn {
            padding: 12px;
            border: 1px solid #565869;
            border-radius: 5px;
            background: transparent;
            color: #ececf1;
            cursor: pointer;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .new-chat-btn:hover {
            background: #2a2b32;
        }
        
        .chat-history {
            flex: 1;
            overflow-y: auto;
        }
        
        .chat-history-item {
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-history-item:hover {
            background: #2a2b32;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            max-width: 800px;
            margin: 0 auto 20px;
            padding: 20px;
            line-height: 1.6;
        }
        
        .message.user {
            background: #343541;
        }
        
        .message.assistant {
            background: #444654;
            border-radius: 5px;
        }
        
        .message-role {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }
        
        .input-area {
            padding: 20px;
            background: #343541;
        }
        
        .input-container {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            align-items: flex-end;
            gap: 10px;
            background: #40414f;
            border-radius: 10px;
            padding: 10px 15px;
        }
        
        .input-box {
            flex: 1;
            background: transparent;
            border: none;
            color: #ececf1;
            font-size: 16px;
            resize: none;
            max-height: 200px;
            outline: none;
        }
        
        .input-box::placeholder {
            color: #8e8ea0;
        }
        
        .voice-btn, .send-btn {
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .voice-btn {
            background: transparent;
            color: #8e8ea0;
        }
        
        .voice-btn:hover {
            color: #ececf1;
        }
        
        .voice-btn.recording {
            background: #ef4444;
            color: white;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .send-btn {
            background: #19c37d;
            color: white;
        }
        
        .send-btn:hover {
            background: #1a7f5a;
        }
        
        .send-btn:disabled {
            background: #40414f;
            color: #8e8ea0;
            cursor: not-allowed;
        }
        
        .menu-btn {
            display: none;
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: #202123;
            border: none;
            color: #ececf1;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: -260px;
                top: 0;
                bottom: 0;
                z-index: 99;
                transition: left 0.3s;
            }
            
            .sidebar.open {
                left: 0;
            }
            
            .menu-btn {
                display: block;
            }
            
            .message {
                padding: 15px;
            }
            
            .input-container {
                padding: 8px 12px;
            }
            
            .input-box {
                font-size: 16px;
            }
        }
        
        .typing-indicator {
            display: flex;
            gap: 5px;
            padding: 20px;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #8e8ea0;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
    </style>
</head>
<body>
    <button class="menu-btn" onclick="toggleSidebar()">☰</button>
    
    <aside class="sidebar" id="sidebar">
        <button class="new-chat-btn" onclick="newChat()">
            <span>+</span> 新对话
        </button>
        <div class="chat-history" id="chatHistory">
        </div>
    </aside>
    
    <main class="main-content">
        <div class="chat-messages" id="chatMessages">
            <div class="message assistant">
                <div class="message-role">🤖 AI 助手</div>
                <div class="message-content">
                    你好！我是你的数字记忆助手。你可以和我聊天，我会记住我们的对话。
                    以后你可以问我「我之前说过什么」，我会帮你找到。
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-container">
                <textarea 
                    class="input-box" 
                    id="inputBox" 
                    placeholder="输入消息，或点击麦克风语音输入..."
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">🎤</button>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </main>

    <script>
        let isRecording = false;
        let recognition = null;
        let finalTranscript = '';  // 累积最终文本，解决停顿后内容丢失问题
        
        // 初始化语音识别
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window) {
                recognition = new webkitSpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'zh-CN';
                
                recognition.onresult = (event) => {
                    let interimTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const result = event.results[i];
                        const text = result[0].transcript;
                        if (result.isFinal) {
                            finalTranscript += text;
                        } else {
                            interimTranscript += text;
                        }
                    }
                    const inputBox = document.getElementById('inputBox');
                    inputBox.value = finalTranscript + interimTranscript;
                    autoResize(inputBox);
                };
                
                recognition.onend = () => {
                    // 如果还在录音状态（只是用户停顿），自动重启识别
                    if (isRecording) {
                        try {
                            recognition.start();
                        } catch (e) {
                            console.error('重启语音识别失败:', e);
                            stopRecording();
                        }
                    }
                };
                
                recognition.onerror = (event) => {
                    console.error('语音识别错误:', event.error);
                    // no-speech 只是没听到声音，不立即结束整次录音
                    if (event.error !== 'no-speech') {
                        stopRecording();
                    }
                };
            }
        }
        
        function toggleVoice() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        
        function startRecording() {
            if (!recognition) {
                initSpeechRecognition();
            }
            if (recognition) {
                // 开始新一轮录音时清空累积文本
                finalTranscript = '';
                try {
                    recognition.start();
                    isRecording = true;
                    document.getElementById('voiceBtn').classList.add('recording');
                } catch (e) {
                    console.error('启动语音识别失败:', e);
                }
            }
        }
        
        function stopRecording() {
            // 先标记状态，避免 onend 中再次自动重启
            isRecording = false;
            if (recognition) {
                try {
                    recognition.stop();
                } catch (e) {
                    console.error('停止语音识别失败:', e);
                }
            }
            document.getElementById('voiceBtn').classList.remove('recording');
        }
        
        async function sendMessage() {
            const inputBox = document.getElementById('inputBox');
            const message = inputBox.value.trim();
            
            if (!message) return;
            
            if (isRecording) stopRecording();
            
            addMessage('user', message);
            inputBox.value = '';
            autoResize(inputBox);
            
            showTypingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                hideTypingIndicator();
                
                addMessage('assistant', data.response);
                
                await saveToMemory(message, data.response);
                
            } catch (error) {
                hideTypingIndicator();
                addMessage('assistant', '抱歉，发生了错误。请稍后再试。');
                console.error('Error:', error);
            }
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            messageDiv.innerHTML = 
                '<div class="message-role">' + (role === 'user' ? '👤 你' : '🤖 AI 助手') + '</div>' +
                '<div class="message-content">' + formatContent(content) + '</div>';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function formatContent(content) {
            return content
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>');
        }
        
        function showTypingIndicator() {
            const messagesDiv = document.getElementById('chatMessages');
            const indicator = document.createElement('div');
            indicator.id = 'typingIndicator';
            indicator.className = 'message assistant';
            indicator.innerHTML = 
                '<div class="message-role">🤖 AI 助手</div>' +
                '<div class="typing-indicator">' +
                '<span></span><span></span><span></span>' +
                '</div>';
            messagesDiv.appendChild(indicator);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function hideTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) indicator.remove();
        }
        
        async function saveToMemory(userMessage, aiResponse) {
            try {
                await fetch('/api/voice', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        content: '[对话] 我说：' + userMessage
                    })
                });
                
                await fetch('/api/voice', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        content: '[对话] AI 回复：' + aiResponse
                    })
                });
            } catch (error) {
                console.error('保存记忆失败:', error);
            }
        }
        
        function newChat() {
            document.getElementById('chatMessages').innerHTML = 
                '<div class="message assistant">' +
                '<div class="message-role">🤖 AI 助手</div>' +
                '<div class="message-content">' +
                '你好！我是你的数字记忆助手。你可以和我聊天，我会记住我们的对话。' +
                '</div></div>';
        }
        
        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('open');
        }
        
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        // 加载左侧历史记忆（最近 20 条带 [对话] 前缀的记录）
        async function loadChatHistory() {
            try {
                const response = await fetch('/api/records');
                const data = await response.json();
                const records = (data && data.records) ? data.records : [];
                
                const historyDiv = document.getElementById('chatHistory');
                historyDiv.innerHTML = '';
                
                const chatRecords = records
                    .filter(r => r.content && r.content.startsWith('[对话]'))
                    .slice(-20)
                    .reverse();
                
                if (chatRecords.length === 0) {
                    historyDiv.innerHTML = '<div style="color: #8e8ea0; padding: 10px; font-size: 14px;">暂无历史记录</div>';
                    return;
                }
                
                chatRecords.forEach(record => {
                    const item = document.createElement('div');
                    item.className = 'chat-history-item';
                    let summary = record.content
                        .replace('[对话] 我说：', '')
                        .replace('[对话] AI 回复：', '');
                    if (summary.length > 30) {
                        summary = summary.substring(0, 30) + '...';
                    }
                    item.textContent = summary;
                    const date = record.date || '';
                    const time = record.time || '';
                    item.title = (date + ' ' + time).trim();
                    historyDiv.appendChild(item);
                });
            } catch (error) {
                console.error('加载历史记录失败:', error);
            }
        }
        
        // 页面加载完成后初始化语音识别并加载历史
        document.addEventListener('DOMContentLoaded', () => {
            initSpeechRecognition();
            loadChatHistory();
        });
    </script>
</body>
</html>"""
    return html

@app.get("/api/records")
async def get_records():
    """API 端点：获取所有记录"""
    records = load_records()
    records.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
    return {"total": len(records), "records": records}

@app.post("/api/voice")
async def add_voice_record(request: VoiceRecordRequest):
    """
    API 端点：添加语音记录（方案 B）
    快捷指令可以通过 POST 请求调用此端点
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")
    
    # 创建新记录
    record = create_record(request.content.strip())
    
    # 加载现有记录并追加
    records = load_records()
    records.append(record)
    save_records(records)
    
    # 同步到RAG系统
    try:
        sync_to_rag_system(record)
    except Exception as e:
        # 同步失败不影响主功能
        logger.warning(f"警告：同步到RAG系统失败: {e}", exc_info=True)
        # 继续返回成功（因为录音已经保存成功）
    
    return {
        "success": True,
        "message": "记录已保存",
        "record": record
    }

@app.get("/api/voice/add")
async def add_voice_record_get(content: str):
    """
    GET 方式添加语音记录
    快捷指令可以直接构建 URL: /api/voice/add?content=文本内容
    这样不需要配置 JSON 请求体，大大简化快捷指令的操作
    """
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空，请使用 ?content=文本内容")
    
    # 创建新记录
    record = create_record(content.strip())
    
    # 加载现有记录并追加
    records = load_records()
    records.append(record)
    save_records(records)
    
    # 同步到RAG系统
    try:
        sync_to_rag_system(record)
    except Exception as e:
        logger.warning(f"警告：同步到RAG系统失败: {e}", exc_info=True)
    
    return {
        "success": True,
        "message": "记录已保存",
        "record": record
    }

@app.put("/api/voice/{record_id}")
async def update_voice_record(record_id: str, request: VoiceRecordRequest):
    """
    API 端点：更新语音记录
    """
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")
    
    # 加载现有记录
    records = load_records()
    
    # 查找要更新的记录
    record_index = None
    for i, record in enumerate(records):
        if record.get('id') == record_id:
            record_index = i
            break
    
    if record_index is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    
    # 更新记录内容
    records[record_index]['content'] = request.content.strip()
    
    # 保存
    save_records(records)
    
    return {
        "success": True,
        "message": "记录已更新",
        "record": records[record_index]
    }

@app.get("/records", response_class=HTMLResponse)
async def records_page():
    """记录列表页面"""
    records = load_records()
    records.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>记录列表 - Voice Journal</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }}
            .app-container {{
                display: flex;
                min-height: 100vh;
            }}
            .sidebar {{
                width: 250px;
                background: #2c3e50;
                color: white;
                padding: 20px 0;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }}
            .sidebar-header {{
                padding: 0 20px 20px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                margin-bottom: 20px;
            }}
            .sidebar-header h1 {{
                font-size: 20px;
                margin: 0;
                color: white;
            }}
            .sidebar-nav {{
                list-style: none;
                padding: 0;
                margin: 0;
            }}
            .sidebar-nav li {{
                margin: 0;
            }}
            .sidebar-nav a {{
                display: block;
                padding: 15px 20px;
                color: rgba(255,255,255,0.8);
                text-decoration: none;
                transition: all 0.3s;
                border-left: 3px solid transparent;
            }}
            .sidebar-nav a:hover {{
                background: rgba(255,255,255,0.1);
                color: white;
            }}
            .sidebar-nav a.active {{
                background: rgba(102, 126, 234, 0.3);
                border-left-color: #667eea;
                color: white;
            }}
            .main-content {{
                flex: 1;
                padding: 20px;
                overflow-y: auto;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .stats {{
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            .record {{
                border-left: 3px solid #667eea;
                padding: 15px 20px;
                margin-bottom: 20px;
                background: #f8f9fa;
                border-radius: 4px;
                position: relative;
            }}
            .record-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}
            .record-id {{
                font-size: 12px;
                color: #999;
                font-family: monospace;
            }}
            .record-time {{
                font-size: 13px;
                color: #666;
            }}
            .record-content {{
                color: #333;
                line-height: 1.6;
                margin-bottom: 10px;
            }}
            .empty {{
                text-align: center;
                color: #999;
                padding: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="sidebar-header">
                    <h1>🎤 Voice Journal</h1>
                    <p style="font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 5px;">& Digital Twin</p>
                </div>
                <ul class="sidebar-nav">
                    <li><a href="/">🎤 录音</a></li>
                    <li><a href="/records" class="active">📝 记录</a></li>
                    <li><a href="/chat">🤖 智能对话</a></li>
                    <li><a href="/settings">⚙️ 设置</a></li>
                </ul>
            </div>
            <div class="main-content">
                <div class="container">
                    <h1>📝 所有记录</h1>
                    <div class="stats">共 {len(records)} 条记录</div>
                    <div id="records-list">
                        {''.join([f'''
                        <div class="record">
                            <div class="record-header">
                                <span class="record-id">{r.get('id', '')}</span>
                                <span class="record-time">{r.get('date', '')} {r.get('time', '')}</span>
                            </div>
                            <div class="record-content">{r.get('content', '').replace('<', '&lt;').replace('>', '&gt;')}</div>
                        </div>
                        ''' for r in records]) if records else '<div class="empty">暂无记录</div>'}
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def incremental_rebuild_index():
    """增量索引重建：只处理新记录"""
    if not FLAG_FILE.exists():
        return
    
    logger.info("🔍 检测到新数据，开始增量索引...")
    
    try:
        # 使用增量索引脚本
        incremental_indexer_path = Path(__file__).parent / "incremental_indexer.py"
        result = subprocess.run(
            ["python3", str(incremental_indexer_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5分钟超时（增量索引应该很快）
        )
        
        if result.stdout:
            logger.info(f"增量索引输出: {result.stdout[:500]}")
        
        if FLAG_FILE.exists():
            FLAG_FILE.unlink()
        
        logger.info("✅ 增量索引完成！")
        
    except subprocess.TimeoutExpired:
        logger.error("✗ 增量索引超时")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ 增量索引失败: {e}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr[:500]}")
        # 增量索引失败时，可以考虑回退到全量重建
        logger.warning("⚠️  增量索引失败，将在下次定时任务时尝试全量重建")
    except Exception as e:
        logger.exception(f"✗ 增量索引异常: {e}")

def update_index_status(status: str, progress: int = 0, message: str = ""):
    """更新索引重建状态"""
    status_data = {
        "status": status,  # "idle", "running", "completed", "failed"
        "progress": progress,  # 0-100
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(INDEX_STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"无法更新索引状态: {e}")

def get_index_status():
    """获取索引重建状态"""
    if not INDEX_STATUS_FILE.exists():
        return {
            "status": "idle",
            "progress": 0,
            "message": "未开始",
            "timestamp": None
        }
    try:
        with open(INDEX_STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"无法读取索引状态: {e}")
        return {
            "status": "idle",
            "progress": 0,
            "message": f"状态读取错误: {e}",
            "timestamp": None
        }

def check_and_rebuild_index():
    """检查并重建索引（定时任务，全量重建作为兜底）"""
    if not FLAG_FILE.exists():
        return
    
    logger.info("🔍 检测到新数据，开始全量重建索引（兜底）...")
    update_index_status("running", 0, "开始全量重建索引...")
    
    try:
        indexer_path = VECTOR_INDEXER_DIR / "indexer.py"
        
        # 使用 Popen 实时读取输出
        import re
        import threading
        
        process = subprocess.Popen(
            ["python3", "-u", str(indexer_path)],  # -u 参数禁用 Python 缓冲
            cwd=str(VECTOR_INDEXER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # 实时读取输出并更新进度
        output_lines = []
        last_progress = 0
        
        def read_output():
            """在单独线程中读取输出"""
            nonlocal last_progress
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                output_lines.append(line)
                line_stripped = line.strip()
                logger.debug(f"索引输出: {line_stripped}")
                
                # 解析进度信息
                if "处理进度:" in line_stripped:
                    # 提取百分比，例如: "处理进度: 50/100 (50%)"
                    match = re.search(r'(\d+)%', line_stripped)
                    if match:
                        progress = min(int(match.group(1)), 95)
                        if progress > last_progress:
                            last_progress = progress
                            update_index_status("running", progress, f"处理中: {line_stripped[:80]}")
                
                elif "批次" in line_stripped and "/" in line_stripped:
                    # 提取批次进度，例如: "批次 5/10"
                    match = re.search(r'批次\s+(\d+)/(\d+)', line_stripped)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        progress = min(int((current / total) * 90), 90)  # 批次处理占90%
                        if progress > last_progress:
                            last_progress = progress
                            update_index_status("running", progress, f"批次 {current}/{total}: {line_stripped[:60]}")
                
                elif "开始生成 Embeddings" in line_stripped or "开始生成向量" in line_stripped:
                    update_index_status("running", 20, "开始生成向量嵌入...")
                elif "Embeddings 生成完成" in line_stripped or "向量嵌入生成完成" in line_stripped:
                    update_index_status("running", 80, "向量嵌入生成完成，正在构建索引...")
                elif "构建 FAISS 索引" in line_stripped or "保存索引" in line_stripped:
                    update_index_status("running", 90, "正在保存索引文件...")
        
        # 启动输出读取线程
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
        # 等待进程完成
        process.wait()
        output_thread.join(timeout=1)  # 等待输出线程完成
        
        # 读取错误输出
        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()
        
        # 检查返回码
        if process.returncode != 0:
            error_msg = f"索引重建失败（返回码: {process.returncode})"
            if stderr_output:
                error_msg += f"\n错误: {stderr_output[:200]}"
            update_index_status("failed", 0, error_msg)
            logger.error(f"✗ 索引重建失败: {error_msg}")
            if stderr_output:
                logger.error(f"错误输出: {stderr_output[:500]}")
            return
        
        # 合并输出
        output = '\n'.join(output_lines)
        if output:
            logger.info(f"索引重建输出: {output[:500]}")
        
        if FLAG_FILE.exists():
            FLAG_FILE.unlink()
        
        # 更新状态：完成
        update_index_status("completed", 100, "索引重建完成！")
        logger.info("✅ 全量索引重建完成！")
        
    except subprocess.TimeoutExpired:
        update_index_status("failed", 0, "索引重建超时（超过10分钟）")
        logger.error("✗ 索引重建超时")
    except Exception as e:
        update_index_status("failed", 0, f"索引重建异常: {str(e)[:200]}")
        logger.exception(f"✗ 索引重建异常: {e}")

# 添加定时任务：每30分钟检查一次（作为兜底，主要依靠实时同步）
# 只在 RAG 可用时添加定时任务
if RAG_AVAILABLE:
    scheduler.add_job(
        check_and_rebuild_index,
        trigger=IntervalTrigger(minutes=30),
        id='rebuild_index_job',
        name='定时重建索引（兜底）',
        replace_existing=True
    )
    logger.info("✅ 定时索引重建任务已添加（每30分钟检查一次，作为兜底）")
    logger.info("✅ 实时同步已启用：录音保存后立即触发索引重建")
else:
    logger.info("ℹ️  RAG 功能不可用，跳过定时索引重建任务")

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    success: bool
    error: str = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    """RAG 聊天 API 端点"""
    if not RAG_AVAILABLE or chat_with_agent is None:
        return ChatResponse(
            response="RAG 功能暂不可用（索引文件未加载）。这是云端演示版，录音功能正常工作。如需完整 RAG 功能，请使用本地版本。",
            success=False,
            error="RAG模块未加载（索引文件缺失）"
        )
    
    try:
        # 获取或创建对话历史
        session_id = request.session_id
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        
        conversation_history = conversation_histories[session_id]
        
        # 调用 RAG 对话函数
        response = chat_with_agent(request.message, conversation_history)
        
        # 更新对话历史
        conversation_history.append({
            "role": "user",
            "content": request.message
        })
        conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # 限制历史长度（保留最近20条消息）
        if len(conversation_history) > 20:
            conversation_histories[session_id] = conversation_history[-20:]
        
        return ChatResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        logger.exception(f"聊天 API 错误: {e}")
        return ChatResponse(
            response=f"处理请求时出错: {str(e)}",
            success=False,
            error=str(e)
        )

@app.get("/api/index-status")
async def get_index_status_api():
    """获取索引重建状态"""
    status = get_index_status()
    return status

@app.post("/api/rebuild-index")
async def rebuild_index_api(background_tasks: BackgroundTasks):
    """手动触发索引重建"""
    if not RAG_AVAILABLE:
        return {
            "success": False,
            "error": "RAG 功能不可用（索引文件缺失）。这是云端演示版，无法重建索引。如需完整功能，请使用本地版本。"
        }
    
    try:
        # 检查是否正在运行
        current_status = get_index_status()
        if current_status.get("status") == "running":
            return {
                "success": False,
                "error": "索引重建正在进行中，请稍候..."
            }
        
        # 创建标记文件
        FLAG_FILE.touch()
        logger.info("✅ 已创建索引重建标记文件")
        
        # 重置状态
        update_index_status("running", 0, "正在启动索引重建...")
        
        # 在后台执行重建
        background_tasks.add_task(check_and_rebuild_index)
        
        return {
            "success": True,
            "message": "索引重建任务已启动，将在后台执行"
        }
    except Exception as e:
        logger.exception(f"手动重建索引失败: {e}")
        update_index_status("failed", 0, f"启动失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """智能对话页面"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>智能对话 - Voice Journal</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            .app-container { display: flex; min-height: 100vh; }
            .sidebar { width: 250px; background: #2c3e50; color: white; padding: 20px 0; }
            .sidebar-header { padding: 0 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px; }
            .sidebar-header h1 { font-size: 20px; margin: 0; color: white; }
            .sidebar-nav { list-style: none; padding: 0; margin: 0; }
            .sidebar-nav li { margin: 0; }
            .sidebar-nav a { display: block; padding: 15px 20px; color: rgba(255,255,255,0.8); text-decoration: none; transition: all 0.3s; border-left: 3px solid transparent; }
            .sidebar-nav a:hover { background: rgba(255,255,255,0.1); color: white; }
            .sidebar-nav a.active { background: rgba(102, 126, 234, 0.3); border-left-color: #667eea; color: white; }
            .main-content { flex: 1; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); height: calc(100vh - 40px); display: flex; flex-direction: column; }
            .chat-header { padding: 20px; border-bottom: 1px solid #eee; }
            .chat-area { flex: 1; overflow-y: auto; padding: 20px; background: #f5f5f5; }
            .message { margin-bottom: 15px; }
            .message.user { text-align: right; }
            .message-content { display: inline-block; padding: 12px 18px; border-radius: 18px; max-width: 70%; }
            .message.user .message-content { background: #667eea; color: white; }
            .message.assistant .message-content { background: white; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .input-area { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
            .input-area input { flex: 1; padding: 12px; border: 2px solid #ddd; border-radius: 25px; font-size: 14px; }
            .input-area input:focus { outline: none; border-color: #667eea; }
            .input-area button { padding: 12px 24px; background: #667eea; color: white; border: none; border-radius: 25px; cursor: pointer; transition: background 0.2s; }
            .input-area button:hover { background: #5568d3; }
            .input-area button:disabled { background: #ccc; cursor: not-allowed; }
            .message.loading { opacity: 0.7; }
            .message.error .message-content { background: #fee; color: #c33; border: 1px solid #fcc; }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="sidebar-header">
                    <h1>🎤 Voice Journal</h1>
                    <p style="font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 5px;">& Digital Twin</p>
                </div>
                <ul class="sidebar-nav">
                    <li><a href="/">🎤 录音</a></li>
                    <li><a href="/records">📝 记录</a></li>
                    <li><a href="/chat" class="active">🤖 智能对话</a></li>
                    <li><a href="/settings">⚙️ 设置</a></li>
                </ul>
            </div>
            <div class="main-content">
                <div class="container">
                    <div class="chat-header">
                        <h1>🤖 Digital Twin 守护者</h1>
                        <p style="color: #666; font-size: 14px; margin-top: 5px;">你的个人记忆库智能助手</p>
                    </div>
                    <div class="chat-area" id="chatArea">
                        <div class="message assistant">
                            <div class="message-content">
                                你好！我是你的 Digital Twin 守护者。我可以帮你回忆过去、查找日记、分析模式。<br><br>
                                试试问我："2024年6月2日我在做什么让我感到开心？"
                            </div>
                        </div>
                    </div>
                    <div class="input-area">
                        <input type="text" id="messageInput" placeholder="输入你的问题..." autocomplete="off">
                        <button onclick="sendMessage()">发送</button>
                    </div>
                </div>
            </div>
        </div>
        <script>
            const sessionId = 'chat_' + Date.now();
            let isLoading = false;
            
            document.getElementById('messageInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !isLoading) sendMessage();
            });
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message || isLoading) return;
                
                const chatArea = document.getElementById('chatArea');
                const sendButton = document.querySelector('.input-area button');
                
                // 显示用户消息
                chatArea.innerHTML += `<div class="message user"><div class="message-content">${escapeHtml(message)}</div></div>`;
                input.value = '';
                chatArea.scrollTop = chatArea.scrollHeight;
                
                // 显示加载消息
                const loadingMsg = document.createElement('div');
                loadingMsg.className = 'message assistant loading';
                loadingMsg.innerHTML = '<div class="message-content">正在思考...</div>';
                chatArea.appendChild(loadingMsg);
                chatArea.scrollTop = chatArea.scrollHeight;
                
                // 禁用输入
                isLoading = true;
                input.disabled = true;
                sendButton.disabled = true;
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // 更新消息
                    loadingMsg.classList.remove('loading');
                    if (data.success) {
                        loadingMsg.innerHTML = `<div class="message-content">${escapeHtml(data.response).replace(/\\n/g, '<br>')}</div>`;
                    } else {
                        loadingMsg.className = 'message error';
                        loadingMsg.innerHTML = `<div class="message-content">错误: ${escapeHtml(data.error || data.response)}</div>`;
                    }
                    
                } catch (error) {
                    loadingMsg.classList.remove('loading');
                    loadingMsg.className = 'message error';
                    loadingMsg.innerHTML = `<div class="message-content">网络错误: ${escapeHtml(error.message)}</div>`;
                } finally {
                    // 恢复输入
                    isLoading = false;
                    input.disabled = false;
                    sendButton.disabled = false;
                    input.focus();
                    chatArea.scrollTop = chatArea.scrollHeight;
                }
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        </script>
    </body>
    </html>
    """
    return html

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    """设置页面"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>设置 - Voice Journal</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            .app-container { display: flex; min-height: 100vh; }
            .sidebar { width: 250px; background: #2c3e50; color: white; padding: 20px 0; }
            .sidebar-header { padding: 0 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px; }
            .sidebar-header h1 { font-size: 20px; margin: 0; color: white; }
            .sidebar-nav { list-style: none; padding: 0; margin: 0; }
            .sidebar-nav li { margin: 0; }
            .sidebar-nav a { display: block; padding: 15px 20px; color: rgba(255,255,255,0.8); text-decoration: none; transition: all 0.3s; border-left: 3px solid transparent; }
            .sidebar-nav a:hover { background: rgba(255,255,255,0.1); color: white; }
            .sidebar-nav a.active { background: rgba(102, 126, 234, 0.3); border-left-color: #667eea; color: white; }
            .main-content { flex: 1; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); padding: 30px; }
            .setting-item { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
            .setting-item h3 { margin-bottom: 10px; color: #333; }
            .setting-item p { color: #666; font-size: 14px; }
            .progress-container { margin-top: 15px; }
            .progress-bar { width: 100%; height: 24px; background: #f0f0f0; border-radius: 12px; overflow: hidden; margin-bottom: 8px; }
            .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; }
            .progress-message { font-size: 13px; color: #666; margin-top: 5px; }
            .status-idle { color: #999; }
            .status-running { color: #667eea; }
            .status-completed { color: #28a745; }
            .status-failed { color: #dc3545; }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="sidebar-header">
                    <h1>🎤 Voice Journal</h1>
                    <p style="font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 5px;">& Digital Twin</p>
                </div>
                <ul class="sidebar-nav">
                    <li><a href="/">🎤 录音</a></li>
                    <li><a href="/records">📝 记录</a></li>
                    <li><a href="/chat">🤖 智能对话</a></li>
                    <li><a href="/settings" class="active">⚙️ 设置</a></li>
                </ul>
            </div>
            <div class="main-content">
                <div class="container">
                    <h1>⚙️ 设置</h1>
                    <div class="setting-item">
                        <h3>索引重建</h3>
                        <p>定时索引重建：每30分钟自动检查并重建索引（作为兜底）</p>
                        <button id="rebuildBtn" onclick="rebuildIndex()" style="margin-top: 10px; padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer;">手动重建索引</button>
                        <div class="progress-container" id="progressContainer" style="display: none;">
                            <div class="progress-bar">
                                <div class="progress-fill" id="progressFill" style="width: 0%;">0%</div>
                            </div>
                            <div class="progress-message" id="progressMessage"></div>
                        </div>
                        <p id="rebuildStatus" style="margin-top: 10px; font-size: 12px;"></p>
                    </div>
                    <div class="setting-item">
                        <h3>数据同步</h3>
                        <p>录音记录会自动同步到 RAG 系统</p>
                    </div>
                </div>
            </div>
        </div>
        <script>
            let statusPollInterval = null;
            
            // 页面加载时检查状态
            window.addEventListener('load', () => {
                checkIndexStatus();
            });
            
            async function checkIndexStatus() {
                try {
                    const response = await fetch('/api/index-status');
                    const data = await response.json();
                    
                    updateStatusDisplay(data);
                    
                    // 如果正在运行，继续轮询
                    if (data.status === 'running') {
                        if (!statusPollInterval) {
                            statusPollInterval = setInterval(checkIndexStatus, 2000); // 每2秒检查一次
                        }
                    } else {
                        // 停止轮询
                        if (statusPollInterval) {
                            clearInterval(statusPollInterval);
                            statusPollInterval = null;
                        }
                    }
                } catch (error) {
                    console.error('获取状态失败:', error);
                }
            }
            
            function updateStatusDisplay(data) {
                const statusEl = document.getElementById('rebuildStatus');
                const progressContainer = document.getElementById('progressContainer');
                const progressFill = document.getElementById('progressFill');
                const progressMessage = document.getElementById('progressMessage');
                const btn = document.getElementById('rebuildBtn');
                
                // 更新状态文本
                statusEl.textContent = data.message || '未开始';
                
                // 根据状态更新样式和显示
                if (data.status === 'idle') {
                    statusEl.className = 'status-idle';
                    progressContainer.style.display = 'none';
                    btn.disabled = false;
                } else if (data.status === 'running') {
                    statusEl.className = 'status-running';
                    progressContainer.style.display = 'block';
                    progressFill.style.width = data.progress + '%';
                    progressFill.textContent = data.progress + '%';
                    progressMessage.textContent = data.message || '正在处理...';
                    btn.disabled = true;
                } else if (data.status === 'completed') {
                    statusEl.className = 'status-completed';
                    statusEl.textContent = '✓ ' + (data.message || '索引重建完成！');
                    progressContainer.style.display = 'block';
                    progressFill.style.width = '100%';
                    progressFill.textContent = '100%';
                    progressMessage.textContent = '✓ ' + (data.message || '索引重建完成！');
                    btn.disabled = false;
                    // 3秒后隐藏进度条
                    setTimeout(() => {
                        progressContainer.style.display = 'none';
                    }, 3000);
                } else if (data.status === 'failed') {
                    statusEl.className = 'status-failed';
                    statusEl.textContent = '✗ ' + (data.message || '索引重建失败');
                    progressContainer.style.display = 'block';
                    progressFill.style.width = '100%';
                    progressFill.style.background = '#dc3545';
                    progressFill.textContent = '失败';
                    progressMessage.textContent = '✗ ' + (data.message || '索引重建失败');
                    btn.disabled = false;
                }
            }
            
            async function rebuildIndex() {
                const btn = document.getElementById('rebuildBtn');
                const status = document.getElementById('rebuildStatus');
                
                btn.disabled = true;
                status.textContent = '正在启动重建任务...';
                status.className = 'status-running';
                
                try {
                    const response = await fetch('/api/rebuild-index', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        // 开始轮询状态
                        checkIndexStatus();
                        if (!statusPollInterval) {
                            statusPollInterval = setInterval(checkIndexStatus, 2000);
                        }
                    } else {
                        status.textContent = '✗ 错误: ' + (data.error || '未知错误');
                        status.className = 'status-failed';
                        btn.disabled = false;
                    }
                } catch (error) {
                    status.textContent = '✗ 网络错误: ' + error.message;
                    status.className = 'status-failed';
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return html

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("正在关闭应用...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("定时任务调度器已关闭")

if __name__ == "__main__":
    import uvicorn
    import os
    # Builder Space 要求使用 PORT 环境变量
    port = int(os.environ.get("PORT", 8000))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        if scheduler.running:
            scheduler.shutdown()
