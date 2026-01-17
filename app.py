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
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Journal")

RECORDS_FILE = Path(__file__).parent / "voice_records.json"
CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"
SCAN_RESULTS_FILE = Path(__file__).parent / "scan_results.json"

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

# ================= Firebase 初始化 =================
# 使用默认凭证（适用于本地开发和云端部署）
FIREBASE_AVAILABLE = False
try:
    # 检查是否已经初始化
    firebase_admin.get_app()
    logger.info("✅ Firebase Admin SDK 已初始化")
    FIREBASE_AVAILABLE = True
except ValueError:
    # 未初始化，尝试初始化
    # 本地开发：需要服务账号 JSON 文件
    # 如果没有，则跳过初始化（API 验证会被禁用）
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "firebase-service-account.json")
    # 也检查旧的凭证文件名（向后兼容）
    old_credentials_path = Path(__file__).parent / "firebase-credentials.json"
    
    credentials_path = None
    if os.path.exists(service_account_path):
        credentials_path = service_account_path
    elif old_credentials_path.exists():
        credentials_path = str(old_credentials_path)
    
    if credentials_path:
        try:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase Admin SDK 初始化成功")
            FIREBASE_AVAILABLE = True
        except Exception as e:
            logger.warning(f"⚠️  Firebase 初始化失败: {e}")
            FIREBASE_AVAILABLE = False
    else:
        logger.warning("⚠️  未找到 Firebase 服务账号文件，API 验证已禁用（开发模式）")
        FIREBASE_AVAILABLE = False

# ================= Token 验证 =================
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """
    验证 Firebase ID Token，返回用户信息。
    如果 Firebase 未初始化或 token 无效，返回开发用户（开发模式）或抛出 401。
    """
    # 检查 Firebase 是否初始化
    try:
        firebase_admin.get_app()
        firebase_initialized = True
    except ValueError:
        firebase_initialized = False
    
    if not firebase_initialized:
        # Firebase 未初始化，开发模式跳过验证
        logger.debug("⚠️  Firebase 未初始化，使用开发模式用户")
        return {"uid": "dev-user", "email": "dev@localhost"}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证信息",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email", "")
        }
    except Exception as e:
        logger.warning(f"⚠️  Token 验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 验证失败: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# 初始化定时任务调度器
scheduler = BackgroundScheduler()
scheduler.start()
logger.info("✅ 定时任务调度器已启动")

def auto_scan():
    """
    自动扫描函数（定时任务调用）
    执行扫描并将结果保存到 scan_results.json
    """
    try:
        logger.info("🔄 [自动扫描] 开始定时扫描...")
        result = _perform_scan()
        
        # 保存结果到文件
        scan_result = {
            "scan_time": datetime.now().isoformat(),
            "result": result,
            "trigger": "auto"  # 标记为自动触发
        }
        
        with open(SCAN_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(scan_result, f, ensure_ascii=False, indent=2)
        
        if "error" in result:
            logger.warning(f"⚠️  [自动扫描] 扫描完成，但有错误: {result.get('error', '未知错误')}")
        else:
            patterns_count = len(result.get('deep_dive_report', {}).get('patterns', []))
            logger.info(f"✅ [自动扫描] 定时扫描完成，识别到 {patterns_count} 个模式")
            
    except Exception as e:
        logger.exception(f"❌ [自动扫描] 定时扫描异常: {e}")
        # 即使出错也保存错误信息
        try:
            scan_result = {
                "scan_time": datetime.now().isoformat(),
                "result": {
                    "error": f"扫描过程出现异常: {str(e)}"
                }
            }
            with open(SCAN_RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(scan_result, f, ensure_ascii=False, indent=2)
        except Exception as save_error:
            logger.error(f"❌ [自动扫描] 保存错误信息失败: {save_error}")

# 添加定时扫描任务（每小时一次）
scheduler.add_job(
    auto_scan,
    trigger=IntervalTrigger(hours=1),
    id='auto_scan_job',
    name='每小时自动扫描',
    replace_existing=True
)
logger.info("✅ 已启动定时扫描任务（每小时一次）")

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
    conversation_id: str | None = None

def generate_id():
    """生成唯一 ID，格式：voice_YYYYMMDD_HHMM"""
    now = datetime.now()
    return f"voice_{now.strftime('%Y%m%d_%H%M')}"

def create_record(content: str, conversation_id: str | None = None, user_id: str | None = None):
    """创建一条记录"""
    now = datetime.now()
    record = {
        "id": generate_id(),
        "source": "voice",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "content": content,
    }
    # 仅当提供会话 ID 时才写入字段，兼容旧数据
    if conversation_id:
        record["conversation_id"] = conversation_id
    # 写入用户 ID（数据隔离）
    if user_id:
        record["user_id"] = user_id
    return record

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


def load_conversations() -> list[dict]:
    """加载会话列表"""
    if CONVERSATIONS_FILE.exists():
        try:
            with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_conversations(conversations: list[dict]) -> None:
    """保存会话列表"""
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

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
            font-family: 'Söhne', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Ubuntu', 'Cantarell', 'Noto Sans', sans-serif;
            height: 100vh;
            display: flex;
            background: #212121;
            color: #ececf1;
        }
        
        .sidebar {
            width: 260px;
            background: #171717;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
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
            background: #212121;
        }
        
        .message {
            max-width: 768px;
            margin: 0 auto 20px;
            padding: 20px;
            line-height: 1.6;
            display: flex;
            gap: 16px;
        }
        
        .message.user {
            background: transparent;
        }
        
        .message.assistant {
            background: transparent;
        }
        
        .message-role {
            display: none;
        }
        
        .avatar {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        
        .ai-avatar {
            background: #10a37f;
        }
        
        .user-avatar {
            background: #5c5c5c;
        }
        
        .message-content {
            flex: 1;
        }
        
        .input-area {
            padding: 20px;
            background: #212121;
        }
        
        .input-container {
            max-width: 768px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 8px;
            background: #2f2f2f;
            border: 1px solid #424242;
            border-radius: 24px;
            padding: 12px 16px 12px 52px;
            position: relative;
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
            padding: 0;
        }
        
        .input-box::placeholder {
            color: #8e8ea0;
        }
        
        .voice-btn, .send-btn {
            width: 32px;
            height: 32px;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            background: transparent;
            color: #9b9b9b;
            padding: 0;
        }
        
        .voice-btn {
            position: absolute;
            left: 12px;
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
            background: #10a37f;
            color: white;
        }
        
        .send-btn:hover {
            background: #0d8c6f;
        }
        
        .send-btn:disabled {
            background: transparent;
            color: #9b9b9b;
            cursor: not-allowed;
        }
        
        .settings-btn {
            position: fixed;
            top: 15px;
            right: 15px;
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: #8e8ea0;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            z-index: 100;
        }
        
        .settings-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            color: #ececf1;
        }
        
        @media (max-width: 768px) {
            .settings-btn {
                top: 10px;
                right: 50px;
            }
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

        .chat-history-item.active {
            background: #2a2b32;
            border-left: 3px solid #10a37f;
        }
        
        .user-info {
            padding: 10px;
            border-top: 1px solid #424242;
            margin-top: 10px;
            font-size: 12px;
            color: #9b9b9b;
        }
        
        .user-email {
            margin-bottom: 8px;
        }
        
        .logout-btn {
            width: 100%;
            padding: 8px;
            background: transparent;
            border: 1px solid #424242;
            border-radius: 6px;
            color: #ececf1;
            cursor: pointer;
            font-size: 12px;
        }
        
        .logout-btn:hover {
            background: #2a2b32;
        }
        
        /* 登录界面样式 */
        .login-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .login-modal.show {
            display: flex;
        }
        
        .login-card {
            background: #171717;
            border-radius: 12px;
            padding: 40px;
            width: 400px;
            max-width: 90vw;
        }
        
        .login-title {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 24px;
            text-align: center;
        }
        
        .login-form input {
            width: 100%;
            padding: 12px;
            margin-bottom: 16px;
            background: #2f2f2f;
            border: 1px solid #424242;
            border-radius: 8px;
            color: #ececf1;
            font-size: 14px;
            outline: none;
        }
        
        .login-form input:focus {
            border-color: #10a37f;
        }
        
        .login-btn {
            width: 100%;
            padding: 12px;
            background: #10a37f;
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            margin-bottom: 12px;
        }
        
        .login-btn:hover {
            background: #0d8c6f;
        }
        
        .login-switch {
            text-align: center;
            color: #9b9b9b;
            font-size: 14px;
            cursor: pointer;
        }
        
        .login-switch:hover {
            color: #ececf1;
        }
        
        .login-error {
            color: #ef4444;
            font-size: 12px;
            margin-bottom: 12px;
            display: none;
        }
        
        .login-error.show {
            display: block;
        }

        .conv-title {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
        }

        .conv-delete {
            display: none;
            color: #8e8ea0;
            cursor: pointer;
            padding: 0 5px;
            font-size: 18px;
        }

        .conv-delete:hover {
            color: #ef4444;
        }

        .chat-history-item:hover .conv-delete {
            display: block;
        }
    </style>
</head>
<body>
    <!-- 登录界面 -->
    <div class="login-modal" id="loginModal">
        <div class="login-card">
            <h1 class="login-title">登录 Digital Memory</h1>
            <div class="login-error" id="loginError"></div>
            <form class="login-form" id="loginForm" onsubmit="handleLogin(event)">
                <input type="email" id="loginEmail" placeholder="邮箱" required>
                <input type="password" id="loginPassword" placeholder="密码" required>
                <input type="password" id="loginConfirmPassword" placeholder="确认密码" style="display: none;">
                <button type="submit" class="login-btn" id="loginBtn">登录</button>
                <div class="login-switch" id="loginSwitch" onclick="toggleLoginMode()">没有账号？注册</div>
            </form>
        </div>
    </div>
    
    <!-- 主应用（登录后显示） -->
    <div id="mainApp" style="display: none; width: 100%; height: 100vh;">
        <button class="menu-btn" onclick="toggleSidebar()">☰</button>
        <button class="settings-btn" onclick="window.location.href='/admin'" title="管理设置">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
            </svg>
        </button>
        
        <aside class="sidebar" id="sidebar">
            <button class="new-chat-btn" onclick="newChat()">
                <span>+</span> 新对话
            </button>
            <div class="chat-history" id="chatHistory">
            </div>
            <div class="user-info" id="userInfo" style="display: none;">
                <div class="user-email" id="userEmail"></div>
                <button class="logout-btn" onclick="handleLogout()">登出</button>
            </div>
        </aside>
        
        <main class="main-content">
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="avatar ai-avatar"></div>
                    <div class="message-content">
                        你好！我是你的数字记忆助手。你可以用文字或语音和我对话，我会记住我们的交流。<br><br>
                        试试问我：「最近两天我说了什么」或「帮我回忆上个月的事」
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-container">
                    <button class="voice-btn" id="voiceBtn" onclick="toggleVoice()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                            <line x1="12" y1="19" x2="12" y2="22"/>
                        </svg>
                    </button>
                    <textarea 
                        class="input-box" 
                        id="inputBox" 
                        placeholder="输入消息..."
                        rows="1"
                        onkeydown="handleKeyDown(event)"
                        oninput="autoResize(this)"
                    ></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M12 4L12 20M12 4L6 10M12 4L18 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
            </div>
        </main>
    </div>

    <script type="module">
// Firebase 配置
const firebaseConfig = {
  apiKey: "AIzaSyDuWwWz2vWm6FV50w5ozL0DFoxfJfcEy0g",
  authDomain: "voice-journal-auth-ba3b0.firebaseapp.com",
  projectId: "voice-journal-auth-ba3b0"
};

// 导入 Firebase JS SDK
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// 初始化 Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// 全局变量
let authToken = null;
let isLoginMode = true;  // true: 登录模式, false: 注册模式

// 状态管理
let isRecording = false;
let recognition = null;
let finalTranscript = '';
let currentConversationId = null;

// Firebase 认证状态监听
onAuthStateChanged(auth, (user) => {
    if (user) {
        // 已登录
        document.getElementById('loginModal').classList.remove('show');
        document.getElementById('mainApp').style.display = 'flex';
        // 获取并存储 ID Token
        user.getIdToken().then(token => {
            authToken = token;
            // 显示用户信息
            const userEmail = document.getElementById('userEmail');
            const userInfo = document.getElementById('userInfo');
            if (userEmail && userInfo) {
                userEmail.textContent = user.email || 'unknown@local';
                userInfo.style.display = 'block';
            }
            // 每小时刷新 Token
            setInterval(() => {
                user.getIdToken(true).then(token => {
                    authToken = token;
                });
            }, 55 * 60 * 1000); // 55分钟刷新一次（Firebase token 有效期 1 小时）
        });
    } else {
        // 未登录
        document.getElementById('loginModal').classList.add('show');
        document.getElementById('mainApp').style.display = 'none';
        authToken = null;
    }
});

// 登录/注册处理
async function handleLogin(event) {
    event.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const confirmPassword = document.getElementById('loginConfirmPassword');
    const errorDiv = document.getElementById('loginError');
    const loginBtn = document.getElementById('loginBtn');
    
    errorDiv.classList.remove('show');
    
    // 注册模式需要确认密码
    if (!isLoginMode && confirmPassword.style.display !== 'none') {
        if (password !== confirmPassword.value) {
            errorDiv.textContent = '密码不一致';
            errorDiv.classList.add('show');
            return;
        }
    }
    
    try {
        loginBtn.disabled = true;
        loginBtn.textContent = isLoginMode ? '登录中...' : '注册中...';
        
        if (isLoginMode) {
            await signInWithEmailAndPassword(auth, email, password);
        } else {
            await createUserWithEmailAndPassword(auth, email, password);
        }
    } catch (error) {
        errorDiv.textContent = isLoginMode ? '登录失败: ' + error.message : '注册失败: ' + error.message;
        errorDiv.classList.add('show');
        loginBtn.disabled = false;
        loginBtn.textContent = isLoginMode ? '登录' : '注册';
    }
}

// 切换登录/注册模式
function toggleLoginMode() {
    isLoginMode = !isLoginMode;
    const loginBtn = document.getElementById('loginBtn');
    const loginSwitch = document.getElementById('loginSwitch');
    const confirmPassword = document.getElementById('loginConfirmPassword');
    const errorDiv = document.getElementById('loginError');
    
    errorDiv.classList.remove('show');
    
    if (isLoginMode) {
        loginBtn.textContent = '登录';
        loginSwitch.textContent = '没有账号？注册';
        confirmPassword.style.display = 'none';
    } else {
        loginBtn.textContent = '注册';
        loginSwitch.textContent = '已有账号？登录';
        confirmPassword.style.display = 'block';
    }
}

// 登出
async function handleLogout() {
    try {
        await signOut(auth);
    } catch (error) {
        console.error('登出失败:', error);
    }
}

// 导出登录相关函数到全局（供 HTML onclick 调用）
window.handleLogin = handleLogin;
window.toggleLoginMode = toggleLoginMode;
window.handleLogout = handleLogout;

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
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            document.getElementById('inputBox').value = finalTranscript + interimTranscript;
            autoResize(document.getElementById('inputBox'));
        };
        
        recognition.onend = () => {
            if (isRecording) {
                recognition.start();
            }
        };
        
        recognition.onerror = (event) => {
            console.error('语音识别错误:', event.error);
            if (event.error !== 'no-speech') {
                stopRecording();
            }
        };
    } else {
        console.warn('当前浏览器不支持 Web Speech API 语音识别');
        alert('当前浏览器不支持语音识别功能，请在桌面版 Chrome 浏览器中使用语音输入。');
    }
}

window.toggleVoice = function toggleVoice() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
};

function startRecording() {
    if (!recognition) {
        initSpeechRecognition();
    }
    if (recognition) {
        finalTranscript = '';
        recognition.start();
        isRecording = true;
        document.getElementById('voiceBtn').classList.add('recording');
    }
}

function stopRecording() {
    isRecording = false;
    if (recognition) {
        recognition.stop();
    }
    document.getElementById('voiceBtn').classList.remove('recording');
}

// 发送消息
window.sendMessage = async function sendMessage() {
    const inputBox = document.getElementById('inputBox');
    const message = inputBox.value.trim();
    
    if (!message) return;
    if (isRecording) stopRecording();
    
    // 如果没有当前会话，先创建一个
    if (!currentConversationId) {
        await createNewConversation();
    }
    
    addMessage('user', message);
    inputBox.value = '';
    finalTranscript = '';
    autoResize(inputBox);
    
    showTypingIndicator();
    
    try {
        const headers = { 'Content-Type': 'application/json' };
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ 
                message: message,
                session_id: currentConversationId 
            })
        });
        
        const data = await response.json();
        hideTypingIndicator();
        addMessage('assistant', data.response);
        
        // 保存到记忆，关联会话ID
        await saveToMemory(message, data.response);
        
        // 刷新左侧会话列表
        await loadConversations();
        
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
    
    const avatarClass = role === 'user' ? 'user-avatar' : 'ai-avatar';
    const avatarSvg = role === 'user' 
        ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#5c5c5c"/><path d="M12 12C13.66 12 15 10.66 15 9C15 7.34 13.66 6 12 6C10.34 6 9 7.34 9 9C9 10.66 10.34 12 12 12ZM12 14C9.33 14 4 15.34 4 18V19H20V18C20 15.34 14.67 14 12 14Z" fill="white"/></svg>'
        : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="12" fill="#10a37f"/><path d="M12 4L14 8L18 9L15 12L16 16L12 14L8 16L9 12L6 9L10 8L12 4Z" fill="white"/></svg>';
    
    messageDiv.innerHTML = 
        '<div class="avatar ' + avatarClass + '">' + avatarSvg + '</div>' +
        '<div class="message-content">' + formatContent(content) + '</div>';
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function formatContent(content) {
    if (!content) return '';
    // 将换行符统一转换为 <br>
    var result = content.split("\\n").join("<br>");
    return result;
}

function showTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const indicator = document.createElement('div');
    indicator.id = 'typingIndicator';
    indicator.className = 'message assistant';
    indicator.innerHTML = 
        '<div class="avatar ai-avatar"></div>' +
        '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    messagesDiv.appendChild(indicator);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

// 保存到记忆，关联会话ID
async function saveToMemory(userMessage, aiResponse) {
    try {
        const headers = { 'Content-Type': 'application/json' };
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        await fetch('/api/voice', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                content: '[对话] 我说：' + userMessage,
                conversation_id: currentConversationId
            })
        });
        
        await fetch('/api/voice', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                content: '[对话] AI 回复：' + aiResponse,
                conversation_id: currentConversationId
            })
        });
    } catch (error) {
        console.error('保存记忆失败:', error);
    }
}

// 创建新会话
async function createNewConversation() {
    try {
        const headers = { 'Content-Type': 'application/json' };
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        const response = await fetch('/api/conversations', {
            method: 'POST',
            headers: headers
        });
        const conv = await response.json();
        currentConversationId = conv.id;
        await loadConversations();
        return conv;
    } catch (error) {
        console.error('创建会话失败:', error);
    }
}

// 新对话按钮
window.newChat = async function newChat() {
    currentConversationId = null;
    
    document.getElementById('chatMessages').innerHTML = 
        '<div class="message assistant">' +
        '<div class="avatar ai-avatar"></div>' +
        '<div class="message-content">' +
        '你好！我是你的数字记忆助手。你可以和我聊天，我会记住我们的对话。' +
        '</div></div>';
    
    document.querySelectorAll('.chat-history-item').forEach(item => {
        item.classList.remove('active');
    });
}

// 加载会话列表
async function loadConversations() {
    try {
        const headers = {};
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        const response = await fetch('/api/conversations', { headers: headers });
        const data = await response.json();
        const conversations = data.conversations || [];
        
        const historyDiv = document.getElementById('chatHistory');
        historyDiv.innerHTML = '';
        
        if (conversations.length === 0) {
            historyDiv.innerHTML = '<div style="color: #8e8ea0; padding: 10px; font-size: 14px;">暂无历史记录</div>';
            return;
        }
        
        conversations.forEach(function(conv) {
            var item = document.createElement('div');
            item.className = 'chat-history-item';
            if (conv.id === currentConversationId) {
                item.classList.add('active');
            }

            // 标题区域
            var titleSpan = document.createElement('span');
            titleSpan.className = 'conv-title';
            titleSpan.textContent = conv.title || '新对话';
            titleSpan.onclick = function() { loadConversation(conv.id, item); };

            // 删除按钮
            var deleteBtn = document.createElement('span');
            deleteBtn.className = 'conv-delete';
            deleteBtn.textContent = '×';
            deleteBtn.title = '删除会话';
            deleteBtn.onclick = function(e) {
                e.stopPropagation();
                deleteConversation(conv.id);
            };

            item.appendChild(titleSpan);
            item.appendChild(deleteBtn);
            item.title = (conv.created_at || '') + ' (' + (conv.message_count || 0) + '条消息)';

            historyDiv.appendChild(item);
        });
    } catch (error) {
        console.error('加载会话列表失败:', error);
    }
}

// 删除会话
async function deleteConversation(convId) {
    if (!confirm('确定要删除这个会话吗？')) {
        return;
    }
    
    try {
        const headers = {};
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        await fetch('/api/conversations/' + convId, {
            method: 'DELETE',
            headers: headers
        });
        
        // 如果删除的是当前会话，重置为新对话界面
        if (convId === currentConversationId) {
            newChat();
        }
        
        await loadConversations();
    } catch (error) {
        console.error('删除会话失败:', error);
    }
}

// 加载特定会话
async function loadConversation(convId, clickedItem) {
    try {
        currentConversationId = convId;
        
        // 更新高亮
        document.querySelectorAll('.chat-history-item').forEach(item => {
            item.classList.remove('active');
        });
        if (clickedItem) {
            clickedItem.classList.add('active');
        }
        
        // 获取会话消息
        const headers = {};
        if (authToken) {
            headers['Authorization'] = 'Bearer ' + authToken;
        }
        
        const response = await fetch('/api/conversations/' + convId + '/messages', { headers: headers });
        const data = await response.json();
        const messages = data.messages || [];
        
        // 清空并重新渲染聊天区域
        const messagesDiv = document.getElementById('chatMessages');
        messagesDiv.innerHTML = '';
        
        if (messages.length === 0) {
            messagesDiv.innerHTML = 
                '<div class="message assistant">' +
                '<div class="avatar ai-avatar"></div>' +
                '<div class="message-content">这个会话还没有消息。</div></div>';
            return;
        }
        
        messages.forEach(msg => {
            if (msg.content && msg.content.startsWith('[对话] 我说：')) {
                addMessage('user', msg.content.replace('[对话] 我说：', ''));
            } else if (msg.content && msg.content.startsWith('[对话] AI 回复：')) {
                addMessage('assistant', msg.content.replace('[对话] AI 回复：', ''));
            }
        });
        
        // 手机端自动关闭侧边栏
        if (window.innerWidth <= 768) {
            document.getElementById('sidebar').classList.remove('open');
        }
        
    } catch (error) {
        console.error('加载会话失败:', error);
    }
}

// 导出到全局作用域（供 HTML onclick/onkeydown 调用）
window.toggleSidebar = function() {
    document.getElementById('sidebar').classList.toggle('open');
};

window.autoResize = function(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
};

window.handleKeyDown = function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
};

// 页面加载完成
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    loadConversations();
});
    </script>
</body>
</html>"""
    return html

@app.get("/api/records")
async def get_records(current_user: dict = Depends(get_current_user)):
    """获取当前用户的记录列表"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求获取记录列表")
    records = load_records()
    # 按用户过滤（兼容旧数据：没有 user_id 的数据对所有人可见）
    user_records = [r for r in records if r.get("user_id", "") == user_id or not r.get("user_id")]
    user_records.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
    return {"total": len(user_records), "records": user_records}


@app.get("/api/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    """获取当前用户的会话列表（按更新时间倒序）"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求获取会话列表")
    conversations = load_conversations()
    # 按用户过滤（兼容旧数据：没有 user_id 的数据对所有人可见）
    user_conversations = [c for c in conversations if c.get("user_id", "") == user_id or not c.get("user_id")]
    user_conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"conversations": user_conversations}


@app.post("/api/conversations")
async def create_conversation(current_user: dict = Depends(get_current_user)):
    """创建新会话"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 创建新会话")
    now = datetime.now()
    conv_id = f"conv_{now.strftime('%Y%m%d_%H%M%S')}"
    iso_now = now.isoformat()

    new_conv = {
        "id": conv_id,
        "title": "新对话",
        "created_at": iso_now,
        "updated_at": iso_now,
        "message_count": 0,
        "user_id": user_id,  # 数据隔离
    }

    conversations = load_conversations()
    conversations.append(new_conv)
    save_conversations(conversations)

    return new_conv


@app.get("/api/conversations/{conv_id}/messages")
async def get_conversation_messages(conv_id: str, current_user: dict = Depends(get_current_user)):
    """获取特定会话的所有消息"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求获取会话 {conv_id} 的消息")
    
    # 验证会话归属
    conversations = load_conversations()
    conv = next((c for c in conversations if c.get("id") == conv_id), None)
    if conv and conv.get("user_id") and conv.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    records = load_records()
    messages = [r for r in records if r.get("conversation_id") == conv_id]
    # 按记录 id 排序（包含时间信息）
    messages.sort(key=lambda x: x.get("id", ""))
    return {"messages": messages}


class ConversationUpdate(BaseModel):
    """会话更新模型（目前仅支持标题）"""
    title: str | None = None


@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, data: ConversationUpdate, current_user: dict = Depends(get_current_user)):
    """更新会话信息（如标题）"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 更新会话 {conv_id}")
    
    conversations = load_conversations()
    updated = False

    for conv in conversations:
        if conv.get("id") == conv_id:
            # 验证会话归属
            if conv.get("user_id") and conv.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="无权访问此会话")
            
            if data.title is not None:
                conv["title"] = data.title
            conv["updated_at"] = datetime.now().isoformat()
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="会话不存在")

    save_conversations(conversations)
    return {"status": "ok"}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, current_user: dict = Depends(get_current_user)):
    """删除会话及其所有消息"""
    user_id = current_user.get("uid", "")
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 删除会话 {conv_id}")
    
    # 验证会话归属
    conversations = load_conversations()
    conv = next((c for c in conversations if c.get("id") == conv_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    if conv.get("user_id") and conv.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    
    # 删除会话
    conversations = [c for c in conversations if c.get("id") != conv_id]
    save_conversations(conversations)

    # 删除该会话的所有消息
    records = load_records()
    deleted_record_ids = [r.get("id") for r in records if r.get("conversation_id") == conv_id]
    records = [r for r in records if r.get("conversation_id") != conv_id]
    save_records(records)
    
    # ========== 清理 RAG 索引 ==========
    if deleted_record_ids:
        try:
            # 从 chunks_metadata.json 中删除对应记录
            if METADATA_PATH.exists():
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                original_count = len(metadata)
                # 过滤掉已删除的记录
                metadata = [m for m in metadata if m.get("id") not in deleted_record_ids]
                new_count = len(metadata)
                
                if new_count < original_count:
                    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    logger.info(f"🗑️ 从 RAG 元数据中删除了 {original_count - new_count} 条记录")
        except Exception as e:
            logger.warning(f"⚠️ 清理 RAG 索引时出错（不影响删除操作）: {e}", exc_info=True)
        
        # ========== 新增：清理 all_chunks.json ==========
        try:
            all_chunks_path = VECTOR_INDEXER_DIR / "all_chunks.json"
            if all_chunks_path.exists():
                with open(all_chunks_path, 'r', encoding='utf-8') as f:
                    all_chunks = json.load(f)
                
                original_count = len(all_chunks)
                # 过滤掉已删除的记录
                all_chunks = [c for c in all_chunks if c.get("id") not in deleted_record_ids]
                new_count = len(all_chunks)
                
                if new_count < original_count:
                    with open(all_chunks_path, 'w', encoding='utf-8') as f:
                        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                    logger.info(f"🗑️ 从 all_chunks.json 中删除了 {original_count - new_count} 条记录")
        except Exception as e:
            logger.warning(f"⚠️ 清理 all_chunks.json 时出错（不影响删除操作）: {e}", exc_info=True)
        # ========== 新增结束 ==========
        
        # ========== 立即触发全量索引重建（不等定时任务）==========
        if deleted_record_ids:
            try:
                # 创建标记文件（check_and_rebuild_index 需要这个标记）
                FLAG_FILE.touch()
                # 立即触发全量索引重建
                scheduler.add_job(
                    check_and_rebuild_index,
                    id=f'delete_rebuild_{conv_id}',
                    name=f'删除后重建索引-{conv_id}',
                    replace_existing=True
                )
                logger.info("📌 已触发删除后全量索引重建")
            except Exception as e:
                logger.warning(f"⚠️ 触发重建失败，已设置标记: {e}", exc_info=True)
        # ========== 重建结束 ==========
    # ========== 清理结束 ==========
    
    # ========== 新增：清理服务端会话历史缓存 ==========
    # 删除该会话对应的服务端对话历史
    if conv_id in conversation_histories:
        del conversation_histories[conv_id]
        logger.info(f"🗑️ 已清理会话 {conv_id} 的服务端历史缓存")
    # 兼容旧的 "default" session（如果所有对话都混在 default 里）
    if "default" in conversation_histories:
        del conversation_histories["default"]
        logger.info(f"🗑️ 已清理 default 服务端历史缓存")
    # ========== 新增结束 ==========

    return {"status": "ok"}

@app.post("/api/voice")
async def add_voice_record(request: VoiceRecordRequest, current_user: dict = Depends(get_current_user)):
    """
    API 端点：添加语音记录（方案 B）
    快捷指令可以通过 POST 请求调用此端点
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 添加语音记录")
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")

    content = request.content.strip()
    conversation_id = request.conversation_id

    # 创建新记录（可带会话 ID）
    user_id = current_user.get("uid", "")
    record = create_record(content, conversation_id=conversation_id, user_id=user_id)
    
    # 加载现有记录并追加
    records = load_records()
    records.append(record)
    save_records(records)
    
    # 如果有会话 ID，更新会话的消息数与标题
    if conversation_id:
        conversations = load_conversations()
        for conv in conversations:
            if conv.get("id") == conversation_id:
                conv["message_count"] = conv.get("message_count", 0) + 1
                conv["updated_at"] = datetime.now().isoformat()
                # 如果是默认标题且是用户发言，可以用内容更新标题
                if conv.get("title") == "新对话" and content.startswith("[对话] 我说："):
                    raw = content.replace("[对话] 我说：", "").strip()
                    title = raw[:25] + ("..." if len(raw) > 25 else "")
                    if title:
                        conv["title"] = title
                break
        save_conversations(conversations)

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
async def add_voice_record_get(content: str, current_user: dict = Depends(get_current_user)):
    """
    GET 方式添加语音记录
    快捷指令可以直接构建 URL: /api/voice/add?content=文本内容
    这样不需要配置 JSON 请求体，大大简化快捷指令的操作
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 通过 GET 方式添加语音记录")
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空，请使用 ?content=文本内容")
    
    # 创建新记录
    user_id = current_user.get("uid", "")
    record = create_record(content.strip(), user_id=user_id)
    
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
async def update_voice_record(record_id: str, request: VoiceRecordRequest, current_user: dict = Depends(get_current_user)):
    """
    API 端点：更新语音记录
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 更新语音记录 {record_id}")
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
                    <li><a href="/scan">🔍 状态扫描</a></li>
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
async def chat_api(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """RAG 聊天 API 端点"""
    logger.info(f"📝 用户 {current_user.get('email', 'unknown')} 发起聊天请求")
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

def _parse_json_response(ai_response: str, stage_name: str = "扫描"):
    """
    解析 AI 返回的 JSON 响应（带容错处理）
    
    Args:
        ai_response: AI 返回的原始文本
        stage_name: 阶段名称（用于日志）
        
    Returns:
        dict: 解析后的 JSON 对象，如果失败则返回 None
    """
    json_error = None
    
    # 方法1: 尝试直接解析
    try:
        return json.loads(ai_response)
    except json.JSONDecodeError as e:
        json_error = e
        logger.debug(f"⚠️  [{stage_name}] 直接解析失败，尝试提取代码块: {e}")
        
        # 方法2: 尝试提取 ```json ... ``` 代码块中的内容
        json_block_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',       # ``` ... ```
            r'```json\s*(.*?)```',      # ```json ... ``` (单行)
            r'```\s*(.*?)```'           # ``` ... ``` (单行)
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, ai_response, re.DOTALL)
            if match:
                extracted_json = match.group(1).strip()
                try:
                    result = json.loads(extracted_json)
                    logger.info(f"✅ [{stage_name}] 从代码块中提取 JSON 成功")
                    return result
                except json.JSONDecodeError:
                    continue
    
    logger.error(f"❌ [{stage_name}] JSON 解析失败: {json_error}")
    return None

def _stage2a_screening(records_data: str, background_content: str, client: OpenAI) -> dict:
    """
    Stage 2a: 初筛（deepseek）
    识别哪些记录值得深挖，给出初步观察
    
    Args:
        records_data: 格式化后的记录文本
        background_content: background.md 的内容
        client: OpenAI 客户端
        
    Returns:
        dict: 初筛结果，包含 relevant_items, initial_observations, suggested_focus
        如果失败则返回 {"error": "..."}
    """
    import time
    start_time = time.time()
    
    logger.info("🔍 [Stage 2a] 开始初筛（deepseek）...")
    
    screening_prompt = f"""分析以下语音记录，识别哪些记录值得深挖，给出初步观察。

分析标准：
{background_content}

待分析记录：
{records_data}

任务：
1. 识别值得关注的记录项（按索引编号）
2. 总结初步观察到的模式和趋势（2-3段话）
3. 建议重点关注的方向

要求：返回 JSON 格式：
{{
  "relevant_items": [
    {{"record_index": 1, "summary": "一句话概括", "why_relevant": "为什么值得关注"}}
  ],
  "initial_observations": "初步观察到的模式和趋势（2-3段话）",
  "suggested_focus": ["情绪波动", "工作压力"]
}}

只返回 JSON，不要其他文字。
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek",
            messages=[
                {
                    "role": "system",
                    "content": "你是个人状态监控初筛分析师。只返回 JSON，格式：{\"relevant_items\": [...], \"initial_observations\": \"...\", \"suggested_focus\": [...]}。不要其他文字。"
                },
                {
                    "role": "user",
                    "content": screening_prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000  # 初筛不需要太长输出
        )
        
        elapsed_time = time.time() - start_time
        
        # 检查响应
        if not response.choices or not response.choices[0].message.content:
            logger.warning("⚠️  [Stage 2a] deepseek 返回空响应")
            return {"error": "deepseek 返回空响应"}
        
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"✅ [Stage 2a] deepseek 初筛完成，耗时 {elapsed_time:.2f}秒，响应长度: {len(ai_response)} 字符")
        
        # 解析 JSON
        screening_result = _parse_json_response(ai_response, "Stage 2a")
        if screening_result is None:
            return {"error": "Stage 2a JSON 解析失败"}
        
        # 验证结构
        if "relevant_items" not in screening_result:
            screening_result["relevant_items"] = []
        if "initial_observations" not in screening_result:
            screening_result["initial_observations"] = "未生成初步观察"
        if "suggested_focus" not in screening_result:
            screening_result["suggested_focus"] = []
        
        logger.info(f"📊 [Stage 2a] 识别到 {len(screening_result['relevant_items'])} 条值得关注的记录")
        return screening_result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.exception(f"❌ [Stage 2a] deepseek 调用失败（耗时 {elapsed_time:.2f}秒）: {e}")
        return {"error": f"Stage 2a 失败: {str(e)}"}

def _stage2b_deep_analysis(screening_result: dict, background_content: str, client: OpenAI) -> dict:
    """
    Stage 2b: 深挖（gemini-2.5-pro）
    基于初筛结果，生成最终的 patterns 和 summary
    
    Args:
        screening_result: Stage 2a 的输出
        background_content: background.md 的内容
        client: OpenAI 客户端
        
    Returns:
        dict: 深度分析结果，包含 patterns 和 summary
        如果失败则返回 {"error": "..."}
    """
    import time
    start_time = time.time()
    
    logger.info("🔍 [Stage 2b] 开始深挖（gemini-2.5-pro）...")
    
    # 格式化 Stage 2a 的输出
    relevant_items_text = ""
    if screening_result.get("relevant_items"):
        for item in screening_result["relevant_items"]:
            relevant_items_text += f"- 记录 {item.get('record_index', '?')}: {item.get('summary', '')}（{item.get('why_relevant', '')}）\n"
    
    deep_analysis_prompt = f"""基于以下初筛结果，进行深度分析，识别情绪模式、工作压力、项目进展、人际关系问题。

分析标准：
{background_content}

初筛结果：
- 值得关注的记录：
{relevant_items_text if relevant_items_text else "（无）"}

- 初步观察：
{screening_result.get('initial_observations', '（无）')}

- 建议关注方向：
{', '.join(screening_result.get('suggested_focus', [])) if screening_result.get('suggested_focus') else '（无）'}

任务：基于以上初筛结果，生成详细的模式识别报告和建议。

要求：返回 JSON 格式，包含 patterns 数组和 summary 字符串。
patterns 格式：{{"importance": "High|Medium|Low", "pattern": "描述", "evidence": "证据", "suggestion": "建议"}}

只返回 JSON，不要其他文字。
"""
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {
                    "role": "system",
                    "content": "你是个人状态监控分析师。只返回 JSON，格式：{\"patterns\": [...], \"summary\": \"...\"}。不要其他文字。"
                },
                {
                    "role": "user",
                    "content": deep_analysis_prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        elapsed_time = time.time() - start_time
        
        # 检查响应
        if not response.choices or not response.choices[0].message.content:
            logger.warning("⚠️  [Stage 2b] gemini-2.5-pro 返回空响应")
            return {"error": "gemini-2.5-pro 返回空响应"}
        
        # 检查是否因为长度限制被截断
        choice = response.choices[0]
        if choice.finish_reason == 'length':
            logger.warning("⚠️  [Stage 2b] 响应被截断（达到 max_tokens 限制）")
        
        ai_response = choice.message.content.strip()
        logger.info(f"✅ [Stage 2b] gemini-2.5-pro 深挖完成，耗时 {elapsed_time:.2f}秒，响应长度: {len(ai_response)} 字符")
        
        # 解析 JSON
        deep_dive_report = _parse_json_response(ai_response, "Stage 2b")
        if deep_dive_report is None:
            return {"error": "Stage 2b JSON 解析失败"}
        
        # 验证结构
        if "patterns" not in deep_dive_report:
            deep_dive_report["patterns"] = []
        if "summary" not in deep_dive_report:
            deep_dive_report["summary"] = "分析完成，但未生成总结。"
        
        logger.info(f"📊 [Stage 2b] 识别到 {len(deep_dive_report.get('patterns', []))} 个模式")
        return deep_dive_report
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.exception(f"❌ [Stage 2b] gemini-2.5-pro 调用失败（耗时 {elapsed_time:.2f}秒）: {e}")
        return {"error": f"Stage 2b 失败: {str(e)}"}

def _single_model_analysis(records_data: str, background_content: str, client: OpenAI) -> dict:
    """
    单模型分析（Fallback）
    原来的单模型逻辑，作为双模型失败时的兜底
    
    Args:
        records_data: 格式化后的记录文本
        background_content: background.md 的内容
        client: OpenAI 客户端
        
    Returns:
        dict: 分析结果，包含 patterns 和 summary
        如果失败则返回 {"error": "..."}
    """
    import time
    start_time = time.time()
    
    logger.info("🔄 [Fallback] 使用单模型模式...")
    
    analysis_prompt = f"""分析以下语音记录，识别情绪模式、工作压力、项目进展、人际关系问题。

分析标准：
{background_content}

待分析记录：
{records_data}

要求：返回 JSON 格式，包含 patterns 数组和 summary 字符串。
patterns 格式：{{"importance": "High|Medium|Low", "pattern": "描述", "evidence": "证据", "suggestion": "建议"}}

只返回 JSON，不要其他文字。
"""
    
    # 尝试使用不同的模型
    models_to_try = ["deepseek", "gemini-2.5-pro", "gpt-5"]
    
    last_error = None
    response = None
    
    for model_name in models_to_try:
        try:
            logger.info(f"   尝试使用模型: {model_name}")
            
            temperature = 1.0 if model_name == "gpt-5" else 0.7
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是个人状态监控分析师。只返回 JSON，格式：{{\"patterns\": [...], \"summary\": \"...\"}}。不要其他文字。"
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=4000
            )
            
            logger.info(f"   ✅ 模型 {model_name} 调用成功")
            
            # 检查是否因为长度限制被截断
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.finish_reason == 'length':
                    logger.warning("⚠️  [Fallback] 响应被截断（达到 max_tokens 限制）")
                    if not choice.message.content:
                        if model_name != models_to_try[-1]:
                            continue
                        else:
                            return {
                                "error": "AI 响应被截断且内容为空。"
                            }
            
            break  # 成功则跳出循环
            
        except Exception as model_error:
            last_error = model_error
            logger.warning(f"   ⚠️  模型 {model_name} 调用失败: {str(model_error)[:200]}")
            if model_name != models_to_try[-1]:
                logger.info(f"   尝试下一个模型...")
                continue
            else:
                # 所有模型都失败
                elapsed_time = time.time() - start_time
                logger.error(f"❌ [Fallback] 所有模型都失败（耗时 {elapsed_time:.2f}秒）")
                return {"error": f"所有模型调用都失败: {str(last_error)[:200]}"}
    
    if response is None:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ [Fallback] 未获得有效响应（耗时 {elapsed_time:.2f}秒）")
        return {"error": "未获得有效响应"}
    
    elapsed_time = time.time() - start_time
    ai_response = response.choices[0].message.content.strip()
    logger.info(f"✅ [Fallback] 单模型分析完成，耗时 {elapsed_time:.2f}秒，响应长度: {len(ai_response)} 字符")
    
    # 解析 JSON
    deep_dive_report = _parse_json_response(ai_response, "Fallback")
    if deep_dive_report is None:
        return {"error": "Fallback JSON 解析失败"}
    
    # 验证结构
    if "patterns" not in deep_dive_report:
        deep_dive_report["patterns"] = []
    if "summary" not in deep_dive_report:
        deep_dive_report["summary"] = "分析完成，但未生成总结。"
    
    logger.info(f"📊 [Fallback] 识别到 {len(deep_dive_report.get('patterns', []))} 个模式")
    return deep_dive_report

def _perform_scan():
    """
    执行扫描的核心逻辑（可复用）
    使用双模型协作架构：Stage 2a (deepseek 初筛) -> Stage 2b (gpt-5 深挖)
    如果失败，降级到单模型模式
    
    Returns:
        dict: 扫描结果，格式为 {
            "scan_period": "...",
            "records_analyzed": int,
            "deep_dive_report": {...}
        } 或包含 "error" 字段的错误结果
    """
    try:
        # ========== Stage 1: 数据收集 ==========
        logger.info("🔍 [扫描] 开始执行个人状态监控扫描...")
        
        # 读取 voice_records.json
        if not RECORDS_FILE.exists():
            return {
                "scan_period": None,
                "records_analyzed": 0,
                "error": "暂无语音记录文件，请先添加一些记录。"
            }
        
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            all_records = json.load(f)
        
        # 过滤最近 7 天的记录
        today = datetime.now().date()
        seven_days_ago = today - timedelta(days=7)
        
        recent_records = []
        for record in all_records:
            record_date_str = record.get('date', '')
            if not record_date_str:
                continue
            
            try:
                record_date = datetime.strptime(record_date_str, '%Y-%m-%d').date()
                if record_date >= seven_days_ago:
                    recent_records.append(record)
            except ValueError:
                logger.warning(f"⚠️  无法解析日期格式: {record_date_str}")
                continue
        
        # 按日期排序（从旧到新）
        recent_records.sort(key=lambda x: x.get('date', ''))
        
        if not recent_records:
            return {
                "scan_period": f"{seven_days_ago} 至 {today}",
                "records_analyzed": 0,
                "error": f"最近 7 天（{seven_days_ago} 至 {today}）没有语音记录。"
            }
        
        logger.info(f"📊 [扫描] 找到 {len(recent_records)} 条最近 7 天的记录")
        
        # ========== Stage 2: 深度分析（双模型协作架构）==========
        # 读取 background.md
        background_file = VECTOR_INDEXER_DIR / "background.md"
        if not background_file.exists():
            return {
                "error": "background.md 文件不存在，无法进行分析。"
            }
        
        with open(background_file, 'r', encoding='utf-8') as f:
            background_content = f.read()
        
        # 构造待分析数据（格式化记录）
        records_text = []
        for i, record in enumerate(recent_records, 1):
            record_id = record.get('id', '')
            record_date = record.get('date', '')
            record_time = record.get('time', '')
            record_content = record.get('content', '')
            # 截断单条记录内容，避免过长（每条记录最多 500 字符）
            if len(record_content) > 500:
                record_content = record_content[:500] + "...[已截断]"
            records_text.append(f"记录 {i} [ID: {record_id}, 日期: {record_date} {record_time}]:\n{record_content}\n")
        
        records_data = "\n".join(records_text)
        
        # 如果总内容太长，进行截断（保留最近的记录）
        MAX_CONTENT_LENGTH = 8000  # 减少到 8000 字符，避免 prompt 过长占用太多 token
        if len(records_data) > MAX_CONTENT_LENGTH:
            logger.warning(f"⚠️  [扫描] 内容过长 ({len(records_data)} 字符)，截断到 {MAX_CONTENT_LENGTH} 字符")
            # 保留最近的记录（从后往前截断）
            records_data = records_data[-MAX_CONTENT_LENGTH:]
            # 找到第一个完整的记录开始位置
            first_newline = records_data.find('\n')
            if first_newline > 0:
                records_data = records_data[first_newline+1:]
            records_data = f"[注意：由于内容过长，仅显示部分记录]\n{records_data}"
        
        # 初始化 OpenAI 客户端
        api_key = os.getenv("AI_BUILDER_TOKEN")
        if not api_key:
            return {
                "error": "AI_BUILDER_TOKEN 未设置，无法进行分析。"
            }
        
        client = OpenAI(
            base_url="https://space.ai-builders.com/backend/v1",
            api_key=api_key,
            timeout=120.0,  # 增加超时时间到 120 秒
            max_retries=3  # 最大重试 3 次
        )
        
        logger.info(f"🤖 [扫描] 开始双模型协作分析...")
        logger.info(f"   - 记录数量: {len(recent_records)} 条")
        logger.info(f"   - 数据长度: {len(records_data)} 字符")
        
        # ========== 尝试双模型协作架构 ==========
        deep_dive_report = None
        
        try:
            # Stage 2a: 初筛（deepseek）
            screening_result = _stage2a_screening(records_data, background_content, client)
            
            if "error" in screening_result:
                logger.warning(f"⚠️  [扫描] Stage 2a 失败，降级到单模型模式: {screening_result['error']}")
                # 降级到单模型模式
                deep_dive_report = _single_model_analysis(records_data, background_content, client)
            else:
                # Stage 2b: 深挖（gpt-5）
                deep_dive_report = _stage2b_deep_analysis(screening_result, background_content, client)
                
                if "error" in deep_dive_report:
                    logger.warning(f"⚠️  [扫描] Stage 2b 失败，降级到单模型模式: {deep_dive_report['error']}")
                    # 降级到单模型模式
                    deep_dive_report = _single_model_analysis(records_data, background_content, client)
                
        except Exception as e:
            logger.exception(f"❌ [扫描] 双模型协作过程中出现异常，降级到单模型模式: {e}")
            # 降级到单模型模式
            deep_dive_report = _single_model_analysis(records_data, background_content, client)
        
        # 检查最终结果
        if deep_dive_report is None or "error" in deep_dive_report:
            error_msg = deep_dive_report.get("error", "未知错误") if deep_dive_report else "未获得分析结果"
            logger.error(f"❌ [扫描] 分析失败: {error_msg}")
            
            # 检查是否是连接问题
            error_str = str(error_msg).lower()
            if "connection" in error_str or "timeout" in error_str:
                return {
                    "error": "AI API 连接超时或失败。可能原因：1) 网络连接问题 2) 请求内容过长 3) API 服务暂时不可用。请稍后重试，或减少分析的数据量。",
                    "details": error_msg[:200],
                    "records_count": len(recent_records),
                    "content_length": len(records_data)
                }
            else:
                return {
                    "error": f"分析失败: {error_msg[:200]}"
                }
        
        # ========== 返回结果 ==========
        result = {
            "scan_period": f"{seven_days_ago} 至 {today}",
            "records_analyzed": len(recent_records),
            "deep_dive_report": deep_dive_report
        }
        
        logger.info(f"✅ [扫描] 扫描完成，识别到 {len(deep_dive_report.get('patterns', []))} 个模式")
        
        return result
        
    except Exception as e:
        logger.exception(f"❌ [扫描] 扫描过程出现异常: {e}")
        return {
            "error": f"扫描过程出现异常: {str(e)}"
        }

@app.get("/api/last-scan")
async def get_last_scan(current_user: dict = Depends(get_current_user)):
    """
    获取最近一次自动扫描的结果
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求获取扫描结果")
    if not SCAN_RESULTS_FILE.exists():
        return JSONResponse(
            status_code=200,
            content={
                "message": "暂无自动扫描结果"
            }
        )
    
    try:
        with open(SCAN_RESULTS_FILE, 'r', encoding='utf-8') as f:
            scan_result = json.load(f)
        return JSONResponse(status_code=200, content=scan_result)
    except Exception as e:
        logger.exception(f"❌ 读取扫描结果失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"读取扫描结果失败: {str(e)}"
            }
        )

@app.post("/api/trigger-auto-scan")
async def trigger_auto_scan(current_user: dict = Depends(get_current_user)):
    """
    手动触发自动扫描（立即执行一次并保存结果）
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 手动触发自动扫描...")
    try:
        auto_scan()  # 直接调用自动扫描函数
        
        # 读取刚保存的结果
        if SCAN_RESULTS_FILE.exists():
            with open(SCAN_RESULTS_FILE, 'r', encoding='utf-8') as f:
                scan_result = json.load(f)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "自动扫描已触发并完成",
                    "scan_result": scan_result
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "扫描完成但结果文件未生成"
                }
            )
    except Exception as e:
        logger.exception(f"❌ [手动触发] 触发自动扫描失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"触发自动扫描失败: {str(e)}"
            }
        )

@app.post("/run-scan")
async def run_scan(current_user: dict = Depends(get_current_user)):
    """
    个人状态监控扫描端点
    
    扫描最近 7 天的语音记录，进行深度分析，识别情绪模式、工作压力、项目进展等。
    返回包含模式识别和建议的分析报告。
    
    注意：手动扫描的结果也会保存到 scan_results.json（与自动扫描一致）
    """
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求执行扫描")
    result = _perform_scan()
    
    # 保存结果到文件（与自动扫描保持一致）
    try:
        scan_result = {
            "scan_time": datetime.now().isoformat(),
            "result": result,
            "trigger": "manual"  # 标记为手动触发
        }
        with open(SCAN_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(scan_result, f, ensure_ascii=False, indent=2)
        logger.info("✅ [手动扫描] 结果已保存到 scan_results.json")
    except Exception as e:
        logger.warning(f"⚠️  [手动扫描] 保存结果失败: {e}")
    
    # 处理错误情况
    if "error" in result:
        status_code = 500 if "error" in result and result.get("scan_period") is None else 200
        return JSONResponse(status_code=status_code, content=result)
    
    return JSONResponse(status_code=200, content=result)

@app.get("/api/index-status")
async def get_index_status_api(current_user: dict = Depends(get_current_user)):
    """获取索引重建状态"""
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求获取索引状态")
    status = get_index_status()
    return status

@app.post("/api/rebuild-index")
async def rebuild_index_api(background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    """手动触发索引重建"""
    logger.info(f"🔐 用户 {current_user.get('email', 'unknown')} 请求重建索引")
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

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理页面：整合记录、扫描、设置"""
    # 获取记录数据
    records = load_records()
    records.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
    records_count = len(records)
    
    # 生成记录列表 HTML
    records_html = ""
    if records:
        for r in records[:50]:  # 只显示最近50条
            records_html += f'''
            <div class="record-item">
                <div class="record-meta">
                    <span class="record-id">{r.get('id', '')}</span>
                    <span class="record-time">{r.get('date', '')} {r.get('time', '')}</span>
                </div>
                <div class="record-content">{r.get('content', '').replace('<', '&lt;').replace('>', '&gt;')[:200]}{"..." if len(r.get('content', '')) > 200 else ""}</div>
            </div>
            '''
    else:
        records_html = '<div class="empty-state">暂无记录</div>'
    
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>管理 - Digital Memory</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #343541;
                color: #ececf1;
                min-height: 100vh;
            }}
            
            .header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 15px 20px;
                background: #202123;
                border-bottom: 1px solid #565869;
            }}
            
            .header h1 {{
                font-size: 18px;
                font-weight: 500;
            }}
            
            .back-btn {{
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: transparent;
                border: 1px solid #565869;
                border-radius: 6px;
                color: #ececf1;
                cursor: pointer;
                font-size: 14px;
                text-decoration: none;
            }}
            
            .back-btn:hover {{
                background: #2a2b32;
            }}
            
            .tabs {{
                display: flex;
                background: #202123;
                border-bottom: 1px solid #565869;
            }}
            
            .tab {{
                padding: 15px 30px;
                background: transparent;
                border: none;
                color: #8e8ea0;
                cursor: pointer;
                font-size: 14px;
                border-bottom: 2px solid transparent;
                transition: all 0.2s;
            }}
            
            .tab:hover {{
                color: #ececf1;
            }}
            
            .tab.active {{
                color: #ececf1;
                border-bottom-color: #19c37d;
            }}
            
            .content {{
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .panel {{
                display: none;
            }}
            
            .panel.active {{
                display: block;
            }}
            
            /* 记录列表样式 */
            .records-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            
            .records-count {{
                color: #8e8ea0;
                font-size: 14px;
            }}
            
            .record-item {{
                background: #40414f;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 12px;
            }}
            
            .record-meta {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                font-size: 12px;
                color: #8e8ea0;
            }}
            
            .record-content {{
                line-height: 1.6;
                font-size: 14px;
            }}
            
            .empty-state {{
                text-align: center;
                color: #8e8ea0;
                padding: 40px;
            }}
            
            /* 扫描样式 */
            .scan-section {{
                background: #40414f;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            
            .scan-section h3 {{
                margin-bottom: 10px;
                font-size: 16px;
            }}
            
            .scan-section p {{
                color: #8e8ea0;
                font-size: 14px;
                margin-bottom: 15px;
            }}
            
            .btn {{
                padding: 10px 20px;
                background: #19c37d;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }}
            
            .btn:hover {{
                background: #1a7f5a;
            }}
            
            .btn:disabled {{
                background: #565869;
                cursor: not-allowed;
            }}
            
            .btn-secondary {{
                background: #565869;
            }}
            
            .btn-secondary:hover {{
                background: #6b6c7b;
            }}
            
            /* 设置样式 */
            .setting-item {{
                background: #40414f;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
            }}
            
            .setting-item h3 {{
                margin-bottom: 8px;
                font-size: 16px;
            }}
            
            .setting-item p {{
                color: #8e8ea0;
                font-size: 14px;
                margin-bottom: 15px;
            }}
            
            .progress-bar {{
                width: 100%;
                height: 8px;
                background: #565869;
                border-radius: 4px;
                overflow: hidden;
                margin: 15px 0;
            }}
            
            .progress-fill {{
                height: 100%;
                background: #19c37d;
                transition: width 0.3s;
            }}
            
            .status-text {{
                font-size: 13px;
                color: #8e8ea0;
            }}
            
            .status-text.success {{ color: #19c37d; }}
            .status-text.error {{ color: #ef4444; }}
            .status-text.running {{ color: #3b82f6; }}
            
            /* 扫描结果样式 */
            .scan-results {{
                margin-top: 20px;
            }}
            
            .pattern-item {{
                background: #2a2b32;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 10px;
                border-left: 3px solid #8e8ea0;
            }}
            
            .pattern-item.high {{ border-left-color: #ef4444; }}
            .pattern-item.medium {{ border-left-color: #f59e0b; }}
            .pattern-item.low {{ border-left-color: #8e8ea0; }}
            
            .pattern-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}
            
            .pattern-importance {{
                font-size: 11px;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: 500;
            }}
            
            .pattern-importance.high {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
            .pattern-importance.medium {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; }}
            .pattern-importance.low {{ background: rgba(142, 142, 160, 0.2); color: #8e8ea0; }}
            
            .pattern-title {{
                font-weight: 500;
            }}
            
            .pattern-content {{
                font-size: 14px;
                color: #8e8ea0;
                line-height: 1.5;
            }}
            
            .pattern-content p {{
                margin-bottom: 5px;
            }}
            
            .scan-summary {{
                background: #2a2b32;
                border-radius: 6px;
                padding: 15px;
                margin-top: 15px;
            }}
            
            .scan-summary h4 {{
                margin-bottom: 10px;
                font-size: 14px;
            }}
            
            .scan-summary p {{
                font-size: 14px;
                line-height: 1.6;
            }}
        </style>
        <script type="module">
            import {{ initializeApp }} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
            import {{ getAuth, onAuthStateChanged }} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";
            
            const firebaseConfig = {{
                apiKey: "AIzaSyDuWwWz2vWm6FV50w5ozL0DFoxfJfcEy0g",
                authDomain: "voice-journal-auth-ba3b0.firebaseapp.com",
                projectId: "voice-journal-auth-ba3b0"
            }};
            
            const app = initializeApp(firebaseConfig);
            const auth = getAuth(app);
            
            // 全局 token 存储
            window.authToken = null;
            
            // 检查登录状态
            onAuthStateChanged(auth, (user) => {{
                if (user) {{
                    user.getIdToken().then(token => {{
                        window.authToken = token;
                        document.getElementById('adminContent').style.display = 'block';
                        document.getElementById('loadingOverlay').style.display = 'none';
                    }});
                }} else {{
                    // 未登录，跳转到主页
                    window.location.href = '/';
                }}
            }});
            
            // Token 刷新（每 55 分钟）
            setInterval(() => {{
                const user = auth.currentUser;
                if (user) {{
                    user.getIdToken(true).then(token => {{
                        window.authToken = token;
                    }});
                }}
            }}, 55 * 60 * 1000);
        </script>
    </head>
    <body>
        <!-- 加载遮罩（等待认证检查） -->
        <div id="loadingOverlay" style="position:fixed;top:0;left:0;right:0;bottom:0;background:#343541;display:flex;align-items:center;justify-content:center;z-index:9999;">
            <div style="color:#ececf1;font-size:16px;">验证登录状态...</div>
        </div>
        
        <div id="adminContent" style="display:none;">
        <div class="header">
            <h1>⚙️ 管理设置</h1>
            <a href="/" class="back-btn">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                返回对话
            </a>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('records')">📝 记录</button>
            <button class="tab" onclick="switchTab('scan')">🔍 状态扫描</button>
            <button class="tab" onclick="switchTab('settings')">⚙️ 设置</button>
        </div>
        
        <div class="content">
            <!-- 记录面板 -->
            <div id="records-panel" class="panel active">
                <div class="records-header">
                    <h2>所有记录</h2>
                    <span class="records-count">共 {records_count} 条记录（显示最近 50 条）</span>
                </div>
                <div id="records-list">
                    {records_html}
                </div>
            </div>
            
            <!-- 扫描面板 -->
            <div id="scan-panel" class="panel">
                <div class="scan-section">
                    <h3>个人状态扫描</h3>
                    <p>扫描最近 7 天的记录，分析情绪模式、工作压力和生活状态。</p>
                    <button class="btn" id="scanBtn" onclick="startScan()">开始扫描</button>
                    <button class="btn btn-secondary" id="triggerAutoBtn" onclick="triggerAutoScan()" style="margin-left: 10px;">触发自动扫描</button>
                </div>
                
                <div id="lastScanInfo" class="scan-section" style="display: none;">
                    <h3>上次扫描结果</h3>
                    <p id="lastScanTime"></p>
                    <div id="lastScanPreview"></div>
                </div>
                
                <div id="scanResults" class="scan-results"></div>
            </div>
            
            <!-- 设置面板 -->
            <div id="settings-panel" class="panel">
                <div class="setting-item">
                    <h3>索引重建</h3>
                    <p>当记录同步出现问题时，可以手动重建 RAG 索引。</p>
                    <button class="btn" id="rebuildBtn" onclick="rebuildIndex()">手动重建索引</button>
                    <div class="progress-bar" id="progressBar" style="display: none;">
                        <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                    </div>
                    <p class="status-text" id="rebuildStatus"></p>
                </div>
                
                <div class="setting-item">
                    <h3>数据同步</h3>
                    <p>录音记录会自动同步到 RAG 系统，新记录会实时更新索引。</p>
                </div>
                
                <div class="setting-item">
                    <h3>定时任务</h3>
                    <p>• 索引重建检查：每 30 分钟<br>• 自动状态扫描：每小时</p>
                </div>
            </div>
        </div>
        
        <script>
            // Tab 切换
            function switchTab(tabName) {{
                // 更新 tab 状态
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                event.target.classList.add('active');
                
                // 更新面板显示
                document.querySelectorAll('.panel').forEach(panel => panel.classList.remove('active'));
                document.getElementById(tabName + '-panel').classList.add('active');
                
                // 切换到扫描时加载上次结果
                if (tabName === 'scan') {{
                    loadLastScan();
                }}
                
                // 切换到设置时检查索引状态
                if (tabName === 'settings') {{
                    checkIndexStatus();
                }}
            }}
            
            // ========== 扫描功能 ==========
            async function loadLastScan() {{
                try {{
                    const response = await fetch('/api/last-scan', {{
                        headers: {{ 'Authorization': 'Bearer ' + window.authToken }}
                    }});
                    if (response.ok) {{
                        const data = await response.json();
                        if (data.scan_time) {{
                            const lastScanInfo = document.getElementById('lastScanInfo');
                            const lastScanTime = document.getElementById('lastScanTime');
                            const lastScanPreview = document.getElementById('lastScanPreview');
                            
                            lastScanInfo.style.display = 'block';
                            
                            const scanDate = new Date(data.scan_time);
                            lastScanTime.textContent = '扫描时间: ' + scanDate.toLocaleString('zh-CN');
                            
                            if (data.result.error) {{
                                lastScanPreview.innerHTML = '<span style="color: #ef4444;">❌ ' + escapeHtml(data.result.error) + '</span>';
                            }} else {{
                                const patterns = data.result.deep_dive_report?.patterns || [];
                                const highCount = patterns.filter(p => p.importance === 'High').length;
                                const mediumCount = patterns.filter(p => p.importance === 'Medium').length;
                                lastScanPreview.innerHTML = '识别到 ' + patterns.length + ' 个模式' +
                                    (highCount > 0 ? ' (<span style="color:#ef4444;">High: ' + highCount + '</span>)' : '') +
                                    (mediumCount > 0 ? ' (<span style="color:#f59e0b;">Medium: ' + mediumCount + '</span>)' : '');
                            }}
                        }}
                    }}
                }} catch (error) {{
                    console.error('加载上次扫描失败:', error);
                }}
            }}
            
            async function startScan() {{
                const btn = document.getElementById('scanBtn');
                const results = document.getElementById('scanResults');
                
                btn.disabled = true;
                btn.textContent = '扫描中...';
                results.innerHTML = '<div class="empty-state">正在分析记录，请稍候...</div>';
                
                try {{
                    const response = await fetch('/run-scan', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + window.authToken }}
                    }});
                    const data = await response.json();
                    
                    if (data.error) {{
                        results.innerHTML = '<div class="scan-section"><p style="color:#ef4444;">❌ ' + escapeHtml(data.error) + '</p></div>';
                    }} else {{
                        displayScanResults(data);
                    }}
                    loadLastScan();
                }} catch (error) {{
                    results.innerHTML = '<div class="scan-section"><p style="color:#ef4444;">❌ 网络错误: ' + escapeHtml(error.message) + '</p></div>';
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '开始扫描';
                }}
            }}
            
            async function triggerAutoScan() {{
                const btn = document.getElementById('triggerAutoBtn');
                btn.disabled = true;
                btn.textContent = '触发中...';
                
                try {{
                    const response = await fetch('/api/trigger-auto-scan', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + window.authToken }}
                    }});
                    const data = await response.json();
                    
                    if (data.success && data.scan_result) {{
                        if (data.scan_result.result && !data.scan_result.result.error) {{
                            displayScanResults(data.scan_result.result);
                        }}
                    }}
                    loadLastScan();
                }} catch (error) {{
                    console.error('触发扫描失败:', error);
                }} finally {{
                    btn.disabled = false;
                    btn.textContent = '触发自动扫描';
                }}
            }}
            
            function displayScanResults(data) {{
                const results = document.getElementById('scanResults');
                const patterns = data.deep_dive_report?.patterns || [];
                const summary = data.deep_dive_report?.summary || '';
                
                let html = '<div class="scan-section"><p>扫描周期: ' + (data.scan_period || 'N/A') + ' | 分析记录: ' + (data.records_analyzed || 0) + ' 条</p></div>';
                
                if (patterns.length === 0) {{
                    html += '<div class="empty-state">未发现明显的模式或问题。</div>';
                }} else {{
                    patterns.forEach(pattern => {{
                        const importance = (pattern.importance || 'low').toLowerCase();
                        html += '<div class="pattern-item ' + importance + '">' +
                            '<div class="pattern-header">' +
                            '<span class="pattern-importance ' + importance + '">' + (pattern.importance || 'Low') + '</span>' +
                            '<span class="pattern-title">' + escapeHtml(pattern.pattern || '') + '</span>' +
                            '</div>' +
                            '<div class="pattern-content">' +
                            '<p><strong>证据：</strong>' + escapeHtml(pattern.evidence || '无') + '</p>' +
                            '<p><strong>建议：</strong>' + escapeHtml(pattern.suggestion || '无') + '</p>' +
                            '</div></div>';
                    }});
                }}
                
                if (summary) {{
                    html += '<div class="scan-summary"><h4>总结</h4><p>' + escapeHtml(summary) + '</p></div>';
                }}
                
                results.innerHTML = html;
            }}
            
            // ========== 索引重建功能 ==========
            let statusPollInterval = null;
            
            async function checkIndexStatus() {{
                try {{
                    const response = await fetch('/api/index-status', {{
                        headers: {{ 'Authorization': 'Bearer ' + window.authToken }}
                    }});
                    const data = await response.json();
                    updateIndexStatusDisplay(data);
                    
                    if (data.status === 'running') {{
                        if (!statusPollInterval) {{
                            statusPollInterval = setInterval(checkIndexStatus, 2000);
                        }}
                    }} else {{
                        if (statusPollInterval) {{
                            clearInterval(statusPollInterval);
                            statusPollInterval = null;
                        }}
                    }}
                }} catch (error) {{
                    console.error('获取状态失败:', error);
                }}
            }}
            
            function updateIndexStatusDisplay(data) {{
                const btn = document.getElementById('rebuildBtn');
                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                const status = document.getElementById('rebuildStatus');
                
                status.textContent = data.message || '';
                status.className = 'status-text';
                
                if (data.status === 'running') {{
                    progressBar.style.display = 'block';
                    progressFill.style.width = data.progress + '%';
                    status.classList.add('running');
                    btn.disabled = true;
                }} else if (data.status === 'completed') {{
                    progressBar.style.display = 'block';
                    progressFill.style.width = '100%';
                    status.classList.add('success');
                    btn.disabled = false;
                    setTimeout(() => {{ progressBar.style.display = 'none'; }}, 3000);
                }} else if (data.status === 'failed') {{
                    progressBar.style.display = 'none';
                    status.classList.add('error');
                    btn.disabled = false;
                }} else {{
                    progressBar.style.display = 'none';
                    btn.disabled = false;
                }}
            }}
            
            async function rebuildIndex() {{
                const btn = document.getElementById('rebuildBtn');
                const status = document.getElementById('rebuildStatus');
                
                btn.disabled = true;
                status.textContent = '正在启动...';
                status.className = 'status-text running';
                
                try {{
                    const response = await fetch('/api/rebuild-index', {{
                        method: 'POST',
                        headers: {{ 'Authorization': 'Bearer ' + window.authToken }}
                    }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        checkIndexStatus();
                        if (!statusPollInterval) {{
                            statusPollInterval = setInterval(checkIndexStatus, 2000);
                        }}
                    }} else {{
                        status.textContent = '❌ ' + (data.error || '启动失败');
                        status.className = 'status-text error';
                        btn.disabled = false;
                    }}
                }} catch (error) {{
                    status.textContent = '❌ 网络错误';
                    status.className = 'status-text error';
                    btn.disabled = false;
                }}
            }}
            
            function escapeHtml(text) {{
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
        </script>
        </div>
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
