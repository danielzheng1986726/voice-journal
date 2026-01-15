#!/usr/bin/env python3
"""
Agent 主程序 - ReAct 模式版
不依赖不稳定的 Native Tool Calling，通过文本协议实现 Agentic RAG
"""

import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from retriever import VectorRetriever

# 加载 .env 文件中的环境变量
load_dotenv()

# ================= 配置与日志 =================
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志（使用轮转日志处理器，限制单个文件大小为 10MB，保留 5 个备份文件）
log_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 使用 RotatingFileHandler 实现日志轮转
# maxBytes: 单个日志文件最大 10MB
# backupCount: 保留 5 个备份文件
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'agent.log'),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_format)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

logger = logging.getLogger("vector_indexer.agent")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

API_BASE_URL = "https://space.ai-builders.com/backend/v1"
RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://localhost:8000")
API_KEY = os.getenv("AI_BUILDER_TOKEN")

# 本地检索器配置（优先使用本地直连模式）
INDEX_PATH = os.getenv("INDEX_PATH", "my_history.index")
METADATA_PATH = os.getenv("METADATA_PATH", "chunks_metadata.json")

# 全局检索器实例（延迟初始化）
_local_retriever: Optional[VectorRetriever] = None

# ================= 核心工具函数 =================

def rewrite_query_with_context(query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    基于对话历史重写查询，增强上下文理解
    
    Args:
        query: 原始查询
        conversation_history: 对话历史
        
    Returns:
        优化后的查询
    """
    if not conversation_history or len(conversation_history) == 0:
        return query
    
    # 提取最近对话中的关键信息
    recent_context = []
    for msg in conversation_history[-6:]:  # 最近3轮对话
        content = msg.get("content", "")
        if content:
            # 提取日期、人名、事件等关键信息
            # 这里可以添加更复杂的NLP处理，但为了简单，我们先提取明显的日期和关键词
            recent_context.append(content[:200])  # 保留前200字符
    
    # 如果查询很短或包含代词，尝试从历史中补充信息
    if len(query) < 10 or any(word in query for word in ["它", "那个", "这个", "那天", "那时候", "之前"]):
        context_text = " ".join(recent_context)
        # 简单的启发式：如果历史中有日期，可以补充到查询中
        # 实际应用中可以使用更复杂的NLP模型
        logger.debug(f"🔄 查询重写: 原始='{query}', 上下文长度={len(context_text)}")
    
    return query

def normalize_date(date_str: str, current_date: datetime) -> Optional[str]:
    """
    将相对日期（如 "yesterday", "last_month"）转换为标准格式
    注意：对于 N_days_ago 格式，直接返回原格式，让 retriever._parse_date_filter 处理
    """
    if not date_str or date_str.lower() == "none":
        return None
    
    date_str = date_str.strip()
    date_str_lower = date_str.lower()
    
    # 处理相对时间
    if date_str_lower == "yesterday":
        yesterday = current_date - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    elif date_str_lower == "today":
        return current_date.strftime("%Y-%m-%d")
    elif date_str_lower == "last_week":
        last_week = current_date - timedelta(days=7)
        return last_week.strftime("%Y-%m-%d")
    elif date_str_lower == "last_month":
        if current_date.month == 1:
            last_month = current_date.replace(year=current_date.year - 1, month=12)
        else:
            last_month = current_date.replace(month=current_date.month - 1)
        return last_month.strftime("%Y-%m")
    elif date_str_lower == "last_year":
        return str(current_date.year - 1)
    
    # 处理 "N_days_ago" 格式（如 "2_days_ago"），直接返回让 retriever 处理
    if date_str_lower.endswith("_days_ago"):
        return date_str  # 保持原格式，让 retriever._parse_date_filter 处理
    
    # 处理 "N_months_ago" 格式
    if date_str_lower.endswith("_months_ago"):
        return date_str  # 保持原格式，让 retriever._parse_date_filter 处理
    
    # 如果已经是标准格式，直接返回
    # 支持 YYYY-MM-DD, YYYY-MM, YYYY-MM-下旬 等格式
    return date_str

def _match_date_filter(item_date: Any, filter_date: str, current_date: datetime) -> bool:
    """
    检查 item_date 是否匹配 date_filter
    
    Args:
        item_date: 记录中的日期（可能是字符串、None等）
        filter_date: 过滤条件（如 "2024-04-25", "2024-04", "2024-11-下旬" 等）
        current_date: 当前日期
        
    Returns:
        是否匹配
    """
    if item_date is None:
        # 如果记录日期为None，根据过滤条件决定是否匹配
        # 如果过滤条件很具体（如具体日期），则不匹配
        # 如果过滤条件很宽泛（如年份），则可以匹配
        return False  # 保守策略：None日期不匹配任何过滤条件
    
    if not isinstance(item_date, str):
        return False
    
    item_date_str = item_date.strip()
    if not item_date_str:
        return False
    
    # 处理"下旬"等特殊格式
    if filter_date.endswith('-下旬'):
        # 提取年月
        year_month = filter_date[:-3]  # "2024-11"
        if item_date_str.startswith(year_month):
            # 检查日期是否在21-31日之间
            try:
                if len(item_date_str) >= 10:  # YYYY-MM-DD格式
                    day = int(item_date_str.split('-')[2])
                    return 21 <= day <= 31
            except:
                pass
        return False
    
    # 简单匹配：检查item_date是否以filter_date开头
    return item_date_str.startswith(filter_date)

def _get_local_retriever() -> Optional[VectorRetriever]:
    """
    获取本地检索器实例（延迟初始化）
    
    Returns:
        VectorRetriever 实例，如果初始化失败则返回 None
    """
    global _local_retriever
    
    # 如果已经初始化，直接返回
    if _local_retriever is not None:
        return _local_retriever
    
    # 检查必要的配置
    if not API_KEY:
        logger.warning("⚠️  API_KEY 未设置，无法初始化本地检索器")
        return None
    
    if not os.path.exists(INDEX_PATH):
        logger.warning(f"⚠️  索引文件不存在: {INDEX_PATH}，无法使用本地检索器")
        return None
    
    if not os.path.exists(METADATA_PATH):
        logger.warning(f"⚠️  元数据文件不存在: {METADATA_PATH}，无法使用本地检索器")
        return None
    
    # 初始化本地检索器
    try:
        logger.info(f"🔧 [本地模式] 正在初始化本地检索器...")
        logger.info(f"   索引文件: {INDEX_PATH}")
        logger.info(f"   元数据文件: {METADATA_PATH}")
        _local_retriever = VectorRetriever(INDEX_PATH, METADATA_PATH, API_KEY)
        logger.info(f"✅ [本地模式] 本地检索器初始化成功！索引向量数: {_local_retriever.index.ntotal}")
        return _local_retriever
    except Exception as e:
        logger.exception(f"❌ [本地模式] 本地检索器初始化失败: {e}")
        return None

def call_retriever(query: str, date_filter: Optional[str] = None, max_results: int = 10, expand_query: bool = False) -> str:
    """
    调用检索服务，实施混合检索策略（关键词检索 + 向量检索）
    
    修复实体混淆幻觉问题：通过关键词暴力检索确保稀有人名（如"张三"）能被准确召回。
    
    Args:
        query: 查询文本
        date_filter: 日期过滤条件（可以是相对时间如 "yesterday" 或标准格式）
        max_results: 最大结果数
        expand_query: 是否扩展查询（如果第一次检索结果不理想，可以尝试扩展查询）
        
    Returns:
        格式化后的检索结果字符串
    """
    # 标准化日期格式
    current_date = datetime.now()
    normalized_date = normalize_date(date_filter, current_date) if date_filter else None
    
    # 如果启用查询扩展，尝试添加相关关键词
    search_query = query
    if expand_query:
        # 简单的查询扩展：添加同义词或相关词
        # 实际应用中可以使用更复杂的NLP模型
        query_words = query.split()
        # 可以添加同义词词典或使用embedding相似度扩展
        logger.debug(f"🔍 [查询扩展] 原始查询: '{query}'")
    
    logger.info(f"🔍 [工具执行] 正在检索: Query='{search_query}', Date='{normalized_date}', Expand={expand_query}")
    
    # ========== 优先使用本地直连模式 ==========
    local_retriever = _get_local_retriever()
    if local_retriever is not None:
        try:
            logger.debug("🚀 [本地模式] 使用混合检索策略（关键词 + 向量）")
            
            keyword_results = []
            vector_results = []
            
            # ========== 1. 关键词暴力检索（针对稀有人名/专有名词）==========
            if len(query.strip()) < 20:  # 疑似人名或专有名词
                logger.debug(f"🔑 [关键词检索] 查询长度 {len(query.strip())} < 20，启用关键词暴力检索")
                query_stripped = query.strip()
                query_lower = query_stripped.lower()
                
                # 将查询词拆分为关键词列表（按空格分割）
                query_keywords = [kw.strip() for kw in query_stripped.split() if kw.strip()]
                is_multi_word = len(query_keywords) > 1
                
                # 遍历所有元数据，进行关键词匹配
                if hasattr(local_retriever, 'metadata') and local_retriever.metadata:
                    for item in local_retriever.metadata:
                        content = item.get('content', '')
                        if not content:
                            continue
                        
                        content_lower = content.lower()
                        
                        # 匹配逻辑：
                        # - 单词查询：要求完整匹配（如 "张三"）
                        # - 多词查询：要求包含所有关键词（如 "内心的小孩 名字" 需要包含 "内心的小孩" 和 "名字"）
                        if is_multi_word:
                            # 多词查询：检查是否包含所有关键词
                            if all(kw.lower() in content_lower for kw in query_keywords):
                                match_found = True
                            else:
                                match_found = False
                        else:
                            # 单词查询：完整匹配
                            match_found = query_lower in content_lower
                        
                        if match_found:
                            # 应用日期过滤（如果有）
                            item_date = item.get('date')
                            if normalized_date:
                                if not _match_date_filter(item_date, normalized_date, current_date):
                                    continue
                            
                            # 创建关键词匹配结果
                            keyword_result = item.copy()
                            keyword_result['_source'] = 'keyword_match'  # 标记为关键词匹配
                            keyword_result['distance'] = 0.0  # 关键词匹配距离为0（最高优先级）
                            keyword_results.append(keyword_result)
                    
                    logger.debug(f"🔑 [关键词检索] 找到 {len(keyword_results)} 条关键词匹配结果")
            
            # ========== 2. 向量检索（原有逻辑 + Post-Retrieval Filtering）==========
            try:
                vector_results_raw = local_retriever.search(
                    query=search_query,
                    top_k=max_results,
                    date_filter=normalized_date,
                    current_date=current_date
                )
                
                # ========== Post-Retrieval Filtering: 检索后清洗 ==========
                # 如果查询词很短，视为精准实体查询，强制检查内容是否包含查询词
                query_stripped = query.strip()
                is_precise_entity_query = len(query_stripped) < 15
                
                vector_results_filtered = []
                
                # 将查询词拆分为 token 列表（按空格分割）
                query_tokens = [token.strip() for token in query_stripped.split() if token.strip()]
                
                # 找到最长的 token（核心实体）
                longest_token = max(query_tokens, key=len) if query_tokens else query_stripped
                longest_token_lower = longest_token.lower()
                
                for r in vector_results_raw:
                    # 标记向量检索结果
                    r['_source'] = 'vector_search'
                    
                    # 如果是精准实体查询，进行清洗
                    if is_precise_entity_query:
                        content = r.get('content', '')
                        if not content:
                            logger.warning(f"⚠️  [检索清洗] 丢弃结果 ID={r.get('id')}：内容为空")
                            continue
                        
                        content_lower = content.lower()
                        # 获取 source 字段，确保是字符串类型
                        source_raw = r.get('source')
                        source = str(source_raw).strip() if source_raw is not None else ''
                        
                        # 获取记录 ID
                        record_id = r.get('id', '')
                        
                        # 产品层面：voice 记录是核心数据源，必须能被检索到
                        # 使用更可靠的判断方式：同时检查 ID 前缀和 source 字段
                        # 优先检查 ID 前缀（更可靠），然后检查 source 字段
                        is_voice = (
                            record_id.startswith('voice_') or 
                            record_id == 'test_manual_001' or
                            source == 'voice'
                        )
                        
                        # 调试：检查 voice 记录
                        if is_voice:
                            logger.info(f"🔍 [调试] 检测到 voice 记录: ID={record_id}, source={repr(source)}, is_voice={is_voice}")
                        
                        # 对于 voice 来源的记录，放宽清洗条件
                        # 因为 voice 记录的内容是用户直接说的，可能不包含查询中的某些修饰词
                        # 特别地，对于宽泛查询（如"记录"、"内容"、"最近"），voice 记录应该全部保留
                        if is_voice:
                            # 定义通用查询词列表（这些词出现时，voice 记录不过滤）
                            generic_words = {'记录', '内容', 'voice', '录音', '语音', '备忘', '最近', '什么', '哪些', '有什么'}
                            
                            # 检查查询是否包含通用词，或者查询很短（可能是宽泛查询）
                            is_generic_query = (
                                any(token.lower() in generic_words for token in query_tokens) or 
                                len(query_stripped) <= 6 or
                                '最近' in query_stripped or
                                '什么' in query_stripped
                            )
                            
                            if is_generic_query:
                                # 对于通用/宽泛查询，voice 记录全部保留（不过滤）
                                logger.debug(f"✅ [检索清洗] 保留 voice 记录 ID={r.get('id')}（通用查询不过滤）")
                                pass  # 不进行过滤，直接保留
                            else:
                                # 对于具体查询，检查是否包含任意关键词（至少一个词长度>1）
                                has_any_keyword = any(
                                    token.lower() in content_lower 
                                    for token in query_tokens 
                                    if len(token) > 1
                                )
                                if not has_any_keyword:
                                    logger.warning(f"⚠️  [检索清洗] 丢弃 voice 记录 ID={r.get('id')}, 日期={r.get('date')}：内容不含任何查询关键词 (查询: '{query_stripped}')")
                                    continue
                        else:
                            # 对于其他来源的记录，使用严格的实体匹配：检查最长 token（核心实体）是否在内容中
                            # 示例：
                            # - Query: "张三" -> Longest: "张三" -> 文档无 张三 -> 丢弃 (正确)
                            # - Query: "内心的小孩 名字" -> Longest: "内心的小孩" -> 文档有 "内心的小孩" -> 保留 (修复目标)
                            if longest_token_lower not in content_lower:
                                logger.warning(f"⚠️  [检索清洗] 丢弃结果 ID={r.get('id')}, 日期={r.get('date')}：内容不含核心实体 '{longest_token}' (查询: '{query_stripped}')")
                                continue
                    
                    # 通过清洗，添加到结果列表
                    vector_results_filtered.append(r)
                
                vector_results = vector_results_filtered
                
                if is_precise_entity_query:
                    filtered_count = len(vector_results_raw) - len(vector_results)
                    logger.debug(f"🔍 [向量检索] 原始结果: {len(vector_results_raw)} 条，清洗后: {len(vector_results)} 条，丢弃: {filtered_count} 条")
                else:
                    logger.debug(f"🔍 [向量检索] 找到 {len(vector_results)} 条向量检索结果")
                    
            except Exception as e:
                logger.warning(f"⚠️  [向量检索] 向量检索失败: {e}")
                vector_results = []
            
            # ========== 3. 合并与去重 ==========
            # 关键词结果优先，然后向量结果
            all_results = []
            seen_ids = set()
            
            # 先添加关键词匹配结果（最高优先级）
            for r in keyword_results:
                result_id = r.get('id')
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(r)
            
            # 再添加向量检索结果（去重）
            for r in vector_results:
                result_id = r.get('id')
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(r)
            
            # 限制总数不超过 max_results（默认10，但为了混合检索，我们允许稍多一些）
            final_results = all_results[:max_results]
            
            # ========== 4. Query Relaxation: 查询放松策略 ==========
            # 如果结果为空，且使用了日期过滤，尝试移除日期限制重新检索
            if not final_results and normalized_date and date_filter:
                logger.info(f"🔄 [Query Relaxation] 带日期检索失败，尝试移除日期限制重新检索: query='{query}', 原日期过滤='{date_filter}'")
                # 递归调用自己，但把 date_filter 设为 None
                # 注意：这里传入的是原始 query，而不是 search_query，因为 search_query 可能被 expand_query 修改过
                relaxed_result = call_retriever(query, date_filter=None, max_results=max_results, expand_query=expand_query)
                # 如果放松查询找到了结果，直接返回
                if relaxed_result and "没有找到" not in relaxed_result and "完全没有" not in relaxed_result:
                    logger.info(f"✅ [Query Relaxation] 移除日期限制后找到结果")
                    return relaxed_result
                # 如果放松查询仍然没有结果，继续执行防幻觉兜底
            
            # ========== 5. 防幻觉兜底 ==========
            if not final_results:
                logger.warning(f"⚠️  [防幻觉] 检索结果为空，返回防幻觉兜底消息")
                return "【系统反馈】数据库中**完全没有**找到包含此关键词的记录。请直接告诉用户'没有找到相关记录'，**严禁**提及其他无关人物，**严禁**编造关系。"
            
            # ========== 6. 格式化结果供 LLM 阅读 ==========
            context_lines = ["【系统反馈】已找到以下相关记录：\n"]
            for i, r in enumerate(final_results, 1):
                date_str = r.get('date', '未知日期')
                content = r.get('content', '')
                distance = r.get('distance', 0.0)
                source_type = r.get('_source', 'unknown')
                
                # 截断过长的内容
                if len(content) > 500:
                    content = content[:500] + "..."
                
                # 显示检索来源（关键词匹配或向量检索）
                source_label = "关键词匹配" if source_type == 'keyword_match' else "向量检索"
                context_lines.append(f"--- 记录 {i} [日期: {date_str}, 相似度: {distance:.4f}, 来源: {source_label}] ---\n{content}\n")
            
            logger.info(f"✅ [混合检索] 检索成功，关键词匹配: {len(keyword_results)}, 向量检索: {len(vector_results)}, 最终结果: {len(final_results)}")
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.exception(f"❌ [本地模式] 本地检索失败: {e}")
            logger.info("🔄 [兜底模式] 切换到 HTTP 调用模式")
            # 继续执行 HTTP 调用作为兜底
    
    # ========== 兜底：使用 HTTP 调用 ==========
    url = f"{RETRIEVER_URL}/retrieve"
    payload = {
        "query": search_query,
        "max_results": max_results
    }
    
    if normalized_date:
        payload["date_filter"] = normalized_date
    
    try:
        logger.debug("🌐 [HTTP模式] 使用 HTTP 调用检索服务")
        # 增加超时时间：检索可能需要较长时间（特别是日期过滤时）
        response = requests.post(url, json=payload, timeout=120)  # 增加到120秒
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        # Query Relaxation: 如果结果为空且使用了日期过滤，尝试移除日期限制重新检索
        if not results and normalized_date and date_filter:
            logger.info(f"🔄 [HTTP模式 Query Relaxation] 带日期检索失败，尝试移除日期限制重新检索: query='{query}', 原日期过滤='{date_filter}'")
            # 递归调用自己，但把 date_filter 设为 None
            relaxed_result = call_retriever(query, date_filter=None, max_results=max_results, expand_query=expand_query)
            # 如果放松查询找到了结果，直接返回
            if relaxed_result and "没有找到" not in relaxed_result and "完全没有" not in relaxed_result:
                logger.info(f"✅ [HTTP模式 Query Relaxation] 移除日期限制后找到结果")
                return relaxed_result
            # 如果放松查询仍然没有结果，继续执行防幻觉兜底
        
        # 防幻觉兜底
        if not results:
            return "【系统反馈】数据库中**完全没有**找到包含此关键词的记录。请直接告诉用户'没有找到相关记录'，**严禁**提及其他无关人物，**严禁**编造关系。"
        
        # 格式化结果供 LLM 阅读
        context_lines = ["【系统反馈】已找到以下相关记录：\n"]
        for i, r in enumerate(results, 1):
            date_str = r.get('date', '未知日期')
            content = r.get('content', '')
            distance = r.get('distance', 0.0)
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            context_lines.append(f"--- 记录 {i} [日期: {date_str}, 相似度: {distance:.4f}] ---\n{content}\n")
        
        logger.debug(f"✅ [HTTP模式] 检索成功，找到 {len(results)} 条结果")
        return "\n".join(context_lines)
        
    except requests.exceptions.RequestException as e:
        logger.exception(f"❌ [HTTP模式] 检索服务调用失败: {e}")
        return f"【系统错误】检索服务暂时不可用: {str(e)}"
    except Exception as e:
        logger.exception(f"❌ [HTTP模式] 检索异常: {e}")
        return f"【系统错误】检索过程出现异常: {str(e)}"

def get_system_prompt(conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    生成 System Prompt，教 AI 使用 ReAct 协议
    
    Args:
        conversation_history: 对话历史（可选），用于上下文理解
    """
    current_date = datetime.now()
    current_date_str = current_date.strftime("%Y-%m-%d")
    current_year = current_date.year
    current_month = current_date.month
    
    # 构建对话历史摘要（如果有）
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        # 提取最近几轮对话的关键信息
        recent_messages = conversation_history[-6:]  # 最近3轮对话
        history_summary = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # 提取用户问题中的关键信息
                history_summary.append(f"用户问过: {content[:100]}")
            elif role == "assistant":
                # 提取AI回答中的关键信息
                history_summary.append(f"我回答过: {content[:100]}")
        
        if history_summary:
            history_context = f"""
# 对话历史上下文
以下是最近的对话历史，帮助你理解用户的意图和上下文：
{chr(10).join(history_summary)}

**重要**：当用户使用代词（如"它"、"那个"、"这个"）或省略主语时，要结合对话历史理解用户指的是什么。
例如，如果用户之前问过"2024年6月2日我在做什么？"，然后问"那天的天气怎么样？"，这里的"那天"指的是2024年6月2日。
"""
    
    return f"""# 身份定义
你是我的 Digital Twin 守护者，不是通用的 ChatGPT。你拥有访问我个人记忆库的能力，包括我的日记、笔记、想法和经历。你的使命是帮助我理解自己、回忆过去、洞察模式。

# 当前日期
今天是 {current_date_str}（格式：YYYY-MM-DD）

这是最重要的！你必须始终知道"今天"是哪一天，才能正确理解时间相关的查询：
- "昨天" = {(current_date - timedelta(days=1)).strftime("%Y-%m-%d")}
- "去年" = {current_year - 1}年
- "上个月" = {current_month - 1 if current_month > 1 else 12}月
{history_context}
# 你的思考协议 (ReAct Protocol)

当用户提问时，你必须先进行**思考 (Thought)**，判断是否需要查询记忆库。

## 什么时候需要查询记忆库？

**必须查询的场景：**
1. 用户询问具体日期发生的事情（如"2024年6月2日我在做什么"）
2. 用户询问关于过去的事件、经历、想法、感受
3. 用户使用时间相关的词汇（如"去年"、"上个月"、"之前"、"曾经"、"那天"）
4. 用户询问关于自己的模式、习惯、决策
5. 用户询问"我记得..."、"我写过..."、"我之前..."
6. 用户的问题涉及个人历史、成长、变化
7. 用户询问"叫什么"、"名字"、"命名"等关于名称的问题
8. 用户询问关于特定概念、人物、事物的名称或定义

**不需要查询的场景：**
- 纯粹的知识性问题（如"什么是机器学习"）
- 当前时间的问题（如"现在几点了"）
- 不需要个人记忆的通用问题
- 简单的打招呼（如"你好"）

## 如何发起查询？

如果需要查询记忆库，请**只输出**一行特殊的指令，格式如下：

```
ACTION: SEARCH query="查询内容" date="日期过滤"
```

**多轮检索策略**：
- 如果第一次检索结果不理想（没有找到相关信息或结果太少），可以在回答中说明"让我尝试用不同的关键词再搜索一次"
- 然后再次输出 ACTION 指令，使用不同的查询词或更宽泛的日期范围
- 这样可以提高检索成功率

**参数说明：**
- `query`: 搜索查询文本，要具体明确，包含关键词
  - 对于情绪/状态类问题：包含"抑郁"、"情绪"、"症状"等关键词
  - 对于事件类问题：包含具体的事件、活动、对象
  - 对于名字类问题：包含核心概念和"名字"关键词
  - **重要**：对于"最近有什么记录"、"最近两天"等宽泛查询，query 应该使用通用关键词如"记录"、"内容"，不要使用"日记"等可能不在内容中的词
  - **上下文理解**：如果用户的问题涉及之前对话中提到的人、事、物，要在query中包含这些信息
  - **查询优化**：如果用户使用代词或省略主语，要结合对话历史补充完整信息
- `date`: 日期过滤条件
  - 具体日期：`"2024-11-27"` 或 `"2024-11-下旬"`（表示2024年11月下旬）
  - 相对时间：`"yesterday"`（昨天）、`"last_month"`（上个月）、`"last_year"`（去年）
  - **最近N天**：`"N_days_ago"`（如 `"2_days_ago"` 表示最近2天，即昨天和今天）
  - **最近N个月**：`"N_months_ago"`（如 `"3_months_ago"` 表示最近3个月）
  - **上下文日期**：如果用户说"那天"、"那时候"、"之前提到的日期"，要结合对话历史确定具体日期
  - 不需要日期过滤：`"None"`

**日期格式说明：**
- 具体日期：`YYYY-MM-DD`（如 `"2024-11-27"`）
- 年月：`YYYY-MM`（如 `"2024-11"`）
- 年月+旬：`YYYY-MM-下旬/上旬/中旬`（如 `"2024-11-下旬"` 表示11月21-30日）
- 年份：`YYYY`（如 `"2024"`）
- 最近N天：`N_days_ago`（如 `"2_days_ago"` 表示最近2天）

## 示例

**示例1：询问具体日期**
用户: "2024年11月下旬我经历的抑郁状态有哪些症状？"
AI: ACTION: SEARCH query="抑郁 症状" date="2024-11-下旬"

**示例2：询问名字**
用户: "我给「内心的小孩」起的名字叫什么？"
AI: ACTION: SEARCH query="内心的小孩 名字" date="None"

**示例3：询问过去的事件**
用户: "去年我去过哪里？"
AI: ACTION: SEARCH query="旅行 去过" date="last_year"

**示例4：询问最近几天的记录**
用户: "最近两天有什么记录？"
AI: ACTION: SEARCH query="记录 内容" date="2_days_ago"

**示例4：不需要查询**
用户: "你好"
AI: 你好！我是你的 Digital Twin 守护者。我可以帮你回忆过去、查找日记、分析模式。

## 核心原则

- **⚠️ 绝对禁止编造或猜测！必须严格基于查询结果回答！**
- **⚠️ 如果查询没有返回结果，必须诚实告知"没有找到相关记录"，绝对不要编造日期、事件或内容！**
- **不要假装已经查了**：如果你没有收到【系统反馈】，就说明你还没查，必须先输出 ACTION 指令。
- **必须基于事实**：如果查询返回了结果，要引用具体的日期、事件、感受（如"根据你的日记，2024年11月27日..."）
- **对于"名字"类问题**：要仔细检查所有返回的结果，寻找明确提到名字的地方
- **如果查询结果中没有相关信息**：诚实告知，不要基于推测给出具体答案

记住：你的能力来自记忆库，而不是编造。如果不知道，就发起 SEARCH。"""

# ================= 主对话逻辑 (ReAct Loop) =================

def chat_with_agent(user_message: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    与 Agent 对话（ReAct 模式）
    
    Args:
        user_message: 用户消息
        conversation_history: 对话历史（可选）
        
    Returns:
        Agent 的回复
    """
    if not API_KEY:
        logger.error("❌ 环境变量 AI_BUILDER_TOKEN 未设置")
        return "错误: 环境变量 AI_BUILDER_TOKEN 未设置"
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # 构建消息列表（传入对话历史以增强上下文理解）
    messages = [{"role": "system", "content": get_system_prompt(conversation_history)}]
    
    # 注意：不在messages中重复添加conversation_history，因为system prompt已经包含了历史摘要
    # 这样可以避免token浪费，同时保持上下文理解能力
    
    messages.append({"role": "user", "content": user_message})
    
    logger.info(f"👤 用户: {user_message}")

    # --- 第一轮：思考与决策 (Reasoning) ---
    try:
        response = client.chat.completions.create(
            model="supermind-agent-v1",
            messages=messages,
            temperature=0.1  # 降低温度，让指令更精准
        )
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"🤖 AI (思考): {ai_response[:200]}...")
        
    except Exception as e:
        logger.exception(f"❌ 模型调用失败: {e}")
        return f"模型调用失败: {e}"

    # --- 第二轮：行动与执行 (Acting) ---
    # 检测 AI 是否输出了 ACTION 指令
    # 支持多种格式：ACTION: SEARCH query="..." date="..."
    action_patterns = [
        r'ACTION:\s*SEARCH\s+query="([^"]+)"\s+date="([^"]+)"',
        r'ACTION:\s*SEARCH\s+query=([^\s]+)\s+date=([^\s]+)',
        r'ACTION:\s*SEARCH\s+query="([^"]+)"',  # 没有 date 参数
    ]
    
    action_match = None
    for pattern in action_patterns:
        action_match = re.search(pattern, ai_response, re.IGNORECASE)
        if action_match:
            break
    
    if action_match:
        # 1. 解析指令
        query = action_match.group(1)
        date_param = action_match.group(2) if len(action_match.groups()) > 1 else "None"
        
        # 2. 基于对话历史优化查询（可选）
        optimized_query = rewrite_query_with_context(query, conversation_history)
        if optimized_query != query:
            logger.info(f"🔄 [ReAct] 查询已优化: '{query}' -> '{optimized_query}'")
            query = optimized_query
        
        logger.info(f"🔧 [ReAct] 检测到 ACTION 指令: query='{query}', date='{date_param}'")
        
        # 3. 执行工具
        search_result = call_retriever(query, date_param, max_results=10)
        logger.info(f"📚 [ReAct] 检索结果长度: {len(search_result)} 字符")
        
        # 3. 将结果作为"观察 (Observation)"反馈给 AI
        messages.append({"role": "assistant", "content": ai_response})
        messages.append({
            "role": "user",
            "content": f"""【查询结果已返回】

{search_result}

请根据以上查询结果，回答我的原始问题。记住：
- 必须基于查询结果中的实际内容回答
- 如果结果中没有相关信息，诚实告知"没有找到相关记录"
- 不要编造或猜测任何内容
- 如果找到了相关记录，要引用具体的日期和内容"""
        })
        
        # 4. 让 AI 根据资料生成最终回答
        try:
            final_response = client.chat.completions.create(
                model="supermind-agent-v1",
                messages=messages,
                temperature=0.7
            )
            final_answer = final_response.choices[0].message.content.strip()
            logger.info(f"✅ [ReAct] 最终回答生成成功，长度: {len(final_answer)} 字符")
            return final_answer
            
        except Exception as e:
            logger.exception(f"❌ 生成最终回答失败: {e}")
            return f"生成最终回答失败: {e}"
            
    else:
        # AI 决定不查库，直接回答
        logger.info(f"✅ [ReAct] AI 决定直接回答（无需查询）")
        return ai_response

def main():
    """主函数：交互式对话"""
    print("=" * 60)
    print("🤖 Digital Twin 守护者 (ReAct 模式)")
    print("=" * 60)
    print("\n提示：")
    print("- 输入 'quit' 或 'exit' 退出")
    print("- 输入 'clear' 清空对话历史")
    print("- 询问关于你的过去、经历、想法时，我会自动查阅记忆库")
    print("=" * 60)
    print()
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("\n👋 再见！")
                break
            
            if user_input.lower() == "clear":
                conversation_history = []
                print("✅ 对话历史已清空\n")
                continue
            
            # 调用 Agent
            print("\n🤖 正在思考...")
            response = chat_with_agent(user_input, conversation_history)
            print(f"\nDigital Twin: {response}\n")
            
            # 更新对话历史
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # 限制历史长度（保留最近10轮对话）
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            logger.exception(f"❌ 对话异常: {e}")
            print()

if __name__ == "__main__":
    main()
