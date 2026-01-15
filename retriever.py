#!/usr/bin/env python3
"""
向量检索器
封装 FAISS 索引加载和相似度搜索功能
"""

import json
import os
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import faiss
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置常量
API_BASE_URL = "https://space.ai-builders.com/backend"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
EMBEDDING_CACHE_SIZE = 1000  # LRU 缓存大小

# 创建日志器
logger = logging.getLogger("vector_indexer.retriever")


class EmbeddingClient:
    """封装embeddings API调用的客户端，带缓存功能"""
    
    def __init__(self, api_key: str, base_url: str = API_BASE_URL, cache_size: int = EMBEDDING_CACHE_SIZE):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()
        self.cache_size = cache_size
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _create_session(self):
        """创建带重试机制的requests session"""
        session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        key_string = f"{model}:{text}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """
        获取单个文本的embedding（带缓存）
        
        Args:
            text: 输入文本
            model: 模型名称
            
        Returns:
            embedding向量
        """
        cache_key = self._get_cache_key(text, model)
        
        # 检查缓存
        if cache_key in self._cache:
            self._cache_hits += 1
            logger.debug(f"💾 缓存命中: text_length={len(text)}")
            return self._cache[cache_key]
        
        # 缓存未命中，调用 API
        self._cache_misses += 1
        url = f"{self.base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": [text],
            "model": model
        }
        
        try:
            logger.debug(f"🔗 调用 Embedding API: model={model}, text_length={len(text)}")
            response = self.session.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            logger.debug(f"✅ Embedding 生成成功，维度: {len(embedding)}")
            
            # 存入缓存（如果缓存已满，删除最旧的项）
            if len(self._cache) >= self.cache_size:
                # 删除第一个（最旧的）项
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = embedding
            return embedding
        
        except requests.exceptions.RequestException as e:
            logger.exception(f"❌ Embedding API 调用失败: {e}")
            raise Exception(f"API调用失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 2)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("🗑️  Embedding 缓存已清空")


class VectorRetriever:
    """向量检索器类"""
    
    def __init__(self, index_path: str, metadata_path: str, api_key: str, enable_cache: bool = True):
        """
        初始化检索器
        
        Args:
            index_path: FAISS索引文件路径
            metadata_path: 元数据JSON文件路径
            api_key: AI Builder API密钥
            enable_cache: 是否启用缓存（默认启用）
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.enable_cache = enable_cache
        self.embedding_client = EmbeddingClient(api_key)
        self._load_index_and_metadata()
    
    def _load_index_and_metadata(self):
        """加载FAISS索引和元数据"""
        logger.info(f"📖 正在加载索引: {self.index_path}")
        
        if not os.path.exists(self.index_path):
            logger.error(f"❌ 索引文件不存在: {self.index_path}")
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        
        if not os.path.exists(self.metadata_path):
            logger.error(f"❌ 元数据文件不存在: {self.metadata_path}")
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
        
        # 加载FAISS索引
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"✅ 索引加载完成，向量数量: {self.index.ntotal}, 维度: {self.index.d}")
        except Exception as e:
            logger.exception(f"❌ 索引加载失败: {e}")
            raise
        
        # 加载元数据
        logger.info(f"📖 正在加载元数据: {self.metadata_path}")
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ 元数据加载完成，记录数量: {len(self.metadata)}")
        except Exception as e:
            logger.exception(f"❌ 元数据加载失败: {e}")
            raise
        
        # 验证索引和元数据数量是否一致
        if self.index.ntotal != len(self.metadata):
            error_msg = f"索引向量数量 ({self.index.ntotal}) 与元数据数量 ({len(self.metadata)}) 不匹配"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    def _parse_date_filter(self, date_filter: Optional[str], current_date: datetime) -> Optional[Tuple[datetime, datetime]]:
        """
        解析日期过滤条件
        
        Args:
            date_filter: 日期过滤字符串，支持：
                - "YYYY-MM-DD" 格式的具体日期
                - "YYYY-MM" 格式的年月（如 "2024-03"）
                - "YYYY" 格式的年份（如 "2024"）
                - "last_year", "last_month", "last_week" 等相对时间
                - "N_months_ago" 格式（如 "3_months_ago"）
                - "N_days_ago" 格式（如 "30_days_ago"）
            current_date: 当前日期
            
        Returns:
            (start_date, end_date) 元组，如果为None则表示不过滤
        """
        if not date_filter:
            return None
        
        date_filter = date_filter.strip().lower()
        
        # 处理相对时间
        if date_filter == "last_year":
            start_date = current_date.replace(year=current_date.year - 1, month=1, day=1)
            end_date = current_date.replace(year=current_date.year - 1, month=12, day=31)
            return (start_date, end_date)
        
        elif date_filter == "last_month":
            if current_date.month == 1:
                start_date = current_date.replace(year=current_date.year - 1, month=12, day=1)
            else:
                start_date = current_date.replace(month=current_date.month - 1, day=1)
            
            # 计算上个月的最后一天
            if current_date.month == 1:
                end_date = current_date.replace(year=current_date.year - 1, month=12, day=31)
            else:
                # 获取上个月的最后一天
                first_day_this_month = current_date.replace(day=1)
                last_day_last_month = first_day_this_month - timedelta(days=1)
                end_date = last_day_last_month
            
            return (start_date, end_date)
        
        elif date_filter == "last_week":
            end_date = current_date - timedelta(days=current_date.weekday() + 1)  # 上周日
            start_date = end_date - timedelta(days=6)  # 上周一
            return (start_date, end_date)
        
        # 处理 "N_months_ago" 格式（如 "3_months_ago"）
        if date_filter.endswith("_months_ago"):
            try:
                months = int(date_filter.replace("_months_ago", ""))
                # 计算 N 个月前的日期范围
                end_date = current_date - timedelta(days=1)  # 昨天
                # 计算 N 个月前的日期
                start_date = current_date
                for _ in range(months):
                    if start_date.month == 1:
                        start_date = start_date.replace(year=start_date.year - 1, month=12)
                    else:
                        start_date = start_date.replace(month=start_date.month - 1)
                start_date = start_date.replace(day=1)  # 月初
                return (start_date, end_date)
            except ValueError:
                pass
        
        # 处理 "N_days_ago" 格式（如 "30_days_ago"）
        # 注意："N_days_ago" 表示"最近N天"，包括今天
        # 例如："2_days_ago" 表示最近2天，即昨天和今天
        if date_filter.endswith("_days_ago"):
            try:
                days = int(date_filter.replace("_days_ago", ""))
                end_date = current_date  # 包括今天
                start_date = current_date - timedelta(days=days - 1)  # 从 N-1 天前开始（包括今天）
                return (start_date, end_date)
            except ValueError:
                pass
        
        # 处理 "YYYY-MM-DD" 格式的具体日期
        try:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d")
            return (filter_date, filter_date)
        except ValueError:
            pass
        
        # 处理 "YYYY-MM" 格式的年月（如 "2024-03"）
        try:
            parts = date_filter.split("-")
            if len(parts) == 2:
                year, month = parts
                year = int(year)
                month = int(month)
                if 1 <= month <= 12:
                    start_date = datetime(year, month, 1)
                    # 计算该月的最后一天
                    if month == 12:
                        end_date = datetime(year, 12, 31)
                    else:
                        next_month = datetime(year, month + 1, 1)
                        end_date = next_month - timedelta(days=1)
                    return (start_date, end_date)
        except (ValueError, AttributeError):
            pass
        
        # 处理 "YYYY-MM-下旬" 或 "YYYY-MM-上旬" 或 "YYYY-MM-中旬" 格式
        # 注意：这里 date_filter 可能是 "2024-11-下旬" 这样的格式
        if "下旬" in date_filter or "上旬" in date_filter or "中旬" in date_filter:
            try:
                # 提取年月
                parts = date_filter.replace("下旬", "").replace("上旬", "").replace("中旬", "").split("-")
                if len(parts) >= 2:
                    year = int(parts[0])
                    month = int(parts[1])
                    if 1 <= month <= 12:
                        if "上旬" in date_filter:
                            # 上旬：1-10日
                            start_date = datetime(year, month, 1)
                            end_date = datetime(year, month, 10)
                        elif "中旬" in date_filter:
                            # 中旬：11-20日
                            start_date = datetime(year, month, 11)
                            end_date = datetime(year, month, 20)
                        elif "下旬" in date_filter:
                            # 下旬：21日-月末
                            start_date = datetime(year, month, 21)
                            # 计算该月的最后一天
                            if month == 12:
                                end_date = datetime(year, 12, 31)
                            else:
                                next_month = datetime(year, month + 1, 1)
                                end_date = next_month - timedelta(days=1)
                        return (start_date, end_date)
            except (ValueError, AttributeError):
                pass
        
        # 处理 "YYYY" 格式的年份（如 "2024"）
        try:
            year = int(date_filter)
            if 1900 <= year <= 2100:  # 合理的年份范围
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
                return (start_date, end_date)
        except ValueError:
            pass
        
        # 如果无法解析，返回None（不过滤）
        logger.warning(f"⚠️  无法解析日期过滤条件: {date_filter}，将忽略日期过滤")
        return None
    
    def _filter_by_date(self, indices: List[int], date_range: Optional[Tuple[datetime, datetime]]) -> List[int]:
        """
        根据日期范围过滤结果
        
        Args:
            indices: 原始索引列表
            date_range: (start_date, end_date) 元组
            
        Returns:
            过滤后的索引列表
        """
        if not date_range:
            return indices
        
        start_date, end_date = date_range
        filtered_indices = []
        
        for idx in indices:
            chunk = self.metadata[idx]
            chunk_date = chunk.get('date')
            
            # 如果chunk没有日期，跳过（不包含在结果中）
            if not chunk_date:
                continue
            
            try:
                # 解析日期字符串
                if isinstance(chunk_date, str):
                    chunk_dt = datetime.strptime(chunk_date, "%Y-%m-%d")
                else:
                    continue
                
                # 检查是否在日期范围内
                if start_date <= chunk_dt <= end_date:
                    filtered_indices.append(idx)
            
            except (ValueError, TypeError):
                # 日期格式错误，跳过
                continue
        
        return filtered_indices
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        date_filter: Optional[str] = None,
        current_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似内容
        
        Args:
            query: 查询文本
            top_k: 返回前K个结果
            date_filter: 可选的日期过滤条件
            current_date: 当前日期（用于解析相对时间），如果为None则使用当前时间
            
        Returns:
            搜索结果列表，每个元素包含：
                - id: chunk ID
                - source: 来源
                - date: 日期
                - content: 内容
                - distance: 距离（越小越相似）
        """
        if current_date is None:
            current_date = datetime.now()
        
        logger.debug(f"🔍 开始搜索: query='{query[:50]}...', top_k={top_k}, date_filter={date_filter}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 生成查询文本的embedding
        try:
            embedding_start = time.time()
            query_embedding = self.embedding_client.get_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            embedding_time = time.time() - embedding_start
            logger.debug(f"✅ 查询向量生成成功，维度: {len(query_embedding)}，耗时: {embedding_time:.2f}秒")
        except Exception as e:
            logger.exception(f"❌ 生成查询向量失败: {e}")
            raise
        
        # 2. 搜索策略
        # 如果设置了日期过滤，需要搜索更多候选结果（因为过滤后可能结果不足）
        if date_filter:
            # 日期过滤时，搜索更多结果以确保有足够的候选
            # 对于具体日期（如 "2024-06-02"），搜索所有结果然后过滤
            # 对于相对时间（如 "last_year"）或日期范围（如 "2024-11-下旬"），搜索更多结果
            date_filter_clean = date_filter.strip()
            if len(date_filter_clean) == 10 and date_filter_clean.count('-') == 2 and date_filter_clean.count('下旬') == 0:
                # 具体日期格式 YYYY-MM-DD（不含"下旬"等）
                # 优化：不需要搜索所有结果，搜索合理数量即可（top_k * 200 应该足够）
                # 这样可以避免在大索引上搜索所有结果导致的性能问题
                search_k = min(top_k * 200, self.index.ntotal)
                logger.debug(f"📅 具体日期查询，搜索 {search_k} 条候选结果（索引总数: {self.index.ntotal}）")
            elif "下旬" in date_filter_clean or "上旬" in date_filter_clean or "中旬" in date_filter_clean:
                # 日期范围（如 "2024-11-下旬"），搜索更多结果以确保覆盖整个范围
                search_k = min(top_k * 100, self.index.ntotal)  # 增加搜索数量
            else:
                # 相对时间，搜索更多结果
                search_k = min(top_k * 50, self.index.ntotal)
        else:
            search_k = top_k
        
        # 2. 执行FAISS搜索
        search_start = time.time()
        distances, indices = self.index.search(query_vector, search_k)
        search_time = time.time() - search_start
        logger.debug(f"🔍 FAISS搜索完成，搜索了 {search_k} 条，耗时: {search_time:.2f}秒")
        
        # 3. 应用日期过滤（如果有）
        if date_filter:
            try:
                date_range = self._parse_date_filter(date_filter, current_date)
                logger.debug(f"📅 日期过滤: {date_filter} -> {date_range}")
                if date_range is None:
                    # 日期解析失败，记录警告但继续搜索（不过滤）
                    logger.warning(f"⚠️  无法解析日期过滤条件 '{date_filter}'，将忽略日期过滤")
                    result_indices = indices[0].tolist()[:top_k]
                else:
                    filtered_indices = self._filter_by_date(indices[0].tolist(), date_range)
                    logger.debug(f"📊 日期过滤结果: 原始 {len(indices[0])} 条 -> 过滤后 {len(filtered_indices)} 条")
                    # 返回过滤后的结果（按相似度排序）
                    result_indices = filtered_indices[:top_k]
            except Exception as e:
                # 日期解析异常，记录错误但继续搜索（不过滤）
                logger.exception(f"⚠️  日期过滤处理异常: {e}，将忽略日期过滤")
                result_indices = indices[0].tolist()[:top_k]
        else:
            result_indices = indices[0].tolist()[:top_k]
        
        # 4. 构建结果并去重
        seen_ids = set()  # 用于去重
        results = []
        
        # 创建索引到距离的映射，提高查找效率并避免异常
        idx_to_distance = {}
        for i, idx in enumerate(indices[0].tolist()):
            idx_to_distance[idx] = float(distances[0][i])
        
        for idx in result_indices:
            try:
                chunk = self.metadata[idx]
                chunk_id = chunk.get("id")
                
                # 去重：如果已经见过这个 ID，跳过
                if chunk_id and chunk_id in seen_ids:
                    continue
                
                seen_ids.add(chunk_id)
                
                # 找到对应的距离（使用映射，避免 index() 可能抛出的异常）
                distance = idx_to_distance.get(idx, 1.0)  # 如果找不到，使用默认值1.0
                
                results.append({
                    "id": chunk_id,
                    "source": chunk.get("source"),
                    "date": chunk.get("date"),
                    "content": chunk.get("content"),
                    "distance": distance
                })
            except (IndexError, KeyError, TypeError) as e:
                # 如果索引无效或元数据有问题，跳过这条记录
                logger.warning(f"⚠️  跳过无效索引 {idx}: {e}")
                continue
        
        # 5. 按距离排序（确保结果按相似度从高到低）
        results.sort(key=lambda x: x["distance"])
        
        # 如果去重后结果不足，尝试补充
        if len(results) < top_k and len(result_indices) < len(indices[0]):
            # 从剩余的候选中补充
            remaining_indices = [idx for idx in indices[0].tolist() if idx not in result_indices]
            for idx in remaining_indices[:top_k - len(results)]:
                chunk = self.metadata[idx]
                chunk_id = chunk.get("id")
                
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    original_idx_in_results = indices[0].tolist().index(idx)
                    distance = float(distances[0][original_idx_in_results])
                    
                    results.append({
                        "id": chunk_id,
                        "source": chunk.get("source"),
                        "date": chunk.get("date"),
                        "content": chunk.get("content"),
                        "distance": distance
                    })
            
            # 再次排序
            results.sort(key=lambda x: x["distance"])
        
        total_time = time.time() - start_time
        logger.debug(f"📊 最终结果: {len(results)} 条（去重后），总耗时: {total_time:.2f}秒")
        return results[:top_k]  # 确保不超过 top_k
