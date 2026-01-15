#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量索引构建脚本
只对新记录建立索引，追加到现有索引中
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import faiss
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置
# 使用脚本所在目录作为 vector_indexer 目录
VECTOR_INDEXER_DIR = Path(__file__).parent
ALL_CHUNKS_FILE = VECTOR_INDEXER_DIR / "all_chunks.json"
INDEX_FILE = VECTOR_INDEXER_DIR / "my_history.index"
METADATA_FILE = VECTOR_INDEXER_DIR / "chunks_metadata.json"
INDEXED_IDS_FILE = VECTOR_INDEXER_DIR / ".indexed_ids.json"  # 记录已索引的ID列表

API_BASE_URL = "https://space.ai-builders.com/backend/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = os.getenv("AI_BUILDER_TOKEN")

if not API_KEY:
    print("❌ 错误: AI_BUILDER_TOKEN 环境变量未设置")
    sys.exit(1)

def get_embedding(text: str) -> List[float]:
    """获取文本的向量表示"""
    url = f"{API_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": [text],
        "model": EMBEDDING_MODEL
    }
    
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    try:
        response = session.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ 获取向量失败: {e}")
        raise

def load_indexed_ids() -> set:
    """加载已索引的记录ID列表"""
    if INDEXED_IDS_FILE.exists():
        try:
            with open(INDEXED_IDS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"⚠️  读取已索引ID列表失败: {e}，将重新索引")
            return set()
    return set()

def save_indexed_ids(indexed_ids: set):
    """保存已索引的记录ID列表"""
    with open(INDEXED_IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(indexed_ids), f, ensure_ascii=False, indent=2)

def incremental_index():
    """增量索引：只处理新记录"""
    
    # 1. 加载数据
    if not ALL_CHUNKS_FILE.exists():
        print(f"❌ 文件不存在: {ALL_CHUNKS_FILE}")
        return False
    
    with open(ALL_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    indexed_ids = load_indexed_ids()
    
    # 2. 找出新记录
    new_chunks = [chunk for chunk in all_chunks if chunk.get('id') not in indexed_ids]
    
    if not new_chunks:
        print("✅ 没有新记录需要索引")
        return True
    
    print(f"📊 发现 {len(new_chunks)} 条新记录需要索引（总共 {len(all_chunks)} 条）")
    
    # 3. 加载现有索引和元数据
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            index = faiss.read_index(str(INDEX_FILE))
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ 加载现有索引: {index.ntotal} 条向量")
        except Exception as e:
            print(f"⚠️  加载现有索引失败: {e}，将创建新索引")
            index = None
            metadata = []
    else:
        print("📝 索引文件不存在，将创建新索引")
        index = None
        metadata = []
    
    # 4. 为新记录生成向量并追加
    new_vectors = []
    new_metadata = []
    
    print(f"🔄 开始处理 {len(new_chunks)} 条新记录...")
    for i, chunk in enumerate(new_chunks, 1):
        try:
            content = chunk.get('content', '')
            if not content:
                print(f"⚠️  跳过空内容记录: {chunk.get('id')}")
                continue
            
            # 生成向量
            print(f"  [{i}/{len(new_chunks)}] 处理: {chunk.get('id')}...", end=' ', flush=True)
            embedding = get_embedding(content)
            new_vectors.append(embedding)
            new_metadata.append(chunk)
            indexed_ids.add(chunk.get('id'))
            print("✓")
            
            # 避免API限流
            if i % 10 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"✗ 错误: {e}")
            continue
    
    if not new_vectors:
        print("⚠️  没有成功生成任何向量")
        return False
    
    # 5. 追加到索引
    if index is None:
        # 创建新索引
        dimension = len(new_vectors[0])
        index = faiss.IndexFlatL2(dimension)
        print(f"📝 创建新索引，维度: {dimension}")
    
    # 转换为numpy数组并追加
    vectors_array = np.array(new_vectors, dtype=np.float32)
    index.add(vectors_array)
    
    # 更新元数据
    metadata.extend(new_metadata)
    
    # 6. 保存索引和元数据
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 保存已索引ID列表
    save_indexed_ids(indexed_ids)
    
    print(f"✅ 增量索引完成！")
    print(f"   - 新增: {len(new_vectors)} 条")
    print(f"   - 总计: {index.ntotal} 条向量")
    print(f"   - 元数据: {len(metadata)} 条")
    
    return True

if __name__ == "__main__":
    try:
        success = incremental_index()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
