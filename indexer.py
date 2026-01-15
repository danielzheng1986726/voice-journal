#!/usr/bin/env python3
"""
向量索引构建脚本
使用 Smart Chunking 策略解决 "Needle in a Haystack" 问题
从 all_chunks.json 构建 FAISS 向量索引
"""

import json
import os
import time
import sys
from typing import List, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import faiss
import numpy as np
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 导入 langchain 文本切分器
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    try:
        # 兼容旧版本 langchain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False
        print("❌ 错误: langchain 文本切分器未安装")
        print("   请安装: pip install langchain-text-splitters")
        print("   或: pip install langchain")
        sys.exit(1)

# ==================== 配置常量 ====================

# API 配置
API_BASE_URL = "https://space.ai-builders.com/backend/v1"  # 注意：包含 /v1
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY_ENV = "AI_BUILDER_TOKEN"

# 批次处理配置
BATCH_SIZE = 20  # 每批处理的 chunk 数量
DELAY_BETWEEN_BATCHES = 1.0  # 批次间延时（秒）
MAX_RETRIES = 3  # API 调用重试次数

# Smart Chunking 配置（黄金区间）
CHUNK_SIZE = 600  # 字符数，让语义更集中，突出细节
CHUNK_OVERLAP = 100  # 字符数，保留上下文，防止句子被切断
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 优先按段落和句子切分

# 默认文件路径（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_FILE = os.path.join(SCRIPT_DIR, "all_chunks.json")
DEFAULT_INDEX_FILE = "my_history.index"
DEFAULT_METADATA_FILE = "chunks_metadata.json"

# ==================== 核心类 ====================

class EmbeddingClient:
    """封装 Embeddings API 调用的客户端，带重试机制"""
    
    def __init__(self, api_key: str, base_url: str = API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()
    
    def _create_session(self):
        """创建带重试机制的 requests session"""
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
    
    def get_embeddings(self, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """
        批量获取 embeddings
        
        Args:
            texts: 文本列表
            model: 模型名称
            
        Returns:
            embeddings 列表，每个元素是一个向量
        """
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": model
        }
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            # 按 index 排序确保顺序与输入一致
            sorted_data = sorted(result["data"], key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in sorted_data]
            return embeddings
        
        except requests.exceptions.RequestException as e:
            print(f"\n❌ API 调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"响应内容: {e.response.text}")
                except:
                    pass
            raise

# ==================== Smart Chunking 处理 ====================

def split_chunk_with_text_splitter(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    使用 RecursiveCharacterTextSplitter 将 chunk 切分为多个小 chunk
    
    实施 Smart Chunking 策略，解决 "Needle in a Haystack" 问题：
    - 使用 langchain 的 RecursiveCharacterTextSplitter 进行智能切分
    - 每个小切片继承父文档的所有元数据（source, date, 其他字段）
    - 处理日期为 null 的情况（保留 null 值在元数据中）
    - 生成唯一的 ID: {parent_id}_part_{index}
    
    Args:
        chunk: 原始chunk字典，包含 'id', 'content', 'source', 'date' 等字段
        
    Returns:
        切分后的chunk列表，每个chunk都有唯一的ID和继承的元数据
    """
    content = chunk.get('content', '')
    if not content or not content.strip():
        return []
    
    parent_id = chunk.get('id', 'unknown')
    parent_source = chunk.get('source')
    parent_date = chunk.get('date')  # 可能是 null
    
    # 使用 RecursiveCharacterTextSplitter 进行智能切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # 600 字符
        chunk_overlap=CHUNK_OVERLAP,      # 100 字符
        separators=CHUNK_SEPARATORS,      # 智能分隔符
        length_function=len
    )
    
    try:
        sub_chunks = splitter.split_text(content)
    except Exception as e:
        print(f"   ⚠️  切分失败 (ID: {parent_id}): {e}")
        return [chunk]  # 如果切分失败，返回原chunk
    
    # 如果只有一个chunk（不需要切分），直接返回原chunk
    if len(sub_chunks) <= 1:
        return [chunk]
    
    # 为每个 sub_chunk 创建新的 chunk 对象
    split_chunks = []
    for index, sub_content in enumerate(sub_chunks):
        if not sub_content.strip():
            continue
        
        # 创建新的 chunk，完整继承父文档的元数据
        new_chunk = {
            'id': f"{parent_id}_part_{index}",      # 唯一ID格式: {parent_id}_part_{index}
            'content': sub_content.strip(),
            'source': parent_source,                # 继承 source
            'date': parent_date,                     # 继承 date（可以是 null）
        }
        
        # 保留其他元数据（如果有）
        for key, value in chunk.items():
            if key not in ['id', 'content', 'source', 'date']:
                new_chunk[key] = value
        
        # 添加切分信息（用于追踪和调试）
        new_chunk['_original_id'] = parent_id       # 父文档ID
        new_chunk['_split_index'] = index           # 切分索引（从0开始）
        new_chunk['_total_splits'] = len(sub_chunks)  # 总切分数
        
        split_chunks.append(new_chunk)
    
    return split_chunks


def load_and_split_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 chunks JSON 文件并应用 Smart Chunking 策略
    
    对所有文档进行智能小切片处理，解决 "Needle in a Haystack" 问题。
    通过将大文档切分为600字符的小切片，确保像"小东东"这样的关键细节不会被稀释。
    
    处理逻辑：
    1. 加载原始数据（JSON格式）
    2. 过滤掉 content 为空的记录（但保留日期为 null 的记录）
    3. 对每个文档的 content 使用 RecursiveCharacterTextSplitter 进行 Smart Chunking
    4. 为每个 sub_chunk 创建新文档，继承所有元数据（source, date, 其他字段）
    5. 处理日期为 null 的情况（在元数据中保留 null 值）
    6. 为每个 sub_chunk 生成唯一 ID: {parent_id}_part_{index}
    
    Args:
        file_path: 输入的 JSON 文件路径
        
    Returns:
        切分后的 chunk 列表，每个 chunk 都有唯一的 ID 和完整的元数据
    """
    print(f"📖 正在读取文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 过滤掉content为空的记录（但保留日期为null的记录）
    valid_chunks = [chunk for chunk in chunks if chunk.get('content') and chunk['content'].strip()]
    skipped = len(chunks) - len(valid_chunks)
    
    if skipped > 0:
        print(f"⚠️  跳过了 {skipped} 条 content 为空的记录")
    
    print(f"✅ 成功加载 {len(valid_chunks)} 条有效记录")
    
    # 统计日期为null的记录数
    null_date_count = sum(1 for chunk in valid_chunks if chunk.get('date') is None)
    if null_date_count > 0:
        print(f"📅 发现 {null_date_count} 条记录的日期为 null（将保留在元数据中）")
    
    # 应用 Smart Chunking 策略
    print(f"\n🔪 应用 Smart Chunking 策略...")
    print(f"   配置: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print(f"   分隔符: {CHUNK_SEPARATORS}")
    print(f"   使用: RecursiveCharacterTextSplitter (langchain)")
    
    processed_chunks = []
    total_original = len(valid_chunks)
    
    for i, chunk in enumerate(valid_chunks):
        # 显示进度
        if (i + 1) % 100 == 0 or i == 0:
            progress = ((i + 1) * 100 // total_original) if total_original > 0 else 0
            print(f"   处理进度: {i+1}/{total_original} ({progress}%)")
        
        # 使用文本切分器切分
        split_chunks = split_chunk_with_text_splitter(chunk)
        processed_chunks.extend(split_chunks)
    
    print(f"\n✅ Smart Chunking 完成！")
    print(f"   原始记录数: {total_original}")
    print(f"   切分后记录数: {len(processed_chunks)}")
    if total_original > 0:
        avg_splits = len(processed_chunks) / total_original
        print(f"   平均每个文档切分为: {avg_splits:.2f} 个chunk")
    
    # 验证：确保所有chunk都有唯一ID
    ids = [chunk.get('id') for chunk in processed_chunks]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        print(f"⚠️  警告: 发现重复ID！总ID数: {len(ids)}, 唯一ID数: {len(unique_ids)}")
    else:
        print(f"✅ 所有 {len(unique_ids)} 个chunk都有唯一ID")
    
    return processed_chunks

# ==================== 向量生成与索引构建 ====================

def process_batches(chunks: List[Dict[str, Any]], client: EmbeddingClient) -> tuple:
    """
    批量处理chunks，生成embeddings
    
    Args:
        chunks: 切分后的chunk列表
        client: EmbeddingClient 实例
        
    Returns:
        (embeddings列表, metadata列表) - 顺序一一对应
    """
    all_embeddings = []
    all_metadata = []
    
    total_chunks = len(chunks)
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🚀 开始生成 Embeddings...")
    print(f"   总chunk数: {total_chunks}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   总批次数: {total_batches}")
    print(f"   批次间延时: {DELAY_BETWEEN_BATCHES}秒\n")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        batch_texts = [chunk['content'] for chunk in batch_chunks]
        
        try:
            print(f"📦 批次 {batch_idx + 1}/{total_batches}: 处理第 {start_idx + 1}-{end_idx} 条 ({len(batch_chunks)}条)...", end=" ")
            
            # 调用 API 生成 embeddings
            batch_embeddings = client.get_embeddings(batch_texts)
            
            # 验证返回的embeddings数量
            if len(batch_embeddings) != len(batch_chunks):
                raise ValueError(f"返回的embeddings数量 ({len(batch_embeddings)}) 与chunk数量 ({len(batch_chunks)}) 不匹配")
            
            # 添加到总列表
            all_embeddings.extend(batch_embeddings)
            all_metadata.extend(batch_chunks)
            
            print("✅")
            
            # 批次间延时（最后一批不需要延时）
            if batch_idx < total_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        except Exception as e:
            print(f"\n❌ 批次 {batch_idx + 1} 处理失败: {e}")
            print(f"   跳过该批次，继续处理...")
            # 可以选择继续或退出
            # 这里选择继续，但会记录错误
    
    print(f"\n✅ Embeddings 生成完成！")
    print(f"   成功处理: {len(all_embeddings)} 条记录")
    
    if len(all_embeddings) != total_chunks:
        print(f"⚠️  警告: 成功处理的记录数 ({len(all_embeddings)}) 与总chunk数 ({total_chunks}) 不一致")
    
    return all_embeddings, all_metadata


def build_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """
    构建FAISS索引
    
    Args:
        embeddings: embeddings列表
        
    Returns:
        FAISS索引对象
    """
    print(f"\n🔨 正在构建FAISS索引...")
    
    # 转换为numpy数组
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    vector_count = len(embeddings)
    
    print(f"   向量维度: {dimension}")
    print(f"   向量数量: {vector_count}")
    
    # 创建IndexFlatL2索引（L2距离）
    index = faiss.IndexFlatL2(dimension)
    
    # 添加向量到索引
    index.add(embeddings_array)
    
    print(f"✅ 索引构建完成！")
    return index


def save_index_and_metadata(index: faiss.Index, metadata: List[Dict[str, Any]], 
                            index_path: str, metadata_path: str):
    """
    保存索引和元数据
    
    Args:
        index: FAISS索引对象
        metadata: 元数据列表（顺序必须与索引一一对应）
        index_path: 索引文件保存路径
        metadata_path: 元数据文件保存路径
    """
    print(f"\n💾 正在保存文件...")
    
    # 验证数量一致性
    if index.ntotal != len(metadata):
        print(f"⚠️  警告: 索引向量数 ({index.ntotal}) 与元数据数 ({len(metadata)}) 不一致")
    
    # 保存FAISS索引
    faiss.write_index(index, index_path)
    print(f"   ✅ 索引已保存: {index_path}")
    
    # 保存元数据（保持JSON格式，null值会被正确保存）
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 元数据已保存: {metadata_path}")
    
    print(f"\n🎉 全部完成！")
    print(f"   索引文件: {index_path}")
    print(f"   元数据文件: {metadata_path}")
    print(f"   总记录数: {len(metadata)}")

# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用 Smart Chunking 策略构建向量索引',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python indexer.py
  python indexer.py --input custom_input.json --output-index custom.index
        """
    )
    parser.add_argument('--input', '-i', 
                       default=DEFAULT_INPUT_FILE,
                       help=f'输入JSON文件路径（默认: {DEFAULT_INPUT_FILE}）')
    parser.add_argument('--output-index', '-o',
                       default=DEFAULT_INDEX_FILE,
                       help=f'输出索引文件路径（默认: {DEFAULT_INDEX_FILE}）')
    parser.add_argument('--output-metadata', '-m',
                       default=DEFAULT_METADATA_FILE,
                       help=f'输出元数据文件路径（默认: {DEFAULT_METADATA_FILE}）')
    args = parser.parse_args()
    
    # 检查API key
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print("❌ 错误: 未找到环境变量 AI_BUILDER_TOKEN")
        print("\n请设置API key:")
        print("   export AI_BUILDER_TOKEN='your_api_key_here'")
        print("\n或创建 .env 文件:")
        print("   echo 'AI_BUILDER_TOKEN=your_api_key_here' > .env")
        sys.exit(1)
    
    # 输入文件路径
    input_file = args.input
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 向量索引构建工具 (Smart Chunking)")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出索引: {args.output_index}")
    print(f"输出元数据: {args.output_metadata}")
    print("=" * 60)
    
    try:
        # 1. 加载chunks并应用Smart Chunking
        chunks = load_and_split_chunks(input_file)
        
        if not chunks:
            print("❌ 没有有效的chunks可处理")
            sys.exit(1)
        
        # 2. 创建API客户端
        client = EmbeddingClient(api_key)
        
        # 3. 批量处理生成embeddings
        embeddings, metadata = process_batches(chunks, client)
        
        if not embeddings:
            print("❌ 没有成功生成任何embeddings")
            sys.exit(1)
        
        # 验证数量一致性
        if len(embeddings) != len(metadata):
            print(f"❌ 错误: embeddings数量 ({len(embeddings)}) 与metadata数量 ({len(metadata)}) 不一致")
            sys.exit(1)
        
        # 4. 构建FAISS索引
        index = build_faiss_index(embeddings)
        
        # 5. 保存索引和元数据
        save_index_and_metadata(index, metadata, 
                               index_path=args.output_index,
                               metadata_path=args.output_metadata)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
