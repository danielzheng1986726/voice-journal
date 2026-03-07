#!/usr/bin/env python3
"""
历史笔记导入脚本

将历史笔记数据导入到 all_chunks.json，然后重建索引。

用法：
    python import_history_notes.py <历史笔记文件路径>

历史笔记文件格式（JSON）：
[
  {
    "id": "note_20140602_001",
    "source": "note",  # 或 "diary", "journal" 等
    "date": "2014-06-02",
    "content": "今天做了人才挖链项目，很开心..."
  }
]

或者如果历史笔记在 voice_records.json 中但还没同步：
    python import_history_notes.py --from-voice-records
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 文件路径
ALL_CHUNKS_FILE = Path(__file__).parent / "all_chunks.json"
VOICE_RECORDS_FILE = Path(__file__).parent / "voice_records.json"

def import_from_file(history_file: str):
    """从指定文件导入历史笔记"""
    history_path = Path(history_file)
    if not history_path.exists():
        print(f"❌ 文件不存在: {history_path}")
        return False
    
    print(f"📖 读取历史笔记文件: {history_path}")
    with open(history_path, 'r', encoding='utf-8') as f:
        history_notes = json.load(f)
    
    print(f"✅ 读取到 {len(history_notes)} 条历史笔记")
    
    # 读取现有的 all_chunks.json
    if ALL_CHUNKS_FILE.exists():
        with open(ALL_CHUNKS_FILE, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        existing_ids = {c.get('id') for c in existing_chunks}
        print(f"📚 现有记录: {len(existing_chunks)} 条")
    else:
        existing_chunks = []
        existing_ids = set()
        print("📚 创建新的 all_chunks.json")
    
    # 导入新记录（去重）
    imported_count = 0
    skipped_count = 0
    null_date_count = 0
    
    for i, note in enumerate(history_notes):
        # 处理 Notion 格式：full_content -> content
        content = note.get('full_content') or note.get('content', '')
        if not content or not content.strip():
            skipped_count += 1
            continue
        
        note_id = note.get('id')
        if not note_id:
            # 如果没有 ID，生成一个（基于日期和内容哈希）
            date_str = note.get('date') or 'unknown'
            content_hash = abs(hash(content[:100])) % 100000
            note_id = f"notion_{date_str.replace('-', '') if date_str != 'unknown' else 'nodate'}_{content_hash}"
        
        if note_id in existing_ids:
            skipped_count += 1
            continue
        
        # 处理日期为 null 的情况
        date_value = note.get('date')
        if date_value is None:
            null_date_count += 1
        
        # 确保格式正确（all_chunks.json 格式：id, source, date, content）
        chunk = {
            "id": note_id,
            "source": note.get("source", "notion"),
            "date": date_value,  # 可以是 null
            "content": content.strip()
        }
        
        existing_chunks.append(chunk)
        existing_ids.add(note_id)
        imported_count += 1
    
    # 保存
    with open(ALL_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✨ 导入完成！")
    print(f"   - 新增: {imported_count} 条")
    print(f"   - 跳过（已存在）: {skipped_count} 条")
    print(f"   - 总计: {len(existing_chunks)} 条")
    
    return True

def import_from_voice_records():
    """从 voice_records.json 导入未同步的记录"""
    if not VOICE_RECORDS_FILE.exists():
        print(f"❌ 文件不存在: {VOICE_RECORDS_FILE}")
        return False
    
    print(f"📖 读取 voice_records.json")
    with open(VOICE_RECORDS_FILE, 'r', encoding='utf-8') as f:
        voice_records = json.load(f)
    
    print(f"✅ 读取到 {len(voice_records)} 条录音记录")
    
    # 读取现有的 all_chunks.json
    if ALL_CHUNKS_FILE.exists():
        with open(ALL_CHUNKS_FILE, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        existing_ids = {c.get('id') for c in existing_chunks}
        print(f"📚 现有记录: {len(existing_chunks)} 条")
    else:
        existing_chunks = []
        existing_ids = set()
        print("📚 创建新的 all_chunks.json")
    
    # 导入未同步的记录
    imported_count = 0
    skipped_count = 0
    
    for record in voice_records:
        record_id = record.get('id')
        if not record_id or record_id in existing_ids:
            skipped_count += 1
            continue
        
        # 转换为 all_chunks 格式（去掉 time 字段）
        chunk = {
            "id": record_id,
            "source": record.get("source", "voice"),
            "date": record.get("date"),
            "content": record.get("content", "")
        }
        
        existing_chunks.append(chunk)
        existing_ids.add(record_id)
        imported_count += 1
    
    # 保存
    with open(ALL_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✨ 导入完成！")
    print(f"   - 新增: {imported_count} 条")
    print(f"   - 跳过（已存在）: {skipped_count} 条")
    print(f"   - 总计: {len(existing_chunks)} 条")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 从文件导入: python import_history_notes.py <历史笔记文件路径>")
        print("  2. 从 voice_records.json 导入: python import_history_notes.py --from-voice-records")
        print("\n历史笔记文件格式（JSON）:")
        print('  [{"id": "note_20140602_001", "source": "note", "date": "2014-06-02", "content": "..."}]')
        sys.exit(1)
    
    if sys.argv[1] == "--from-voice-records":
        success = import_from_voice_records()
    else:
        success = import_from_file(sys.argv[1])
    
    if success:
        print("\n📌 下一步：重建索引")
        print("   运行: python indexer.py")
        print("   或在 Admin 页面点击'手动重建索引'")

if __name__ == "__main__":
    main()
