#!/usr/bin/env python3
"""
数据迁移脚本：将旧数据归属到指定用户

用法：
    python migrate_user_data.py <user_id>

示例：
    python migrate_user_data.py aYXzFRDYjtUaD1eYzCrfzwfHalp1
"""

import json
import sys
from pathlib import Path

# 文件路径
CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"
RECORDS_FILE = Path(__file__).parent / "voice_records.json"

def migrate_conversations(user_id: str):
    """迁移会话数据"""
    if not CONVERSATIONS_FILE.exists():
        print(f"⚠️  文件不存在: {CONVERSATIONS_FILE}")
        return 0
    
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    migrated_count = 0
    for conv in conversations:
        if not conv.get("user_id"):
            conv["user_id"] = user_id
            migrated_count += 1
    
    if migrated_count > 0:
        with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        print(f"✅ 迁移了 {migrated_count} 个会话到用户 {user_id}")
    else:
        print(f"ℹ️  所有会话已有 user_id，无需迁移")
    
    return migrated_count

def migrate_records(user_id: str):
    """迁移记录数据"""
    if not RECORDS_FILE.exists():
        print(f"⚠️  文件不存在: {RECORDS_FILE}")
        return 0
    
    with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    migrated_count = 0
    for record in records:
        if not record.get("user_id"):
            record["user_id"] = user_id
            migrated_count += 1
    
    if migrated_count > 0:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ 迁移了 {migrated_count} 条记录到用户 {user_id}")
    else:
        print(f"ℹ️  所有记录已有 user_id，无需迁移")
    
    return migrated_count

def main():
    if len(sys.argv) < 2:
        print("用法: python migrate_user_data.py <user_id>")
        print("示例: python migrate_user_data.py aYXzFRDYjtUaD1eYzCrfzwfHalp1")
        sys.exit(1)
    
    user_id = sys.argv[1]
    print(f"🔄 开始迁移数据到用户: {user_id}\n")
    
    # 备份原文件
    if CONVERSATIONS_FILE.exists():
        backup_file = CONVERSATIONS_FILE.with_suffix('.json.bak')
        import shutil
        shutil.copy2(CONVERSATIONS_FILE, backup_file)
        print(f"📦 已备份: {backup_file}")
    
    if RECORDS_FILE.exists():
        backup_file = RECORDS_FILE.with_suffix('.json.bak')
        import shutil
        shutil.copy2(RECORDS_FILE, backup_file)
        print(f"📦 已备份: {backup_file}")
    
    print()
    
    # 执行迁移
    conv_count = migrate_conversations(user_id)
    record_count = migrate_records(user_id)
    
    print(f"\n✨ 迁移完成！")
    print(f"   - 会话: {conv_count} 个")
    print(f"   - 记录: {record_count} 条")

if __name__ == "__main__":
    main()
