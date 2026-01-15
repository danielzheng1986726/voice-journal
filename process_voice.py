#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音记录处理脚本
功能：将语音转写的文本转换为格式化的 JSON 记录并保存
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

def generate_id():
    """生成唯一 ID，格式：voice_YYYYMMDD_HHMM"""
    now = datetime.now()
    return f"voice_{now.strftime('%Y%m%d_%H%M')}"

def create_record(content):
    """创建一条记录"""
    now = datetime.now()
    return {
        "id": generate_id(),
        "source": "voice",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "content": content
    }

def load_records(file_path):
    """加载现有记录"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_records(records, file_path):
    """保存记录到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def main():
    # 获取输入文本
    if len(sys.argv) > 1:
        # 从命令行参数读取
        content = ' '.join(sys.argv[1:])
    else:
        # 从标准输入读取
        content = sys.stdin.read().strip()
    
    if not content:
        print("错误：未提供文本内容", file=sys.stderr)
        sys.exit(1)
    
    # 创建记录
    record = create_record(content)
    
    # 打印 JSON 到终端
    print(json.dumps(record, ensure_ascii=False, indent=2))
    
    # 追加到文件
    records_file = Path(__file__).parent / "voice_records.json"
    records = load_records(records_file)
    records.append(record)
    save_records(records, records_file)
    
    print(f"\n✓ 记录已保存到 {records_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
