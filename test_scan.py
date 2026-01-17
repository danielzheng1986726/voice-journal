#!/usr/bin/env python3
"""测试扫描功能，查看双模型协作日志"""
import requests
import json
import time

print("🚀 开始测试扫描功能...")
print("=" * 60)

try:
    start_time = time.time()
    
    # 触发扫描
    print("📡 发送扫描请求...")
    response = requests.post(
        "http://localhost:8000/run-scan",
        timeout=180  # 3分钟超时
    )
    
    elapsed = time.time() - start_time
    print(f"⏱️  总耗时: {elapsed:.2f} 秒")
    print(f"📊 状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ 扫描成功！")
        print(f"📅 扫描周期: {result.get('scan_period', 'N/A')}")
        print(f"📝 分析记录数: {result.get('records_analyzed', 0)}")
        
        if 'deep_dive_report' in result:
            patterns = result['deep_dive_report'].get('patterns', [])
            print(f"🔍 识别到 {len(patterns)} 个模式")
            
            # 显示模式摘要
            for i, pattern in enumerate(patterns[:3], 1):
                print(f"\n  模式 {i} [{pattern.get('importance', 'N/A')}]:")
                print(f"    {pattern.get('pattern', 'N/A')[:80]}...")
        else:
            print("⚠️  未找到 deep_dive_report")
            print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
    else:
        print(f"❌ 扫描失败，状态码: {response.status_code}")
        print(f"响应内容: {response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("❌ 请求超时（超过3分钟）")
    print("💡 提示：双模型调用可能需要较长时间，请检查服务日志")
except Exception as e:
    print(f"❌ 错误: {e}")

print("\n" + "=" * 60)
print("📋 检查扫描结果文件...")
try:
    with open('scan_results.json', 'r', encoding='utf-8') as f:
        scan_data = json.load(f)
        print(f"✅ 文件存在，最后扫描时间: {scan_data.get('scan_time', 'N/A')}")
        if 'result' in scan_data and 'deep_dive_report' in scan_data['result']:
            patterns_count = len(scan_data['result']['deep_dive_report'].get('patterns', []))
            print(f"📊 识别到 {patterns_count} 个模式")
except FileNotFoundError:
    print("⚠️  scan_results.json 文件不存在")
except Exception as e:
    print(f"❌ 读取文件错误: {e}")
