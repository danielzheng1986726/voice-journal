# /run-scan 端点问题诊断与解决方案总结

## 问题概述

在实现 `/run-scan` 端点时，遇到了多个问题，最终定位到是 **token 预算分配** 和 **API 兼容性** 问题。

---

## 问题演进过程

### 问题 1: API 连接错误（已解决）

**现象：**
```
httpx.RemoteProtocolError: Server disconnected without sending a response.
openai.APIConnectionError: Connection error.
```

**原因：**
- 使用了 `response_format={"type": "json_object"}` 参数
- AI Builder Space 的 API 不支持此参数（这是 OpenAI 原生特性）
- 服务端收到不认识的参数时直接断开连接

**解决方案：**
- ✅ 移除了 `response_format` 参数
- ✅ 改为在 prompt 中明确要求返回 JSON 格式
- ✅ 添加了 JSON 解析容错处理（支持提取 ```json 代码块）

---

### 问题 2: 响应内容为空（已解决）

**现象：**
```
INFO:app:✅ [扫描] AI 分析完成，响应长度: 0 字符
ERROR:app:❌ [扫描] JSON 解析失败: Expecting value: line 1 column 1 (char 0)
```

**调试信息（第一次失败）：**
```
🔍 [调试] finish_reason='length'
🔍 [调试] content 值: ''
🔍 [调试] completion_tokens=2000
🔍 [调试] prompt_tokens=1993
```

**根本原因：**
1. **Token 预算分配问题**：
   - `prompt_tokens=1993` + `max_tokens=2000` ≈ 4000 tokens
   - Prompt 太长，占用了太多 token
   - AI 还没开始输出有效内容就被截断
   - AI Builder Space 在 `finish_reason='length'` 时返回空 content（与 OpenAI 原版行为不同）

2. **内容截断阈值过高**：
   - `MAX_CONTENT_LENGTH = 15000` 字符
   - 导致 prompt 过长

**解决方案：**
1. ✅ 将 `max_tokens` 从 2000 增加到 4000
2. ✅ 将 `MAX_CONTENT_LENGTH` 从 15000 减少到 8000
3. ✅ 添加 `finish_reason` 检查，如果被截断且 content 为空，返回友好错误

---

### 问题 3: 最终成功（使用 deepseek 模型）

**调试信息（成功）：**
```
🔍 [调试] finish_reason='stop'  ✅ 正常完成
🔍 [调试] content 值: '现在让我分析这些语音记录...\n\n```json\n{...}\n```'
🔍 [调试] completion_tokens=602
🔍 [调试] prompt_tokens=33166  ⚠️ 仍然很大，但 deepseek 模型成功处理
🔍 [调试] model='deepseek'  ✅ 自动切换到备选模型
```

**关键发现：**
- `gpt-5` 模型可能因为 prompt 过长而失败
- 代码自动切换到 `deepseek` 模型并成功
- 响应包含完整的 JSON（被 ```json 代码块包裹）
- JSON 解析容错处理成功提取了代码块中的内容

---

## 最终代码修改

### 1. 移除不兼容参数
```python
# 移除了
response_format={"type": "json_object"}
```

### 2. 调整 Token 预算
```python
# 修改前
MAX_CONTENT_LENGTH = 15000
max_tokens=2000

# 修改后
MAX_CONTENT_LENGTH = 8000  # 减少 prompt 长度
max_tokens=4000  # 增加输出空间
```

### 3. 添加 finish_reason 检查
```python
if choice.finish_reason == 'length':
    logger.warning("⚠️  [扫描] 响应被截断（达到 max_tokens 限制）")
    if not choice.message.content:
        return JSONResponse(
            status_code=500,
            content={
                "error": "AI 响应被截断且内容为空。请减少分析的记录数量，或稍后重试。",
                "details": f"prompt_tokens: {response.usage.prompt_tokens}, max_tokens: 4000",
                "suggestion": "尝试减少扫描天数或记录数量"
            }
        )
```

### 4. 添加模型备选方案
```python
models_to_try = ["gpt-5", "deepseek"]  # 如果 gpt-5 失败，自动尝试 deepseek
```

### 5. 添加 JSON 解析容错
```python
# 支持提取 ```json ... ``` 代码块中的内容
json_block_patterns = [
    r'```json\s*\n(.*?)\n```',
    r'```\s*\n(.*?)\n```',
    # ...
]
```

---

## 关键学习点

### 1. API 兼容性问题
- 第三方 API 网关不一定实现所有 OpenAI 原生特性
- 使用 prompt 约束输出格式更通用

### 2. Token 预算分配
- LLM 有总 token 限制（通常 4000-8000）
- 需要在 prompt（输入）和 completion（输出）之间合理分配
- Prompt 过长会挤压输出空间

### 3. 错误处理策略
- 检查 `finish_reason` 判断是否被截断
- 不同 API 实现的行为可能不同（如 AI Builder Space 在截断时返回空 content）
- 提供友好的错误提示和解决建议

### 4. 模型备选方案
- 不同模型对长 prompt 的容忍度不同
- 实现模型自动切换可以提高成功率

---

## 当前状态

✅ **问题已解决**
- API 调用成功
- 响应包含完整内容
- JSON 解析成功
- 返回了正确的分析报告

⚠️ **仍需关注**
- `prompt_tokens=33166` 仍然很大（可能超出某些模型的限制）
- 如果记录数量继续增加，可能需要进一步优化：
  - 进一步减少单条记录长度
  - 分批处理记录
  - 使用更智能的内容摘要

---

## 调试信息文件

完整的调试日志保存在：`/Users/xiaodongzheng/vector_indexer/debug_scan.log`

### 最新成功案例（2026-01-16）

**关键指标：**
- 模型：`deepseek` ✅
- finish_reason：`stop`（正常完成）✅
- completion_tokens：720
- prompt_tokens：65654 ⚠️（非常大，但 deepseek 成功处理）
- total_tokens：66374
- content：包含完整的 JSON 分析报告（被 ```json 代码块包裹）✅

**响应内容示例：**
```json
{
  "patterns": [
    {
      "importance": "High",
      "pattern": "反复查询历史抑郁症状记录，显示对心理健康状态的持续关注和潜在担忧",
      "evidence": "记录3、15、17中多次查询2024年11月抑郁症状...",
      "suggestion": "建议关注当前情绪状态，如果对历史抑郁症状有持续担忧，考虑进行心理健康评估..."
    },
    // ... 更多模式
  ],
  "summary": "分析显示用户对个人心理健康状态有较高关注度..."
}
```

**关键发现：**
- `deepseek` 模型能够处理非常大的 prompt（65654 tokens）
- JSON 被正确包裹在 ```json 代码块中
- JSON 解析容错处理成功提取了内容
- 分析结果质量良好，包含了 High/Medium/Low 重要性分类

---

## 建议

1. **监控 token 使用**：如果 prompt_tokens 继续增长，考虑进一步优化
2. **考虑分批处理**：如果记录数量很大，可以分批分析再合并
3. **优化 prompt**：进一步简化 prompt，减少不必要的说明文字
4. **添加缓存**：对于相同的数据，可以缓存分析结果
