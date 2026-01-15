# 快速开始指南

## 1. 安装依赖

```bash
cd vector_indexer
pip install -r requirements.txt
```

## 2. 配置API Key

**方法1：环境变量（推荐）**

```bash
export AI_BUILDER_TOKEN='your_api_key_here'
```

**方法2：.env文件**

```bash
# 复制示例文件
cp .env.example .env

# 编辑.env文件，填入你的API key
nano .env  # 或使用其他编辑器
```

## 3. 准备输入文件

将你的 `all_chunks.json` 文件放在 `vector_indexer` 目录下。

## 4. 运行脚本

```bash
python indexer.py
```

## 5. 查看结果

运行完成后，会在当前目录生成：
- `my_history.index` - FAISS向量索引
- `chunks_metadata.json` - 元数据文件

## 使用自定义文件路径

```bash
# 指定输入文件
python indexer.py --input /path/to/your/chunks.json

# 指定输出文件
python indexer.py --output-index my_index.index --output-metadata metadata.json

# 完整示例
python indexer.py \
  --input /path/to/chunks.json \
  --output-index output.index \
  --output-metadata output_metadata.json
```

## 查看帮助

```bash
python indexer.py --help
```
