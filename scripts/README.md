# Batch Processing Scripts

## Batch Compute Fingerprints

### Python Script (推荐 Windows)

```powershell
# 编辑 scripts/batch_compute.py 配置模型路径
uv run python scripts/batch_compute.py
```

### Bash Script (Linux/macOS/WSL)

```bash
# 编辑 scripts/batch_compute.sh 配置模型路径
bash scripts/batch_compute.sh
```

## 配置说明

在脚本中修改以下配置：

```python
# 设备配置
ATTN_DEVICE = "cuda:0"  # 或 "cuda:1", "cpu" 等

# 模型路径映射
MODEL_PATHS = {
    "Llama-2-7B": "/path/to/llama-2-7b",
    "Qwen2.5-7B": "/path/to/qwen2.5-7b",
    # 添加更多模型
}

# 要处理的模型列表
MODELS = [
    "Llama-2-7B",
    "Qwen2.5-7B",
    # 取消注释以处理更多模型
]
```

## 输出

指纹文件将保存到：
```
output/comput_W/fingerprint_{model_name}.json
```

## 示例

```powershell
# 1. 编辑配置
# 修改 scripts/batch_compute.py 中的 MODEL_PATHS 和 MODELS

# 2. 运行批处理
uv run python scripts/batch_compute.py

# 3. 比较结果
uv run attndiff-compare --base output/comput_W/fingerprint_Llama-2-7B.json --dir output/comput_W --cka linear
```
