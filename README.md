# Transformer PyTorch 中英机器翻译

这是一个基于 PyTorch 实现的 Transformer 中英机器翻译项目，包含模型结构、训练流程、Beam Search 解码、SentencePiece 分词器以及单句翻译入口。

## 功能

- 从头实现 Transformer 编码器与解码器
- 支持中英平行语料训练、验证与测试
- 使用 SentencePiece 完成中英文分词
- 使用 Beam Search 生成中文译文
- 使用 SacreBLEU 评估翻译质量

## 项目结构

```text
.
├── main.py                 # 训练与评估入口
├── translate.py            # 单句翻译入口
├── config.py               # 模型、训练和文件路径配置
├── beam_decoder.py         # Beam Search 解码
├── model/                  # Transformer 模型与训练工具
├── tools/                  # 数据加载、分词器和实验目录工具
├── tokenizer/              # SentencePiece 模型与词表
└── data/                   # 语料处理脚本与本地数据目录
```

## 环境要求

- Python 3.10（原项目使用版本）
- PyTorch（请根据 CPU 或 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装）

安装其余依赖：

```bash
pip install -r requirements.txt
```

## 准备数据

训练数据未提交到仓库。请在 `data/json/` 下准备以下文件：

```text
data/json/train.json
data/json/dev.json
data/json/test.json
```

每个文件都是 JSON 数组，每条数据由英文原文和中文译文组成：

```json
[
  ["Hello world.", "你好，世界。"],
  ["How are you?", "你好吗？"]
]
```

如果已有对齐的英文与中文语料，可参考 `data/get_corpus.py`、`data/analyze_corpus.py` 和 `data/json_unicode_preview.py` 进行处理。

## 训练

在 `config.py` 中调整超参数、GPU 编号和数据路径，然后运行：

```bash
python main.py
```

训练产生的权重会保存在 `run/train/exp/weights/`。权重文件默认被 Git 忽略，避免超过 GitHub 的文件大小限制。

## 单句翻译

将训练好的权重放到 `config.py` 中 `test_model_path` 指定的位置，然后运行：

```bash
python translate.py
```

按提示输入英文句子即可获得中文翻译结果。

## 仓库说明

- 原始模型权重单文件约 376 MB，未纳入仓库。
- 原始训练语料和教学资料未纳入仓库；使用或公开数据时，请确认相应的数据与文档许可。
- 项目目前未声明开源许可证，默认保留所有权利。
