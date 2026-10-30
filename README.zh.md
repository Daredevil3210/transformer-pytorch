[English](README.md) | [简体中文](README.zh.md)

# PyTorch Transformer 中英机器翻译

面向学习的英译中项目，手写实现 Transformer 编码器与解码器，并包含 SentencePiece 分词、训练工具和 Beam Search 推理流程。

仓库提供源码与分词器模型，不包含平行语料和训练后的翻译权重，未提供可复现的基准成绩或自动化测试套件。

## 学习内容

- 多头注意力、正弦位置编码、残差连接与前馈网络。
- Padding Mask、因果 Mask、Teacher Forcing 与词元级交叉熵。
- Adam 与 Noam 学习率调度，以及面向 CUDA 的并行损失计算。
- Beam Search 解码和使用中文分词模式的 SacreBLEU 验证。
- 交互式英文单句翻译。

`config.py` 默认使用 6 层编码器/解码器、512 维模型、8 个注意力头、2048 维前馈层，中英文词表各为 32,000。这些数字是配置参数，不代表模型质量。

## 环境准备

依赖文件的注释记录了 Python 3.10 和历史 PyTorch/CUDA 组合；PyTorch 需要单独安装。选择适合本机硬件的 PyTorch 版本后，再安装其余固定版本依赖：

```bash
python -m venv .venv
# 在当前终端激活 .venv，并单独安装 PyTorch。
python -m pip install -r requirements.txt
```

其余依赖包括 NumPy、SentencePiece、SacreBLEU 和 tqdm。历史硬件说明并非经过验证的兼容性矩阵。应用命令应在仓库根目录执行。

## 数据与配置

训练前准备以下三个 UTF-8 JSON 文件。虽然测试调用处于注释状态，`main.py` 仍会创建测试数据集：

```text
data/json/train.json
data/json/dev.json
data/json/test.json
```

每个文件是 JSON 数组，每条记录为 `[英文原文, 中文译文]`：

```json
[
  ["Hello world.", "你好，世界。"],
  ["How are you?", "你好吗？"]
]
```

请使用有权使用的语料。仓库未提供语料下载流程。`data/get_corpus.py` 从已有 JSON 划分中提取逐行文本，并不生成数据划分；语料分析和 Unicode 预览脚本可按需使用。

程序加载 `tokenizer/eng.model` 和 `tokenizer/chn.model`。重新训练分词器时，需保持词表大小与 PAD/BOS/EOS ID 和 `config.py` 一致，并使用新分词器重新训练翻译权重。`tokenizer/tokenize.py` 要求先准备 `data/corpus.en` 和 `data/corpus.ch`，使用相对于 `tokenizer/` 的路径。

运行前检查设备配置。默认使用 CUDA 设备 0（`gpu_id = '0'`、`device_id = [0]`），批次大小 32，训练 3 轮；`main.py` 和 `translate.py` 也会将 `CUDA_VISIBLE_DEVICES` 设为 `0`。训练所用 `MultiGPULossCompute` 的复制与分发操作依赖 CUDA，仅将 `gpu_id` 改为空字符串并不能实现 CPU 训练。CPU 推理也需要匹配的权重加载和设备设置。

## 训练与翻译

准备好数据、依赖和设备配置后执行：

```bash
python main.py
```

训练每轮计算开发集损失与 BLEU，将最佳和最后一轮权重保存到 `run/train/exp/weights/`；若已有实验目录，则依次使用 `exp1`、`exp2` 等目录。

将 `config.test_model_path` 改为实际存在且与架构、分词器匹配的权重路径，再运行：

```bash
python translate.py
```

按提示输入英文句子，使用终端中断快捷键退出。默认的 `best_bleu_26.30.pth` 只是示例文件名，仓库未提供该权重，也不表示独立验证的分数。原说明记录翻译权重约 376 MB，未纳入 Git。

## 目录说明

```text
main.py                训练与开发集评估
translate.py           交互式翻译
config.py              架构、路径、权重和设备配置
beam_decoder.py        Beam Search
model/tf_model.py      Transformer 实现
model/train_utils.py   并行损失计算与 Noam 优化器
tools/                 数据集、Mask、分词器与实验目录工具
tokenizer/             SentencePiece 模型与词表
data/                  语料工具与本地 JSON 数据
weights/               本地权重目录占位
```

## 评估限制与来源

开发集评估使用 `sacrebleu.corpus_bleu(..., tokenize='zh')`。被注释的 `test()` 向 `evaluate()` 多传了一个参数，测试集评估前需要修正。仓库没有可直接使用的测试命令或自动化测试套件。本次文档更新未运行训练或推理。

原始训练语料与教学资料未纳入仓库，相关权利归原作者或提供方所有。仓库未声明许可证文件，未提供明确的开源许可授权。
