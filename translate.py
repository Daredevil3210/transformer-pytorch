import torch
import config
import logging
import numpy as np
from tools.tokenizer_utils import english_tokenizer_load
from model.tf_model import make_model
from tools.tokenizer_utils import chinese_tokenizer_load
from beam_decoder import beam_search

logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d', level=logging.INFO)


def translate(src, model):
    """用训练好的模型进行预测单句，打印模型翻译结果"""

    # 加载中文分词器
    sp_chn = chinese_tokenizer_load()

    with torch.no_grad():  # 禁用梯度计算，以节省内存
        # 加载训练好的模型权重
        model.load_state_dict(torch.load(config.test_model_path))
        model.eval()  # 将模型设置为评估模式

        # 创建源句子的掩码（mask），以确保填充的部分不会参与计算
        src_mask = (src != 0).unsqueeze(-2)

        # 使用束搜索（beam search）进行解码
        decode_result, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,  # 最大翻译长度
            config.padding_idx,  # 填充符号的索引
            config.bos_idx,  # 句子开始符号的索引
            config.eos_idx,  # 句子结束符号的索引
            config.beam_size,  # 束搜索的大小
            config.device  # 设备（CPU或GPU）
        )

        # 从解码结果中提取最优结果
        decode_result = [h[0] for h in decode_result]

        # 使用中文分词器将解码结果的id转化为实际的中文词语
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]

        # # 打印并返回翻译结果的第一句
        # print(translation[0])
        return translation[0]


def one_sentence_translate(sent):
    """翻译单句英文"""

    # 初始化翻译模型，使用指定的参数（词汇表大小、层数、模型维度等）
    model = make_model(
        config.src_vocab_size,  # 源语言词汇表大小
        config.tgt_vocab_size,  # 目标语言词汇表大小
        config.n_layers,  # 模型的层数
        config.d_model,  # 模型的维度（通常是隐藏层的大小）
        config.d_ff,  # 前馈网络的维度
        config.n_heads,  # 注意力头的数量
        config.dropout  # dropout比率
    )

    # 加载源语言和目标语言的分词器，用于获取BOS和EOS标记
    BOS = english_tokenizer_load().bos_id()  # 获取开始符号（BOS）的ID，通常是2
    EOS = english_tokenizer_load().eos_id()  # 获取结束符号（EOS）的ID，通常是3

    # 将输入的句子转化为token IDs，添加BOS和EOS标记
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]

    # 将句子转换为长整型Tensor，并发送到指定的设备（如GPU或CPU）
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)

    # 调用translate函数进行翻译
    return translate(batch_input, model)


def translate_example():
    """单句翻译示例"""
    "The government has implemented various policies toimprove the living standards of its citizens."
    "政府实施了诸多政策，改善公民的生活水平。"

    while True:  # 使用循环，让用户可以反复输入句子
        # 提示用户输入英文句子
        sent = input("请输入英文句子进行翻译：")

        translation = one_sentence_translate(sent)
        # 调用翻译函数进行翻译
        print("翻译结果：", translation)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    translate_example()
