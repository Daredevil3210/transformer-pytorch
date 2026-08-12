import torch

"""
模型超参数配置
这一类参数决定了模型架构的规模和计算复杂度。
更高的维度、更多的层和头可以增强模型的能力，但也会增加计算和训练时间。
"""
# d_model = 512 表示模型的每个token的表示将使用512维的向量，这也决定了Transformer中间层的大小。
d_model = 512
# 多头注意力机制中的头数。
n_heads = 8
# n_layers = 6表示模型中有6个Transformer编码器和解码器层。
n_layers = 6
# 自注意力机制中每个头的键（Key）向量的维度。
d_k = 64
# 自注意力机制中每个头的值（Value）向量的维度。
d_v = 64
# d_ff是前馈网络隐藏层的大小。d_ff = 2048表示前馈层的维度为2048。
d_ff = 2048
# dropout = 0.1表示在训练过程中，随机丢弃10%的神经元来避免模型过拟合。
dropout = 0.1

"""
词汇表和标记配置
这些参数控制词汇表的大小和特殊标记的设置。
这一类参数涉及到数据预处理、分词和输入的标记设置。
"""
# 源语言（英语）的词汇表大小。
src_vocab_size = 32000
# 目标语言（中文）的词汇表大小。
tgt_vocab_size = 32000
# padding_idx = 0表示填充token的索引为0，这通常用于填充短句，使得每个句子都具有相同的长度。
padding_idx = 0
# bos_idx = 2表示句子的开始符号（BOS）的索引是2。
bos_idx = 2
# eos_idx = 3表示句子的结束符号（EOS）的索引是3。
eos_idx = 3

"""
训练配置
这些参数控制训练过程中的配置和训练策略。
"""
# 训练时的批次大小。
batch_size = 32
# 训练的总轮次。
epoch_num = 3
# 学习率（learning rate）。
lr = 3e-4

"""
解码和生成设置
这些参数控制模型生成输出时的行为，影响生成的句子质量。
"""
# greed decode的最大句子长度
# max_len = 60表示解码时生成的最大句子长度为60个token。
max_len = 60
# 在计算BLEU评分时使用的Beam Search的大小。
# beam_size = 3表示在解码时，使用大小为3的Beam Search进行翻译。
beam_size = 3

"""
文件路径和模型配置
这些参数用于定义文件路径和是否加载预训练模型的设置。
"""
data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'

model_path = './weights/transformer_model.pth'
test_model_path = './run/train/exp/weights/best_bleu_26.30.pth'

"""
设备配置
这些参数用于配置模型运行的硬件设备。
"""
# 指定使用的GPU设备的ID。
# 指定设备ID的列表。
gpu_id = '0'
device_id = [0]
# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')