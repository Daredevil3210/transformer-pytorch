import config
import torch
from torch.utils.data import DataLoader

from tools.data_loader import MTDataset
from model.tf_model import make_model
import logging
import sacrebleu
from tqdm import tqdm

from beam_decoder import beam_search
from model.train_utils import  MultiGPULossCompute, get_std_opt
from tools.tokenizer_utils import chinese_tokenizer_load
from tools.create_exp_folder import create_exp_folder


logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', level=logging.INFO)

def run_epoch(data, model, loss_compute):
    total_tokens = 0.   # 初始化token的总数
    total_loss = 0.  # 初始化总损失

    # 遍历整个数据集（数据为batch的形式）
    for batch in tqdm(data):  # tqdm用于显示处理进度条
        # 模型前向传播，得到预测结果out
        # batch.src：输入的源语言数据，batch.trg：目标语言数据，batch.src_mask：源语言mask，batch.trg_mask：目标语言mask
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        # 使用loss_compute计算损失
        # batch.trg_y：目标输出数据，batch.ntokens：非填充部分的token数量（有效token数量）
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        # 累加损失和有效tokens的数量
        total_loss += loss
        total_tokens += batch.ntokens

    # 返回每个token的平均损失
    return total_loss / total_tokens

def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    # best_bleu_score初始化
    best_bleu_score = -float('inf')  # 初始最佳BLEU分数为负无穷
    # 创建保存权重的路径
    exp_folder, weights_folder = create_exp_folder()

    # 开始训练循环，迭代每个epoch
    for epoch in range(1, config.epoch_num + 1):
        logging.info(f"第{epoch}轮模型训练与验证")
        # 设置模型为训练模式
        model.train()
        # 进行一个epoch的训练，返回当前的训练损失
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))

        # 设置模型为评估模式（即不计算梯度，优化）
        model.eval()
        # 进行一个epoch的验证，返回当前的验证损失
        dev_loss = run_epoch(dev_data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))

        # 计算模型在验证集（dev_data）上的BLEU分数
        bleu_score = evaluate(dev_data, model)
        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {dev_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")

        # 如果当前epoch的模型的BLEU分数更优，则保存最佳模型
        if bleu_score > best_bleu_score:
            # 如果之前已存在最优模型，先删除
            if best_bleu_score != -float('inf'):
                old_model_path = f"{weights_folder}/best_bleu_{best_bleu_score:.2f}.pth"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            model_path_best = f"{weights_folder}/best_bleu_{bleu_score:.2f}.pth"
            # 保存当前模型的状态字典到指定路径
            torch.save(model.state_dict(), model_path_best)
            # 更新最佳BLEU分数
            best_bleu_score = bleu_score
            # 记录最佳模型保存信息到日志

        # 保存当前模型（最后一次训练）
        if epoch == config.epoch_num:  # 判断是否达到设定的训练轮数
            model_path_last = f"{weights_folder}/last_bleu_{bleu_score:.2f}.pth"  # 构建模型保存路径，包含BLEU分数
            torch.save(model.state_dict(), model_path_last)  # 保存模型的状态字典

def evaluate(data, model):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()  # 加载中文分词器
    trg = []  # 存储目标句子（真实句子）
    res = []  # 存储模型翻译的结果
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):  # 使用tqdm显示进度条
            cn_sent = batch.trg_text  # 获取当前批次的中文句子
            src = batch.src   # 获取当前批次的源语言（英文）句子
            src_mask = (src != 0).unsqueeze(-2)    # 为源语言句子创建mask，排除padding部分

            # 使用束搜索生成模型翻译结果
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)

            # `decode_result`是一个包含多个翻译结果的列表，取最优结果
            decode_result = [h[0] for h in decode_result]
            # 解码后的id转为中文句子
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)  # 将当前批次的真实句子添加到`trg`中
            res.extend(translation)  # 将模型的翻译结果添加到`res`中

    # 计算BLEU分数，使用SacreBLEU工具库
    trg = [trg]  # 真实目标句子
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')  # 计算BLEU分数
    return float(bleu.score)  # 返回BLEU分数


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def run():
    # 创建训练数据集和开发数据集
    # 使用MTDataset类分别加载训练数据和开发数据
    train_dataset = MTDataset(config.train_data_path)   # 初始化训练数据集，使用配置中指定的训练数据路径
    dev_dataset = MTDataset(config.dev_data_path)   # 初始化开发数据集，使用配置中指定的开发数据路径
    test_dataset = MTDataset(config.test_data_path)

    # 创建训练数据加载器，用于训练过程中批量加载数据
    # shuffle=True 表示在每个epoch开始时会打乱数据顺序，以增加模型的泛化能力
    # batch_size=config.batch_size 表示每个批次的样本数量，具体值由配置文件决定
    # collate_fn=train_dataset.collate_fn 表示自定义的数据整理函数，用于处理每个批次的数据
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    #  将模型包装成数据并行模式,这样可以在多个GPU上并行处理数据，提高训练效率
    model_par = torch.nn.DataParallel(model)

    # 训练阶段，选择损失函数和优化器
    # CrossEntropyLoss是常见的分类问题损失函数，ignore_index=0表示忽略填充部分
    # reduction='sum'表示计算损失时会对所有token的损失求和
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 调用get_std_opt函数获取标准的Noam优化器，这通常包括学习率调度器（如预热后衰减）
    optimizer = get_std_opt(model)

    # 开始训练
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    # test(test_dataloader, model, criterion)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    run()
