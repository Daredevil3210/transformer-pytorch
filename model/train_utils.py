import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        """
            初始化多GPU训练中的损失计算类
            :param generator: 生成器模型，通常是生成预测分布的网络
            :param criterion: 损失函数，用于计算预测和目标之间的差距
            :param devices: 使用的GPU设备列表
            :param opt: 优化器对象，进行参数更新
            :param chunk_size: 将数据分割成多个小块进行计算的大小
        """
        # 将生成器复制到多个GPU上
        self.generator = generator
        # 使用nn.parallel.replicate将损失函数复制到多个GPU
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt  # 优化器（可选）
        self.devices = devices  # 设备列表，包含多个GPU的ID
        self.chunk_size = chunk_size  # 每次计算的数据块大小


    def __call__(self, out, targets, normalize):
        """
            进行多GPU的损失计算和训练
            :param out: 模型输出
            :param targets: 目标数据（真实标签）
            :param normalize: 用于规范化损失的常数
            :return: 总损失值（乘以normalize）
        """
        total = 0.0  # 初始化总损失
        # 将生成器复制到多个GPU上
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        # 将模型输出分发到多个GPU
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        # 初始化损失梯度列表（每个GPU一个）
        out_grad = [[] for _ in out_scatter]
        # 将目标标签分发到多个GPU
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # 将数据划分为多个块（chunk），进行批量计算
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # 对每个chunk进行预测
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]

            # 并行应用生成器进行预测
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算每个块的损失
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # 汇总损失并进行规范化
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # 反向传播损失到Transformer的输出
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # 反向传播所有损失，通过Transformer进行参数更新
        if self.opt is not None:
            # 将所有GPU的梯度合并
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out # 保存原始的输出
            # 将梯度合并到主设备上（通常是第一个GPU）
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)  # 进行反向传播更新参数
            self.opt.step()
            self.opt.optimizer.zero_grad()  # 清除梯度（使用Noam优化器）
        return total * normalize  # 返回总损失（乘以normalize）

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
            初始化优化器包装类
            :param model_size: 模型的大小，通常是d_model的大小，用于计算学习率
            :param factor: 用于计算学习率的因子（通常是学习率的初始值）
            :param warmup: 预热步数，决定学习率从较小值到较大值的增长速度
            :param optimizer: 实际使用的优化器（例如Adam、SGD等）
        """
        self.optimizer = optimizer  # 存储实际的优化器（如Adam）
        self._step = 0  # 当前训练的步数
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 学习率的因子
        self.model_size = model_size  # 模型的大小（通常是d_model，决定学习率的尺度）
        self._rate = 0  # 当前的学习率

    def step(self):
        """更新优化器的参数和学习率"""
        self._step += 1  # 增加当前步数
        rate = self.rate()  # 计算当前的学习率
        # 更新优化器中所有参数的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 设置当前学习率
        self._rate = rate  # 更新学习率
        self.optimizer.step()  # 执行一次优化步骤（更新参数）

    def rate(self, step=None):
        """根据当前步数计算学习率"""
        # 如果没有传入step，使用当前步数
        if step is None:
            step = self._step
        # 学习率计算公式：factor * (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    # 创建并返回一个NoamOpt优化器，包含Adam优化器作为基础
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))