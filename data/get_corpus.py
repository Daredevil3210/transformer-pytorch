import json  # 导入json模块，用于处理JSON格式的数据

if __name__ == "__main__":  # 主程序入口，当脚本被直接执行时运行
    files = ['train', 'dev', 'test']  # 定义文件列表，包含训练集、开发集和测试集
    ch_path = 'corpus.ch'  # 中文语料库文件路径
    en_path = 'corpus.en'  # 英文语料库文件路径
    ch_lines = []  # 用于存储中文句子的列表
    en_lines = []  # 用于存储英文句子的列表

    for file in files:  # 遍历文件列表
        # 加载JSON格式的语料文件，使用utf-8编码
        corpus = json.load(open('./json/' + file + '.json', 'r', encoding="utf-8"))
        # 遍历语料中的每一项，提取英文和中文句子
        for item in corpus:
            en_lines.append(item[0] + '\n')  # 将英文句子添加到列表，并添加换行符
            ch_lines.append(item[1] + '\n')  # 将中文句子添加到列表，并添加换行符

    # 将中文句子写入文件
    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)  # 一次性写入所有中文句子

    # 将英文句子写入文件
    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)  # 一次性写入所有英文句子

    # 输出中文句子的行数
    # lines of Chinese: 252777
    print("lines of Chinese: ", len(ch_lines))
    # 输出英文句子的行数
    # lines of English: 252777
    print("lines of English: ", len(en_lines))
    # 输出完成提示信息
    print("-------- Get Corpus ! --------")
