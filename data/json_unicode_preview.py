import json

path = "../data/json/test.json"  # 你的这段文件
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 自动把 \uXXXX 还原

# 看前两条，ensure_ascii=False 用于“打印中文”
print(json.dumps(data[:3], ensure_ascii=False, indent=2))

# data 的结构应当是：[[en, zh], [en, zh], ...]
print(type(data), len(data), type(data[0]), len(data[0]))
