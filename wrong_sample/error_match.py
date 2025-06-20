import json

def normalize_question(question: str) -> str:
    """标准化问题：小写处理并移除末尾标点"""
    question = question.strip().lower()
    if question and question[-1] in {'?', '.', '!'}:
        question = question[:-1].strip()
    return question

# 1. 读取错误问题文件并构建标准化问题集合
with open('rage_webqsp_GPT-4.1_error.json', 'r', encoding='utf-8') as f:
    error_data = json.load(f)
    
error_questions = set()
for item in error_data:
    normalized = normalize_question(item["question"])
    error_questions.add(normalized)

# 2. 读取WebQSP文件并匹配问题
with open('../data/WebQSP.json', 'r', encoding='utf-8') as f:
    webqsp_data = json.load(f)

matching_items = []
for item in webqsp_data:
    # 标准化原始问题
    normalized_raw = normalize_question(item["RawQuestion"])
    
    # 检查是否在错误问题集中
    if normalized_raw in error_questions:
        matching_items.append(item)

# 3. 保存匹配结果到新文件
with open('error_questions.json', 'w', encoding='utf-8') as f:
    json.dump(matching_items, f, indent=2, ensure_ascii=False)

print(f"找到 {len(matching_items)} 个匹配项，已保存到 matched_questions.json")