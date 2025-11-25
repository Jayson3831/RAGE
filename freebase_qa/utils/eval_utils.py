import json
import re
import pickle
from config.settings import DATASET_PATHS, PKL_DATASET_PATHS
from typing import List, Tuple, Union


def prepare_dataset_for_eval(dataset_name, output_file):
    dataset = DATASET_PATHS[dataset_name]
    with open(dataset, encoding='utf-8') as f:
        datas = json.load(f)

    if 'noisy' in dataset_name:
        question_string = 'Noisy_Question'
    elif 'webqsp' in dataset_name:
        question_string = 'RawQuestion'
    else:
        question_string = 'question'
    with open(output_file, encoding='utf-8') as f:
        output_datas = json.load(f)
    return datas, question_string, output_datas

def load_pkl_dataset(dataset_name, output_file):
    dataset = PKL_DATASET_PATHS[dataset_name]
    with open(dataset, 'rb') as f:
        datas = pickle.load(f)
    question_string = 'question'
    with open(output_file, encoding='utf-8') as f:
        output_datas = json.load(f)
    return datas, question_string, output_datas

def align(dataset_name, question_string, data, ground_truth_datas):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data['question']][0]
    if dataset_name == 'cwq' or dataset_name == 'noisy_cwq' or dataset_name == 'ow_grailqa' or dataset_name == 'ow_webqsp' or 'cwq' in dataset_name:
        answer = origin_data["answer"]
        answer_list.append(answer)

    elif dataset_name == 'webqsp' or dataset_name == 'noisy_webqsp' or 'webqsp' in dataset_name:
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa' or dataset_name == 'noisy_grailqa' or 'grailqa' in dataset_name:
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webq' or dataset_name == 'noisy_webq':
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)

    return list(set(answer_list))

def pkl_align(question_string, data, ground_truth_datas):
    origin_data = [j for j in ground_truth_datas if j[question_string] == data['question']][0]
    answers = origin_data["answers"]

    return answers

def check_string(string):
    return "{" in string

# def clean_results(string):
#     if "{" in string:
#         start = string.find("{") + 1
#         end = string.find("}")
#         content = string[start:end]
#         return content
#     else:
#         return "NULL"
    
def clean_results(string):
    matches = re.findall(r"\{(.*?)\}", string)
    if not matches:
        return string  # 没有匹配项
    if matches[0] == "Yes":
        if len(matches) > 1:
            return matches[1]
        return string
    return matches[0]  # 不是 Yes，就取第一个

def check_refuse(string):
    refuse_words = ["however", "sorry"]
    return any(word in string.lower() for word in refuse_words)


def exact_match(response, answer):
    if isinstance(response, list):
        cleaned_list = [item.strip().lower() for item in response]
        clean_result = ",".join(cleaned_list)
    elif isinstance(response, str):
        clean_result = response.strip().replace(" ","").lower()
    clean_answer = answer.strip().replace(" ","").lower()
    if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
        return True
    return False

def eval_hit(prediction, answers):
    for answer in answers:
        if exact_match(prediction, answer):
            return 1
    return 0

def eval_acc(prediction, answers):
    matched = 0
    if len(answers) == 0:
        return 0
    for answer in answers:
        if exact_match(prediction, answer):
            matched += 1
    return matched / len(answers)

def precision_recall_f1(predictions: Union[str, List[str]], answers: List[str]) -> Tuple[float, float, float]:
    """
    计算精确率、召回率和F1分数，适配单个或多个预测

    Args:
        predictions: 模型预测的答案（可以是单个字符串或字符串列表）
        answers: 所有可接受的正确答案列表
    """
    # 统一处理预测为列表形式
    if isinstance(predictions, str):
        pred_list = [predictions]
    else:
        pred_list = predictions

    # 处理空值情况
    if not answers and not pred_list:
        return 1.0, 1.0, 1.0

    if not answers:
        return 0.0, 1.0, 0.0

    if not pred_list:
        return 1.0, 0.0, 0.0

    # 单个预测的特殊处理
    if len(pred_list) == 1:
        return precision_recall_f1_single(pred_list[0], answers)

    # 多个预测的情况
    return precision_recall_f1_multiple(pred_list, answers)


def precision_recall_f1_single(prediction: str, answers: List[str]) -> Tuple[float, float, float]:
    """
    处理单个预测的情况
    """
    # 检查预测是否匹配任何一个答案
    is_correct = any(exact_match(prediction, ans) for ans in answers)

    # 精确率：单个预测，要么全对要么全错
    precision = 1.0 if is_correct else 0.0

    # 召回率：如果正确，则召回率为1/正确答案数量
    if is_correct:
        # 找到匹配的答案数量（可能有多个答案匹配同一个预测）
        matched_answers_count = sum(1 for ans in answers if exact_match(prediction, ans))
        recall = matched_answers_count / len(answers)
    else:
        recall = 0.0

    # F1分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def precision_recall_f1_multiple(predictions: List[str], answers: List[str]) -> Tuple[float, float, float]:
    """
    处理多个预测的情况
    """
    # 使用exact_match判断预测是否正确
    true_positives = 0
    matched_answers = set()

    # 计算真正例(TP)
    for pred in predictions:
        for ans in answers:
            if exact_match(pred, ans) and id(ans) not in matched_answers:
                true_positives += 1
                matched_answers.add(id(ans))
                break

    # 计算假正例(FP)
    false_positives = len(predictions) - true_positives

    # 计算假反例(FN)
    false_negatives = 0
    for ans in answers:
        matched = False
        for pred in predictions:
            if exact_match(pred, ans):
                matched = True
                break
        if not matched:
            false_negatives += 1

    # 计算精确率
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    # 计算召回率
    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    # 计算F1分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def mean_reciprocal_rank(predictions: str, answers: List[str]) -> float:
    """
    计算 Reciprocal Rank (RR, 倒数排名)。它是 Mean Reciprocal Rank (MRR) 对单个样本的贡献值。
    找到第一个正确答案的排名 (R)，返回 1/R。如果没有正确答案，返回 0。
    """
    if not predictions or not answers:
        return 0.0

    # 遍历预测结果，找到第一个匹配的正确答案
    for rank, prediction in enumerate(predictions, start=1):
        # 检查当前预测是否与任何一个正确答案匹配
        if any(exact_match(prediction, answer) for answer in answers):
            return 1.0 / rank

    return 0.0

def save_result2json(dataset_name, Hit, Accuracy, MRR, Recall, Precision, F1, model, method):
    results_data = {
        'dataset': dataset_name,
        'model': model,
        'method': method,
        'Hit@1': Hit,
        'Accuracy': Accuracy,
        'MRR': MRR,
        'Recall': Recall,
        'Precision': Precision,
        'F1': F1
    }
    with open('../results/{}_{}_{}_acc.json'.format(method, dataset_name, model), 'a', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

def check_wrong_result(method, dataset, model, data):
    with open('../wrong_sample/{}_{}_{}_error.json'.format(method, dataset, model), 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_content(s):
    matches = re.findall(r'\{(.*?)\}', s)
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    elif len(matches) >= 1:
        return matches[0]
    else:
        return 'NULL'