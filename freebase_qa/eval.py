from utils.eval_utils import *
from utils.file_utils import FileUtils
import os, sys
os.chdir(sys.path[0])  #使用文件所在目录

def eval_em(dataset, output_file, model_name, method, constraints_refuse=True):
    model_stem = model_name.split("/")[-1]
    # load json dataset
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(dataset, output_file)

    # load pkl dataset
    # ground_truth_datas, question_string, output_datas = load_pkl_dataset(dataset, output_file)

    acc_list = []
    hit_list = []
    mrr_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for data in output_datas:
        # load json dataset
        answers = align(dataset, question_string, data, ground_truth_datas)

        # load pkl dataset
        # answers = pkl_align(question_string, data, ground_truth_datas)

        results = data['results']
        if check_string(results):
            response = clean_results(results)
        else:
            response = results
            if constraints_refuse and check_string(response):
                continue
        hit = eval_hit(response, answers)
        acc = eval_acc(response, answers)
        mrr = mean_reciprocal_rank(response, answers)
        precision, recall, f1 = precision_recall_f1(response, answers)
        hit_list.append(hit)
        acc_list.append(acc)
        mrr_list.append(mrr)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    Hit = sum(hit_list) * 100 / len(hit_list)
    Accuracy = sum(acc_list) * 100 / len(acc_list)
    MRR = sum(mrr_list) * 100 / len(mrr_list)
    Precision = sum(precision_list) * 100 / len(precision_list)
    Recall = sum(recall_list) * 100 / len(recall_list)
    F1 = sum(f1_list) * 100 / len(f1_list)
    print("Hit: " + str(Hit))
    print("Accuracy: " + str(Accuracy))
    print("MRR: " + str(MRR))
    print("Precision: " + str(Precision))
    print("Recall: " + str(Recall))
    print("F1: " + str(F1))

    save_result2json(dataset, Hit, Accuracy, MRR, Precision, Recall, F1, model_stem, method)

if __name__ == '__main__':
    dataset = 'webqsp'
    model = 'gpt-4o-mini'
    method = 'rage'
    prune_tools = 'llm'  # 'llm' or 'sbert'
    json_file = f"../outputs/{method}_{dataset}_{model}_{prune_tools}.json"
    jsonl_file = f"../outputs/{method}_{dataset}_{model}_{prune_tools}.jsonl"

    FileUtils.jsonl2json(jsonl_file, json_file)
    eval_em(dataset, json_file, model, method)
    