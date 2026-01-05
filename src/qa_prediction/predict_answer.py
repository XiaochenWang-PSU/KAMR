import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader
import time
# dist.init_process_group(backend='nccl')
import warnings
import pandas as pd
from datasets import Dataset
warnings.filterwarnings(
    "ignore",
    message=r"Both `max_new_tokens`.*`max_length`.*will take precedence.*",
    category=UserWarning
)
#def get_output_file(path, force=False):
#    if not os.path.exists(path) or force:
#        fout = open(path, "w")
#        return fout, []
#    else:
#        with open(path, "r") as f:
#            processed_results = []
#            for line in f:
#                try:
#                    results = json.loads(line)
#                except:
#                    raise ValueError("Error in line: ", line)
#                processed_results.append(results["id"])
#        fout = open(path, "a")
#        return fout, processed_results

def get_output_file(path, force=False):
        fout = open(path, "w")
        return fout, []

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    # --- convert to Dataset if needed ---
    if isinstance(qa_dataset, list):
        # ensure it's a list of dicts
        if all(isinstance(x, dict) for x in qa_dataset):
            qa_dataset = Dataset.from_list(qa_dataset)
        else:
            raise TypeError("qa_dataset list must contain dict elements only.")

    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample

    print(type(qa_dataset[0]))
    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)

    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )

    return qa_dataset

#def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
#    question_to_rule = dict()
#    for data in rule_dataset:
#        qid = data["id"]
#        predicted_paths = data["prediction"]
#        ground_paths = data["ground_paths"]
#        question_to_rule[qid] = {
#            "predicted_paths": predicted_paths,
#            "ground_paths": ground_paths,
#        }
#
#    def find_rule(sample):
#        qid = sample["id"]
#        sample["predicted_paths"] = []
#        sample["ground_paths"] = []
#        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
#        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
#        return sample  # TODO: ignore the sample with zero paths.
#
#    print(type(qa_dataset[0]))
#    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
#    if filter_empty:
#        qa_dataset = qa_dataset.filter(
#            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
#        )
#    return qa_dataset



#
#def prediction(data, processed_list, input_builder, model):
#
#
#    question = data["question"]
#    answer = data["answer"]
#    id = data["id"]
##    if id in processed_list:
##        return None
#    if model is None:
#        prediction = input_builder.direct_answer(data)
#        return {
#            "id": id,
#            "question": question,
#            "prediction": prediction,
#            "ground_truth": answer,
#            "input": question,
#        }
#    input = input_builder.process_input(data)
#    prediction = model.generate_sentence(input)
#    # print(input)
#    if prediction is None:
#        return None
#    result = {
#        "id": id,
#        "question": question,
#        "prediction": prediction,
#        "ground_truth": answer,
#        "input": input,
#    }
#    return result
    
def prediction(data_batch, processed_list, input_builder, model):
    results = []
    ids = [d["id"] for d in data_batch]
    questions = [d["question"] for d in data_batch]
    answers = [d["answer"] for d in data_batch]
    inputs = [input_builder.process_input(d) for d in data_batch]
    # inputs = input_builder.process_input_batch(data_batch, batch_size=128)
    predictions = model.generate_sentence(inputs)
    if predictions is None:
        return None

    for id, question, answer, input_text, pred_text in zip(ids, questions, answers, inputs, predictions):
        result = {
            "id": id,
            "question": question,
            "prediction": [pred_text],  # wrap in list to match ground_truth format
            "ground_truth": answer,
            "input": input_text,
        }
        # print(result["question"], result["prediction"], result["ground_truth"])
        # print('\n')

        results.append(result)
    return results  # list of dicts

def custom_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        batch_out[key] = [item[key] for item in batch]
    return batch_out

def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    if args.d in ["RoG-cwq", "RoG-webqsp"]:
        dataset = load_dataset(input_file, split=args.split)
    elif args.d == "gr_h":
        dataset = pd.read_json("GRBench/gr_h.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_2h":
        dataset = pd.read_json("PQ/2H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_3h":
        dataset = pd.read_json("PQ/3H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    # dataset = load_dataset(input_file, split=args.split)
    # dataset = dataset.select(range(10))
    
    

    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
    print("Load dataset from finished")
#    output_dir = os.path.join(
#        args.predict_path, args.d, args.model_name, args.split, rule_postfix, args.model_name, str(time.time())
#    )
    if args.rule_path:
        rule_name = os.path.basename(args.rule_path).split('.')[0]
        output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, rule_name, 
    )
    else:
        output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, "no_rule", 
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions_trunk.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)
    print(output_file)
    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        # dataset = dataset.select(range(10))
        res_list = prediction(dataset, processed_list, input_builder, model)
        if res_list is not None:
            for res in res_list:
                # print(json.dumps(res))
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
            fout.flush()
#        for data in tqdm(dataset):
#            res = prediction(data, processed_list, input_builder, model)
#            if res is not None:
#                if args.debug:
#                    print(json.dumps(res))
#                fout.write(json.dumps(res) + "\n")
#                fout.flush()
#    else:
#        batch_size = 4
#        for i in tqdm(range(0, len(dataset), batch_size)):
#            batch_data = dataset[i : i + batch_size]
#            # print("batch_data loaded")
#            res_list = prediction(batch_data, processed_list, input_builder, model)
#            # print(res_list)
#            for res in res_list:
#                if res is not None:
#                    if args.debug:
#                        print(json.dumps(res))
#                    fout.write(json.dumps(res) + "\n")
#                    fout.flush()
    fout.close()

    eval_result(output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    argparser.add_argument("--deepspeed_config", type=str, help="Path to DeepSpeed config file")

    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default=None
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
