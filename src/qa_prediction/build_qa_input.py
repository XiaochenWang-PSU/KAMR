import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
import json
from math import ceil
from gen_rule_path import TwoTowerModel




class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    RETRIEVAL_MCQ_INSTRUCTION = """Based on the retrieved facts, please answer the given question. Please select the answers from the given choices and return the answers only."""
    RETRIEVAL_SAQ_INSTRUCTION = """Based on the retrieved facts, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Retrieved Facts:\n{context}\n\n"""
    # GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(self, prompt_path, add_rule = False, use_true = False, cot = False, explain = False, use_random = False, each_line = False, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
        
        
        cache_file = "cache_ours"
               
                # Initialize disk cache
        self.cache_file = cache_file
#        if os.path.exists(self.cache_file):
#            with open(self.cache_file, "r") as f:
#                self.cache = json.load(f)
#        else:
#            self.cache = {}
        
        print("builder imported")
        
    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
    def retrieve_top_k(self, query_text, topk=20):
        with torch.no_grad():
            q_vec = self.retriever_model.encode_queries([query_text])

            sims = cosine_similarity(q_vec, self.doc_vecs)
            sorted_indices = torch.argsort(sims, descending=True).tolist()[:topk]
        top_docs = [self.kid_to_doc[self.doc_ids[idx]] for idx in sorted_indices]
        return "\n".join(top_docs)
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
        
    def retrieve_top_k_from_subgraph_batch(self, query_texts, subgraph_batch, topk=20):
        # query_texts: List[str]
        # subgraph_batch: List[List[Tuple[str, str, str]]] for each question
    
        results = []
        for query_text, subgraph_triplets in zip(query_texts, subgraph_batch):
            subgraph_docs = [" | ".join(triplet) for triplet in subgraph_triplets]
    
            with torch.no_grad():
                doc_vecs = self.retriever_model.encode_keys(subgraph_docs)
                q_vec = self.retriever_model.encode_queries([query_text])  # shape (1, d)
                sims = cosine_similarity(q_vec, doc_vecs)
    
                if sims.ndim == 0:
                    sims = sims.unsqueeze(0)
                elif sims.ndim == 2:
                    sims = sims.squeeze(0)
    
                topk_actual = min(topk, sims.size(0))
                top_scores, top_indices = torch.topk(sims, k=topk_actual)
    
            top_triplets = [subgraph_docs[idx] for idx in top_indices.tolist()]
            results.append("\n".join(top_triplets))
      
        return results  # List[str]
#    def retrieve_top_k_from_subgraph(self, query_text, subgraph_triplets, topk=50):
#        # Prepare triplet texts in the same format as retriever training
#        subgraph_docs = [" | ".join(triplet) for triplet in subgraph_triplets]
#    
#        # Encode subgraph triplets with key encoder
#        with torch.no_grad():
#            doc_vecs = self.retriever_model.encode_keys(subgraph_docs)
#            q_vec = self.retriever_model.encode_queries([query_text])
#    
#            # Compute cosine similarity ? shape (1, num_docs)
#            sims = cosine_similarity(q_vec, doc_vecs)# .squeeze(0)  # ? shape (num_docs,)
#            if sims.ndim == 0:
#                sims = sims.unsqueeze(0)
#            elif sims.ndim == 2:
#                sims = sims.squeeze(0)
#            # Get top-k indices and scores
#            topk = min(topk, sims.size(0))
#            top_scores, top_indices = torch.topk(sims, k=topk)
#    
#        # Build list with triplet + score
#        top_triplets_with_scores = [
#            # f"{subgraph_docs[idx]} (score: {top_scores[i].item():.4f})"
#            f"{subgraph_docs[idx]}"
#            for i, idx in enumerate(top_indices.tolist())
#        ]
#    
#        # Return as joined context string
#        return "\n".join(top_triplets_with_scores)
#    def retrieve_top_k_from_subgraph(self, query_text, subgraph_triplets, topk=50):
#        # Build string key (convert triplets to string so JSON can store it)
#        triplet_key = [tuple(triplet) for triplet in subgraph_triplets]
#        cache_key = json.dumps({"query": query_text, "triplets": triplet_key})
#
#        if cache_key in self.cache:
#            return self.cache[cache_key]
#
#        subgraph_docs = [" | ".join(triplet) for triplet in subgraph_triplets]
#
#        with torch.no_grad():
#            doc_vecs = self.retriever_model.encode_keys(subgraph_docs)
#            q_vec = self.retriever_model.encode_queries([query_text])
#
#            sims = cosine_similarity(q_vec, doc_vecs)
#            if sims.ndim == 0:
#                sims = sims.unsqueeze(0)
#            elif sims.ndim == 2:
#                sims = sims.squeeze(0)
#
#            topk = min(topk, sims.size(0))
#            top_scores, top_indices = torch.topk(sims, k=topk)
#
#        top_triplets_with_scores = [f"{subgraph_docs[idx]}" for i, idx in enumerate(top_indices.tolist())]
#        result = "\n".join(top_triplets_with_scores)
#
#        # Store in cache and save to disk
#        self.cache[cache_key] = result
#        self._save_cache()
#
#        return result
    def process_input_batch(self, question_dicts, batch_size=64):
        # Auto-wrap if a single dict (Case B)

            # Assume dict of lists format, like {"question": [...], "graph": [...], "choices": [...]}
            questions_all = question_dicts["question"]
#            choices_all = question_dicts["choices"]
#            questions_all = [
#    q + ("\n " + a if a else "") 
#    for q, a in zip(questions_all, choices_all)
#]
            graphs_all = question_dicts["graph"]
            choices_all = question_dicts["choices"]
            total = len(questions_all)
    
            prompts = []
            for i in tqdm(range(0, total, batch_size)):
                questions = [
                    q + ('?' if not q.endswith('?') else '') 
                    for q in questions_all[i:i+batch_size]
                ]
                subgraphs = graphs_all[i:i+batch_size]
                choices_batch = choices_all[i:i+batch_size]
                
                query = [
    q + ("\n " + a if a else "") 
    for q, a in zip(questions, choices_batch)]
                
                contexts = self.retrieve_top_k_from_subgraph_batch(query, subgraphs)
                # contexts = ['' for i in questions]
                for q, choice, context in zip(questions, choices_batch, contexts):
                    input = self.QUESTION.format(question=q)
                    instruction = self.RETRIEVAL_MCQ_INSTRUCTION if choice else self.RETRIEVAL_SAQ_INSTRUCTION
                    if choice:
                        input += self.CHOICES.format(choices='\n'.join(choice))
    
                    if self.cot: instruction += self.COT
                    if self.explain: instruction += self.EXPLAIN
                    if self.each_line: instruction += self.EACH_LINE
                    
                    # prompt = input
                    prompt = self.prompt_template.format(
                        instruction=instruction,
                        input=self.GRAPH_CONTEXT.format(context=context) + input
                    )
                    prompts.append(prompt)
                    # print(prompt)
            #print(prompts[:10])
            return prompts
    
#        else:
#            # Original behavior (list of dicts)
#            return self._process_input_batch_from_list(question_dicts, batch_size)


#    def process_input(self, question_dict):
#
#        question = question_dict['question']
#        # return question
#        # print(question_dict['graph'])
#        if not question.endswith('?'):
#            question += '?'
#
#        subgraph_triplets = question_dict['graph']
#        context = self.retrieve_top_k_from_subgraph(question, subgraph_triplets)
#
#        # context = self.retrieve_top_k(question)  # ?? top-10 retrieved facts as context
#
#
#        input = self.QUESTION.format(question=question)
#
#        # MCQ
#        if len(question_dict['choices']) > 0:
#            choices = '\n'.join(question_dict['choices'])
#            input += self.CHOICES.format(choices=choices)
#            instruction = self.RETRIEVAL_MCQ_INSTRUCTION#  if self.add_rule else self.MCQ_INSTRUCTION
#        # SAQ
#        else:
#            instruction = self.RETRIEVAL_SAQ_INSTRUCTION#  if self.add_rule else self.SAQ_INSTRUCTION
#
#        if self.cot:
#            instruction += self.COT
#        if self.explain:
#            instruction += self.EXPLAIN
#        if self.each_line:
#            instruction += self.EACH_LINE
#
#        other_prompt = self.prompt_template.format(instruction=instruction, input=self.GRAPH_CONTEXT.format(context=context) + input)
#        input = self.prompt_template.format(instruction=instruction, input=self.GRAPH_CONTEXT.format(context=context) + input)
#        # print(input)
#        # return input
#        return input
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']

        
        if not question.endswith('?'):
            question += '?'
        # return question

        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
            
        input = self.QUESTION.format(question = question)
        # MCQ
        if 'choices' in question_dict.keys():
            if len(question_dict['choices']) > 0:
                choices = '\n'.join(question_dict['choices'])
                input += self.CHOICES.format(choices = choices)
                if self.add_rule:
                    instruction = self.MCQ_RULE_INSTRUCTION
                else:
                    instruction = self.MCQ_INSTRUCTION
            # SAQ
            else:
                if self.add_rule:
                    instruction = self.SAQ_RULE_INSTRUCTION
                else:
                    instruction = self.SAQ_INSTRUCTION
        else:
                if self.add_rule:
                    instruction = self.SAQ_RULE_INSTRUCTION
                else:
                    instruction = self.SAQ_INSTRUCTION
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            # context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            # print(context)
            # print(question_dict['predicted_paths'][0][0])
            # print([ '|'.join(i[0][0]) for i in question_dict['predicted_paths'][:1]])
            context = ''
            if not question_dict['predicted_paths']:
                return input
            if isinstance(question_dict['predicted_paths'][0][0], list):  # It's a list of paths (List[List[List[str]]]):
                
                total_triplets = 0
                for path in question_dict['predicted_paths']:#[:50]:
                    for triple in path:
                        context += ' | '.join(triple) + '\n'
                        total_triplets += 1
                        if total_triplets > 50:
                            break
                    context += '\n'  # extra newline between paths
                    if total_triplets > 50:
                            break
    
                # context = [ '|'.join(i[0]) for i in question_dict['predicted_paths'][:50]]
            else:
                for triple in question_dict['predicted_paths']:#[:30]:
                    context += ' | '.join(triple)
                    context += '\n'
                # context = [ '\n'.join(i.join('|')) for i in question_dict['predicted_paths']]
            input = self.GRAPH_CONTEXT.format(context = context) + input
        # print(self.add_rule)
        input = self.prompt_template.format(instruction = instruction, input = input)
        # print(input)
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
            