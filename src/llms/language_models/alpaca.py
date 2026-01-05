from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
import time 
from tqdm import tqdm
class Alpaca(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='/nfsdata/shared/llm-models/alpaca/7B/')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = 512 - 100
    
    def load_model(self, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        return model
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=False)
        # self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
        # self.generator = pipeline("text-generation",  max_new_tokens=self.args.max_new_tokens, model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None), batch_size=2)
        self.generator = pipeline("text-generation",  model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
    @torch.inference_mode()
    def generate_sentence(self, dataset): 
        res = []
        s_time = time.time()
        for output in tqdm(self.generator(dataset, return_full_text=False, max_new_tokens=self.args.max_new_tokens)):
            if isinstance(output, list):
                res.extend([item['generated_text'] for item in output])
            else:
                res.append(output['generated_text'])
        print(time.time() - s_time)
        return res  # return list of strings

#    def generate_sentence(self, llm_input):
#        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
#        return outputs[0]['generated_text'] # type: ignore