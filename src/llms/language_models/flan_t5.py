#from transformers import pipeline, AutoModel, AutoTokenizer
#import torch
#from .base_language_model import BaseLanguageModel
#from tqdm import tqdm
#import time 
#import warnings
#warnings.filterwarnings(
#    "ignore",
#    message=r"Both `max_new_tokens`.*`max_length`.*will take precedence.*",
#    category=UserWarning
#)
#
#class FlanT5(BaseLanguageModel):
#    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
#    @staticmethod
#    def add_args(parser):
#        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='google/flan-t5-xl')
#        parser.add_argument('--max_new_tokens', type=int, help="max length", default=128)
#        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
#
#    def __init__(self, args):
#        self.args = args
#        self.maximun_token = 512 - 5
#        
#    def load_model(self, **kwargs):
#        model = AutoModel.from_pretrained(**kwargs)
#        return model
#    
#    def tokenize(self, text):
#        return len(self.tokenizer.tokenize(text))
#    
#    def prepare_for_inference(self, **model_kwargs):
#        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,
#        use_fast=False)
#        # self.generator = pipeline("text2text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
##        if "max_length" in model_kwargs:
##            del model_kwargs["max_length"]
#        model_kwargs["max_length"] = 2048
#        self.generator = pipeline("text2text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
#    
#    @torch.inference_mode()
##    def generate_sentence(self, llm_input):
##        outputs = self.generator(llm_input)
##        return outputs[0]['generated_text'] # type: ignore
#
##    def generate_sentence(self, dataset):
##        res = []
##        for output in tqdm(self.generator(dataset)):
##            
##            res.append(output['generated_text'])
##            # res.append += [output[0]['generated_text'] for output in outputs]
##
##        return res
#
#
#    def generate_sentence(self, dataset): 
#        res = []
#        s_time = time.time()
#        for output in tqdm(self.generator(dataset, max_new_tokens=self.args.max_new_tokens)):
#            if isinstance(output, list):
#                res.extend([item['generated_text'] for item in output])
#            else:
#                res.append(output['generated_text'])
#        print(time.time() - s_time)
#        return res  # return list of strings


from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .base_language_model import BaseLanguageModel
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Both `max_new_tokens`.*`max_length`.*will take precedence.*",
    category=UserWarning
)


class FlanT5(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--model_path',
            type=str,
            help="HUGGING FACE MODEL or model path",
            default='google/flan-t5-xl'
        )
        parser.add_argument(
            '--max_new_tokens',
            type=int,
            help="max length of generated tokens",
            default=128
        )
        parser.add_argument(
            '--dtype',
            choices=['fp32', 'fp16', 'bf16'],
            default='fp16'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            help="batch size for generation",
            default=32
        )

    def __init__(self, args):
        self.args = args
        self.maximun_token = 512 - 5
        self.tokenizer = None
        self.generator = None

    def load_model(self, **kwargs):
        # kept for compatibility, though not used in prepare_for_inference
        model = AutoModel.from_pretrained(**kwargs)
        return model

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        # use fast tokenizer for speed
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            use_fast=True
        )

        # ensure pad_token exists (T5 usually already has it, but we guard anyway)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # set model_max_length for input truncation
        self.tokenizer.model_max_length = self.maximun_token

        # do NOT set model_kwargs["max_length"] here to avoid the warning
        # generation will be controlled by max_new_tokens + truncation=True
        if "max_length" in model_kwargs:
            del model_kwargs["max_length"]

        torch_dtype = self.DTYPE.get(self.args.dtype, None)
        if not torch.cuda.is_available():
            torch_dtype = torch.float32

        num_gpus = torch.cuda.device_count()
        torch_dtype = self.DTYPE.get(self.args.dtype, None)
        if not torch.cuda.is_available():
            torch_dtype = torch.float32
        else:
            if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                torch_dtype = torch.float16
        
        if torch.cuda.is_available() and num_gpus > 1:
            # multi-GPU, keep device_map="auto"
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model_path,
                torch_dtype=torch_dtype,
                device_map="auto",   # <--- multi-GPU
                **model_kwargs
            )
            # reduce KV cache pressure
            model.config.use_cache = False
        
            self.generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                batch_size=self.args.batch_size,
                # IMPORTANT: no `device=` when using device_map="auto"
            )
        else:
            # CPU
            self.generator = pipeline(
                "text2text-generation",
                model=self.args.model_path,
                tokenizer=self.tokenizer,
                device=-1,
                model_kwargs=model_kwargs,
                torch_dtype=torch_dtype,
                batch_size=self.args.batch_size,
            )

    @torch.inference_mode()
    def generate_sentence(self, dataset):
        assert self.generator is not None, "Call prepare_for_inference() first."
    
        res = []
        s_time = time.time()
        pbar = tqdm(total=len(dataset), desc="Flan-T5 generating")
    
        for output in self.generator(
            dataset,
            max_new_tokens=self.args.max_new_tokens,
            truncation=True,
            num_beams=1,     # <--- disable beam search
            do_sample=False, # <--- deterministic, lighter decode
        ):
            if isinstance(output, list):
                res.extend([item['generated_text'] for item in output])
                pbar.update(len(output))
            else:
                res.append(output['generated_text'])
                pbar.update(1)
    
        pbar.close()
        print(f"Flan-T5 generation time for {len(dataset)} inputs: {time.time() - s_time:.2f} s")
        return res
