#from transformers import pipeline, AutoTokenizer
#import torch
#from .base_language_model import BaseLanguageModel
#from transformers import LlamaTokenizer
#from transformers.pipelines.pt_utils import KeyDataset
#from tqdm import tqdm
#import time
#class Llama(BaseLanguageModel):
#    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
#    @staticmethod
#    def add_args(parser):
#        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
#        parser.add_argument('--max_new_tokens', type=int, help="max length", default=128)
#        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
#
#
#    def __init__(self, args):
#        self.args = args
#        self.maximun_token = 4096 - 32
#        
#    def load_model(self, **kwargs):
#        model = LlamaTokenizer.from_pretrained(**kwargs)
#        return model
#    
#    def tokenize(self, text):
#        return len(self.tokenizer.tokenize(text))
#    
#    def prepare_for_inference(self, **model_kwargs):
#        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_auth_token=True, 
#        use_fast=False)
#        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#        model_kwargs.update({'use_auth_token': True})
#       # self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
#        # self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None), batch_size=16)
#        self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
#    def truncate_input(self, text):
#        tokens = self.tokenizer.tokenize(text)
#        if len(tokens) > self.maximun_token:
#            tokens = tokens[:self.maximun_token]
#        return self.tokenizer.convert_tokens_to_string(tokens)
#    @torch.inference_mode()
#    def generate_sentence(self, dataset): 
#        res = []
#        s_time = time.time()
#        inputs = [self.truncate_input(text) for text in dataset]
#    
#        for output in tqdm(self.generator(inputs, return_full_text=False, max_new_tokens=self.args.max_new_tokens)):
#            if isinstance(output, list):
#                res.extend([item['generated_text'] for item in output])
#            else:
#                res.append(output['generated_text'])
#    
#        print(time.time() - s_time)
#        return res
#    
##    def generate_sentence(self, llm_input):
##        # print(llm_input)
##        # print(type(llm_input))
##        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
##        return outputs[0]['generated_text'] # type: ignore
#




from transformers import pipeline, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import time


class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--model_path',
            type=str,
            help="HUGGING FACE MODEL or model path",
            default='meta-llama/Llama-2-7b-chat-hf'
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
            default=8
        )

    def __init__(self, args):
        self.args = args
        # maximum number of input tokens (context window minus safety margin)
        self.maximun_token = 4096 - 32
        self.tokenizer = None
        self.generator = None

    def load_model(self, **kwargs):
        # kept for compatibility, though not used in prepare_for_inference
        model = LlamaTokenizer.from_pretrained(**kwargs)
        return model

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        # use fast tokenizer for speed
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            use_auth_token=True,
            use_fast=True
        )

        # LLaMA tokenizer typically has no pad token; set it to eos so batching works
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # set max length for input truncation
        self.tokenizer.model_max_length = self.maximun_token

        model_kwargs.update({'use_auth_token': True})

        torch_dtype = self.DTYPE.get(self.args.dtype, None)
        if not torch.cuda.is_available():
            # on CPU, fp32 is safer
            torch_dtype = torch.float32

        num_gpus = torch.cuda.device_count()

        # Single-GPU / CPU / Multi-GPU handling
        if torch.cuda.is_available() and num_gpus >= 1:
            if num_gpus > 1:
                # Multi-GPU: use accelerate's device_map="auto"
                self.generator = pipeline(
                    "text-generation",
                    model=self.args.model_path,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    model_kwargs=model_kwargs,
                    torch_dtype=torch_dtype,
                    batch_size=self.args.batch_size,
                )
            else:
                # Single GPU: pin to cuda:0 for simplicity
                self.generator = pipeline(
                    "text-generation",
                    model=self.args.model_path,
                    tokenizer=self.tokenizer,
                    device=0,
                    model_kwargs=model_kwargs,
                    torch_dtype=torch_dtype,
                    batch_size=self.args.batch_size,
                )
        else:
            # CPU-only
            self.generator = pipeline(
                "text-generation",
                model=self.args.model_path,
                tokenizer=self.tokenizer,
                device=-1,
                model_kwargs=model_kwargs,
                torch_dtype=torch_dtype,
                batch_size=self.args.batch_size,
            )

    def truncate_input(self, text):
        # You can keep this manual truncation or rely on truncation=True in generate();
        # keeping it here to preserve original behavior.
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.maximun_token:
            tokens = tokens[:self.maximun_token]
        return self.tokenizer.convert_tokens_to_string(tokens)

    @torch.inference_mode()
    def generate_sentence(self, dataset):
        """
        dataset: list of strings
        Returns: list of generated strings
        """
        assert self.generator is not None, "Call prepare_for_inference() first."

        res = []
        s_time = time.time()

        # Optional: pre-truncate to be extra safe
        inputs = [self.truncate_input(text) for text in dataset]

        # batching is handled inside the pipeline via batch_size set in prepare_for_inference
        for output in tqdm(
            self.generator(
                inputs,
                return_full_text=False,
                max_new_tokens=self.args.max_new_tokens,
                truncation=True,
                do_sample=False,      # <--- force greedy
                temperature=1.0,      # (won't matter if do_sample=False)
                top_p=1.0,
                top_k=0,
            )
        ):
            if isinstance(output, list):
                res.extend([item['generated_text'] for item in output])
            else:
                res.append(output['generated_text'])

        print(f"LLaMA generation time for {len(dataset)} inputs: {time.time() - s_time:.2f} s")
        return res
