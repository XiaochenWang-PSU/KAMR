from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
from tqdm import tqdm
import time


class Qwen3(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--model_path',
            type=str,
            help="Hugging Face model id or local path",
            # Options: 'Qwen/Qwen3-8B' (base) or 'Qwen/Qwen3-8B-Instruct'
            default='Qwen/Qwen3-8B'
        )
        parser.add_argument(
            '--max_new_tokens',
            type=int,
            help="max length of generated tokens",
            default=512
        )
        parser.add_argument(
            '--dtype',
            choices=['fp32', 'fp16', 'bf16'],
            default='bf16'
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            help="batch size for generation",
            default=8
        )
        parser.add_argument(
            '--enable_thinking',
            action='store_true',
            help="Enable Qwen3 thinking mode during chat template rendering."
        )
        parser.add_argument(
            '--trust_remote_code',
            action='store_true',
            help="Enable if the model repo requires custom code."
        )

    def __init__(self, args):
        self.args = args
        # conservative safety margin for context (adjust if you target a bigger context window)
        self.maximun_token = 4096 - 32
        self.tokenizer = None
        self.model = None
        self._think_end_id = None  # token id for </think>

    def load_model(self, **kwargs):
        # kept for API compatibility; not needed if prepare_for_inference() is used
        tok = AutoTokenizer.from_pretrained(**kwargs)
        return tok

    def tokenize(self, text):
        assert self.tokenizer is not None, "Call prepare_for_inference() first."
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        trust_remote_code = bool(getattr(self.args, "trust_remote_code", False))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            use_auth_token=True,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        
        # causal LMs: left padding + pad=eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"   # <-- change this

        # prefer the model's advertised max if available
        if hasattr(self.tokenizer, "model_max_length") and self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0:
            self.maximun_token = min(self.maximun_token, int(self.tokenizer.model_max_length) - 32)

        torch_dtype = self.DTYPE.get(self.args.dtype, None)
        if not torch.cuda.is_available():
            torch_dtype = torch.float32  # safer on CPU

        # load model (device_map='auto' uses accelerate under the hood if available)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch_dtype if torch_dtype is not None else "auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
            use_auth_token=True,
            **model_kwargs
        )
        self.model.eval()

        # cache </think> token id if present
        try:
            self._think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
            if isinstance(self._think_end_id, list):  # some tokenizers may return list
                self._think_end_id = self._think_end_id[0]
            if self._think_end_id == self.tokenizer.unk_token_id:
                self._think_end_id = None
        except Exception:
            self._think_end_id = None

    def _apply_chat_template_batch(self, prompts):
        """
        Build chat-formatted strings with (optional) thinking for a list of user prompts.
        """
        texts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.args.enable_thinking),
            )
            texts.append(text)
        return texts

    def _truncate_inputs(self, texts):
        """
        Pre-truncate by tokenizing then decoding back to string.
        """
        truncated = []
        for t in texts:
            tokens = self.tokenizer.tokenize(t)
            if len(tokens) > self.maximun_token:
                tokens = tokens[:self.maximun_token]
                t = self.tokenizer.convert_tokens_to_string(tokens)
            truncated.append(t)
        return truncated

    def _split_thought_and_answer(self, output_ids):
        """
        Split model output into (thinking_content, content) by </think> token id if available.
        Fallback: return ("", full_decoded_output).
        """
        if self._think_end_id is not None:
            # find last occurrence of </think>
            try:
                rev_index = output_ids[::-1].index(self._think_end_id)
                index = len(output_ids) - rev_index
                thinking_ids = output_ids[:index]
                answer_ids = output_ids[index:]
                thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")
                answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip("\n")
                return thinking_text, answer_text
            except ValueError:
                pass

        # fallback: try string-based split on decoded text
        decoded_full = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        marker = "</think>"
        if marker in decoded_full:
            # split at the last marker
            split_idx = decoded_full.rfind(marker) + len(marker)
            return decoded_full[:split_idx - len(marker)].strip("\n"), decoded_full[split_idx:].strip("\n")

        # no thinking segment found
        return "", self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    @torch.inference_mode()
    def generate_sentence(self, dataset):
        """
        dataset: list[str] (each a single prompt; you can concatenate RAG context yourself upstream)
        returns: list[str] (final 'content' without thinking)
        """
        assert self.model is not None and self.tokenizer is not None, "Call prepare_for_inference() first."

        res = []
        s_time = time.time()

        # 1) Build chat-formatted prompts with optional thinking
        chat_texts = self._apply_chat_template_batch(dataset)

        # 2) Pre-truncate defensively
        chat_texts = self._truncate_inputs(chat_texts)

        # 3) Batch over inputs for efficiency
        bs = max(1, int(self.args.batch_size))
        for i in tqdm(range(0, len(chat_texts), bs)):
            batch_texts = chat_texts[i:i+bs]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,          # now left-pads
                truncation=True,
                max_length=self.maximun_token,
            ).to(self.model.device)

            # greedy decoding; adjust if you want sampling
            gen = self.model.generate(
                **enc,
                max_new_tokens=int(self.args.max_new_tokens),
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # slice off the prompt part and parse thinking/content
            for j in range(gen.size(0)):
                output_ids = gen[j][len(enc.input_ids[j]):].tolist()
                _thinking, content = self._split_thought_and_answer(output_ids)
                res.append(content)

        print(f"Qwen3 generation time for {len(dataset)} inputs: {time.time() - s_time:.2f} s")
        return res
