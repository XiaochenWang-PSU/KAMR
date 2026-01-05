
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
api_key="EMPTY")
from dotenv import load_dotenv
import os
import time
from .start_fastchat_api import start_fastchat_api
class LLMProxy(object):

    @staticmethod
    def regist_args(parser):
        parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf') # Llama-2-7b-chat-hf
        parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
        parser.add_argument("--conv_template", type=str, default="llama-2")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--disable_auto_start", action="store_true")

    def __init__(self, args) -> None:
        self.args = args
        if "gpt-4" in args.model_name or "gpt-3.5" in args.model_name:
            # Load key for OpenAI API
            load_dotenv()
            # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.getenv("OPENAI_ORG"))'
            # openai.organization = os.getenv("OPENAI_ORG")
        else:
            # Use local API
            if not args.disable_auto_start:
                start_fastchat_api(args.model_name, args.model_path, args.conv_template, args.host, args.port)
            # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=f"http://{args.host}:{args.port}/v1")'
            # openai.api_base = f"http://{args.host}:{args.port}/v1"

    @staticmethod
    def query(message, model_name, timeout=60, max_retry=3):
        '''
        Query ChatGPT API
        :param message:
        :return:
        '''
        retry = 0
        while True:
            try:
                response = client.chat.completions.create(model=model_name,
                messages=[{"role": "user", "content": message}],
                request_timeout=timeout)
                result = response.choices[0].message.content.strip()
                return result
            except Exception as e:
                print(e)
                retry += 1
                if retry >= max_retry:
                    raise e
                time.sleep(30)
                continue