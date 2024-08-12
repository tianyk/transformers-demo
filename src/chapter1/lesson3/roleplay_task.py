import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Tuple

def parse_chatml(chatml_content):
    pattern = re.compile(r'<\|im_start\|>(?:\s+)([a-zA-Z]+)(?:\s+)(.*?)<\|im_end\|>', re.DOTALL)
    messages = []
    for match in pattern.finditer(chatml_content):
        role, content = match.groups()
        messages.append({"role": role, "content": content.strip()})
    return messages

def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_name_or_path = "ClosedCharacter/Peach-9B-8k-Roleplay"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, 
        trust_remote_code=True, device_map="auto")
    return model, tokenizer

def main():
    print("欢迎来到角色扮演游戏！")
    print("模型加载中...")
    model, tokenizer = load_model()
    role = input("请输入角色：").strip()

    # 上下文对话
    messages = []
    messages.append({"role": "system", "content": "你是%s" % role})
    messages.append({"role": "use", "content": "你好，你是谁？"})

    try: 
        print("游戏开始！")
        while True: 
            input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors="pt",)
            output = model.generate(
                inputs=input_ids.to("mps"), 
                temperature=0.3, 
                top_p=0.5, 
                no_repeat_ngram_size=6,
                repetition_penalty=1.1,
                max_new_tokens=512)
            response = tokenizer.decode(output[0])

            messages = parse_chatml(response)
            print("输出：%s" % messages[-1]['content'])

            # 用户输入
            message = input("请输入消息：").strip()
            messages.append({"role": "user", "content": message})
    except KeyboardInterrupt:
        print("游戏结束！")
    except Exception as e:
        print("未知异常：%s" % e)

if __name__ == "__main__":
    main()