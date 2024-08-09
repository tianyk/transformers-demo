# 填空任务
from transformers import pipeline

# 支持 fill-mask 任务的模型列表：https://huggingface.co/models?pipeline_tag=fill-mask
unmasker = pipeline("fill-mask", device=0)
# top_k 控制返回的结果数量，这里返回两个结果
output = unmasker("Hello, my name is <mask>.", top_k=2)

print(output)
# [{'score': 0.005134810693562031, 'token': 1573, 'token_str': ' Chris', 'sequence': 'Hello, my name is Chris.'}, {'score': 0.004900422878563404, 'token': 6939, 'token_str': ' Laura', 'sequence': 'Hello, my name is Laura.'}]