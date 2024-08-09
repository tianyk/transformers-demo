# 语气分析任务
# see https://huggingface.co/learn/nlp-course/zh-CN/chapter1/3?
from transformers import pipeline

# 这里 sentiment-analysis 为预训练模型的任务名称，他主要用来分析句子的情绪。
# 他使用的是 distilbert/distilbert-base-uncased-finetuned-sst-2-english 模型
# 等同于 pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english") 
# 模型会自动下载到本地缓存目录，如果已经下载过，就不会再次下载。在 macOS 上，缓存目录为 .cache/huggingface/hub 目录下。
# device=0 表示使用第一个 GPU 设备。在Apple Silicon 芯片上，还可以传 device="mps"，
# 他表示使用 Apple的MPS(Metal Performance Shaders) 框架。
classifier = pipeline("sentiment-analysis", device=0)
output = classifier("I've been waiting for a HuggingFace course my whole life.")

print(output)
# [{'label': 'POSITIVE', 'score': 0.9598048329353333}]

output = classifier("I hate this so much!")
print(output)
# [{'label': 'NEGATIVE', 'score': 0.9994558691978455}]