# 实体识别任务
from transformers import pipeline

#  grouped_entities=True 以告诉pipeline将对应于同一实体的句子部分重新组合在一起：
# 这里模型正确地将“Hugging”和“Face”分组为一个组织，即使名称由多个词组成。
ner = pipeline("ner", device=0, grouped_entities=True)
output = ner("Hugging Face is a French company based in New-York.")

print(output)