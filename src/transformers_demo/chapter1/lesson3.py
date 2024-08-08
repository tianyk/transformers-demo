from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)
output = classifier("I've been waiting for a HuggingFace course my whole life.")

print(output)
# [{'label': 'POSITIVE', 'score': 0.9598048329353333}]