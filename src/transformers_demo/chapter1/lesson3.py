from transformers import pipeline

classifier = pipeline("sentiment-analysis", device=0)
output = classifier("I've been waiting for a HuggingFace course my whole life.")

print(output)