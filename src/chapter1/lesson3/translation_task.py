from transformers import pipeline

translation = pipeline("translation", device=0, model="Helsinki-NLP/opus-mt-fr-en")
output = translation("Bonjour, je m'appelle Darryl.")

print(output)