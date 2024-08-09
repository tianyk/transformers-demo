# 文本生成任务
from transformers import pipeline

generator = pipeline("text-generation", device=0)
output = generator("Hello everyone. I'm a beginner in the world of cross-talk. My name is")

print(output)
# [{'generated_text': 'Hello everyone. I\'m a beginner in the world of cross-talk. My name is Darryl, which means "God\'s Hand". I like to put aside that I\'m not really a beginner. But I\'ve read and seen countless books'}]

# 使用 distilgpt2 模型
generator = pipeline("text-generation", model="distilgpt2", device=0)
# num_return_sequences 控制生成多少个不同的序列
# max_length 控制输出文本的总长度
output = generator("I'm a beginner in the world of cross-talk. My name is", max_length=50, num_return_sequences=2)
print(output)
# [{'generated_text': "I'm a beginner in the world of cross-talk. My name is Jeff Kline. I am a former teacher at the California Christian Academy, where I taught a range of classes and has taught for over a decade, and a fellow who has"}, {'generated_text': "I'm a beginner in the world of cross-talk. My name is Ryan, and this book teaches you:\n\n\nFitting Cross Communication to a Living Soul\n\nWhen you feel like cross communication is under your feet, don't be afraid"}]
