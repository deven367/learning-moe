# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")

test_str = "Hello, how are you?"
tokens = tokenizer(test_str, return_tensors="pt")

logits = model(**tokens).logits
print(logits)