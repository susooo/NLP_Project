from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("alphahg/ke-t5-small-finetuned-paper")

model = T5ForConditionalGeneration.from_pretrained("alphahg/ke-t5-small-finetuned-paper")

input_ids = tokenizer("summarize: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))