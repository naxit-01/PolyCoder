import torch

# Kontrola, zda je dostupné CUDA (což znamená, že máte GPU)
if torch.cuda.is_available():  
  device = torch.device("cuda")
  print('Running on GPU:', torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print('Running on CPU')

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.23.0")

tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")

# Přesunutí modelu na GPU
model = model.to(device)

prompt = '''def binarySearch(arr, left, right, x):
    mid = (left +'''
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Přesunutí vstupních dat na GPU
input_ids = input_ids.to(device)

result = model.generate(input_ids, max_length=50, num_beams=4, num_return_sequences=4)
for res in result:
    print(tokenizer.decode(res))

print("end")
