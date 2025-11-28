from gguf import GGUFReader
import torch
VECTOR_PATH = 'vectors/happy_diffmean.gguf'
PT_VECTOR_PATH = 'vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt'

reader = GGUFReader(VECTOR_PATH)
tensor_names = [t.name for t in reader.tensors]
print("Available tensors:", tensor_names)
print(len(tensor_names))

vector = torch.load(PT_VECTOR_PATH)
print(vector.shape)
