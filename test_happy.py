from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_steer_vector=True,
          enforce_eager=True, tensor_parallel_size=1, enable_chunked_prefill=False, max_model_len=16384)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(8, 26))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="/mnt/nw/home/m.yu/repos/EasySteer/vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt",
                                      scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(
    text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest("happy", 2, steer_vector_local_path="/mnt/nw/home/m.yu/repos/EasySteer/vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt",
                                   scale=2.0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
happy_output = llm.generate(
    text, steer_vector_request=happy_request, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)
