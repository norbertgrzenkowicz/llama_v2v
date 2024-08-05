from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from typing import List, Dict
import init_models

# sequences = pipeline(
#     'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     truncation = True,
#     max_length=400,
# )

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")


def define_message(
    who_is_llama: str = "You are a Gym bro which always respond normally but ends sentences with motivation speech.",
    message: str = "Who are You?",
):
    return [
        {"role": "system", "content": who_is_llama},
        {"role": "user", "content": message},
    ]


def inference(pipeline, messages: List[Dict[str, str]]):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][-1]


if __name__ == "__main__":
    notebook_login()

    llama = init_models().llama
    llama_output = inference(llama, define_message())

    print(llama_output)
