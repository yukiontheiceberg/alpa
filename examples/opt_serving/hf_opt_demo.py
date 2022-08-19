"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
from transformers import AutoTokenizer, FlaxOPTForCausalLM, OPTForCausalLM
from opt_serving.model.wrapper import get_model
import numpy as np

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

generate_params = {"do_sample": False, "num_beams": 1}

model = FlaxOPTForCausalLM.from_pretrained("facebook/opt-125m")#.to("cuda")

# Generate
prompt = [
    "Paris is the capital city of",
    # "Today is a good day and I'd like to",
    # "Computer Science studies the area of",
    # "University of California Berkeley is a public university"
]
input_ids = tokenizer(prompt, return_tensors="jax", padding="longest").input_ids#.to("cuda")

outputs = model.generate(input_ids=input_ids,
                         pad_token_id=tokenizer.pad_token_id,
                         max_length=64,
                         **generate_params)
# Print results

# outputs = model(input_ids)
print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs.sequences):
    print("{}: {}".format(i, tokenizer.decode(output,
                                              skip_special_tokens=True)))
    print(100 * '-')
