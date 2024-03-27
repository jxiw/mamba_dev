from ssm.mambabyte  import ByteMambaLMHeadModel
from ssm.mambasubword import SubWordMambaLMHeadModel

from mamba_ssm.utils.generation2 import InferenceParams

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os
import torch.distributed as dist

from transformers import PretrainedConfig, PreTrainedModel
import random
import tensorflow_datasets as tfds

dist.init_process_group(init_method='env://', backend='nccl')
    
vocab_fname = '/home/jw2544/SSMWord/preprocessing/pg_subword_tokenizer'
encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)

# we change the test to validation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

mambabyte_hidden_size = 1792
mambabyte_num_hidden_layers = 48

# mambabyte
mambabyte_config = {
    "d_model": mambabyte_hidden_size,
    "fused_add_norm": True,
    "hidden_size": mambabyte_hidden_size,
    "n_layer": mambabyte_num_hidden_layers,
    "pad_vocab_size_multiple": 8,
    "residual_in_fp32": True,
    "rms_norm": True,
    "ssm_cfg": {},
    "vocab_size": 256,
    "pad_id": 0,
    "gen": True,
}

mambabyte_config = PretrainedConfig(**{**mambabyte_config, 'vocab_size': 256})
mambabyte = ByteMambaLMHeadModel(config=mambabyte_config, dtype=torch.bfloat16, device="cuda")
mambabyte= torch.nn.parallel.DistributedDataParallel(
    mambabyte,
    device_ids=[0],
    output_device=0,
)
mambabyte.load_state_dict(torch.load(f"/share/rush/pile/pg19_971M/model.pt", map_location=torch.device('cpu')))

mambabyte=mambabyte.module
mambabyte.eval()

prompt = '''
June 1st.  Having taken our leaves of Sir W. Batten and my Lady, who are
gone this morning to keep their Whitsuntide, Sir W. Pen and I and Mr.
Gauden by water to Woolwich, and there went from ship to ship to give
order for and take notice of their forwardness to go forth, and then to
Deptford and did the like, having dined at Woolwich with Captain Poole at
the tavern there.  From Deptford we walked to Redriffe, calling at the
half-way house, and there come into a room where there was infinite of new
cakes placed that are made against Whitsuntide, and there we were very
merry.  By water home, and there did businesses of the office. Among
others got my Lord's imprest of L1000 and Mr. Creed's of L10,000 against
this voyage their bills signed.  Having wrote letters into the country and
read some things I went to bed.

2nd.  Up and to my office, where '''

# I sat a h

def text_to_byte_tokens(text_to_test):
    text_to_test = text_to_test.replace('\n', '\r\n')
    input_ids = np.frombuffer(text_to_test.encode('utf-8'), dtype=np.uint8)
    input_ids = torch.from_numpy(input_ids)[None, :].long().cuda()
    return input_ids

def byte_tokens_to_text(byte_array):
    return bytes(byte_array).decode('utf-8').replace('\r\n', '\n')

prompt_tokens = text_to_byte_tokens(prompt)

output = mambabyte(prompt_tokens)
x = torch.argmax(output.logits, dim=-1)

print("x value:", x[:, -1])
print("x:", byte_tokens_to_text([x[0, -1]]))

# torchrun --nproc_per_node=1 --master_port=13398 test_byte_model.py > byte_verify.txt

verify_inference_params = InferenceParams(max_seqlen=2048, max_batch_size=1, draft_model=False)

prompt1 = '''
June 1st.  Having taken our leaves of Sir W. Batten and my Lady, who are
gone this morning to keep their Whitsuntide, Sir W. Pen and I and Mr.
Gauden by water to Woolwich, and there went from ship to ship to give
order for and take notice of their forwardness to go forth, and then to
Deptford and did the like, having dined at Woolwich with Captain Poole at
the tavern there.  From Deptford we walked to Redriffe, calling at the
half-way house, and there come into a room where there was infinite of new
cakes placed that are made against Whitsuntide, and there we were very
merry.  By water home, and there did businesses of the office. Among
others got my Lord's imprest of L1000 and Mr. Creed's of L10,000 against
this voyage their bills signed.  Having wrote letters into the country and
read some things I went to bed.

2nd.  Up and to my office, where '''

prompt2 = 'I sat '
prompt3 = 'an '
prompt4 = 'h'

prompt_tokens1 = text_to_byte_tokens(prompt1)
prompt_tokens2 = text_to_byte_tokens(prompt2)
prompt_tokens3 = text_to_byte_tokens(prompt3)
prompt_tokens4 = text_to_byte_tokens(prompt4)

output = mambabyte(prompt_tokens1, inference_params=verify_inference_params)
x1 = torch.argmax(output.logits, dim=-1)

print("restart:restart:restart:")
print(verify_inference_params.key_value_memory_dict[0])

print("x1 value:", x1[:, -1])
print("x1:", byte_tokens_to_text([x1[0, -1]]))

output = mambabyte(prompt_tokens2, inference_params=verify_inference_params)
x2 = torch.argmax(output.logits, dim=-1)

print("x2 value:", x2[:, -1])
print("x2:", byte_tokens_to_text([x2[0, -1]]))

output = mambabyte(prompt_tokens3, inference_params=verify_inference_params)
x3 = torch.argmax(output.logits, dim=-1)

print("x3 value:", x3[:, -1])
print("x3:", byte_tokens_to_text([x3[0, -1]]))

output4 = mambabyte(prompt_tokens4, inference_params=verify_inference_params)
x4 = torch.argmax(output4.logits, dim=-1)

print("x4 value:", x4[:, -1])
print("x4:", byte_tokens_to_text([x4[0, -1]]))