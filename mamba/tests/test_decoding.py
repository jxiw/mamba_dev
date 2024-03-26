from ssm.mambabyte  import ByteMambaLMHeadModel
from ssm.mambasubword import SubWordMambaLMHeadModel

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

mambasubword_hidden_size = 1792
mambasubword_num_hidden_layers = 48

# small mamba subword model
mambasubword_config = {
    "d_model": 1024,
    "fused_add_norm": True,
    "hidden_size": mambasubword_hidden_size,
    "n_layer": mambasubword_num_hidden_layers,
    "pad_vocab_size_multiple": 8,
    "residual_in_fp32": True,
    "rms_norm": True,
    "ssm_cfg": {},
    "vocab_size": 32070,
    "pad_id": 0,
    "gen": True,
}
mambasubword_config = PretrainedConfig(**{**mambasubword_config, 'vocab_size': 32070})
mambasubword = SubWordMambaLMHeadModel(config=mambasubword_config, dtype=torch.bfloat16, device="cuda")
mambasubword = torch.nn.parallel.DistributedDataParallel(
    mambasubword,
    device_ids=[0],
    output_device=0,
)
mambasubword.load_state_dict(torch.load(f"/home/jw2544/SSMWord/train/mamba_h1024_l48_2048_subword_small/checkpoint/step-51000/model.pt", map_location=torch.device('cpu')))  

# large mamba subword model
# mamba_config = {
#     "d_model": 1792,
#     "fused_add_norm": True,
#     "hidden_size": 1792,
#     "n_layer": 48,
#     "pad_vocab_size_multiple": 8,
#     "residual_in_fp32": True,
#     "rms_norm": True,
#     "ssm_cfg": {},
#     "vocab_size": 32070,
#     "pad_id": 0,
#     "gen": True,
# }
# mambasubword_config = PretrainedConfig(**{**mamba_config, 'vocab_size': 32070})
# mambasubword = SubWordMambaLMHeadModel(config=mambasubword_config, dtype=torch.bfloat16, device="cuda")
# mambasubword = torch.nn.parallel.DistributedDataParallel(
#     mambasubword,
#     device_ids=[opt.local_rank],
#     output_device=opt.local_rank,
# )
# mambasubword.load_state_dict(torch.load(f"/home/jw2544/SSMWord/train/mamba_h1792_l48_2048_subword_best/model.pt", map_location=torch.device('cpu')))

mambabyte=mambabyte.module
mambasubword=mambasubword.module

print("load finish!")

mambabyte.eval()
mambasubword.eval()

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
'''

def text_to_subword_tokens(text_to_test):
    pg_train_tokens = encoder.encode(text_to_test)
    input_ids = torch.tensor(pg_train_tokens)[None, :]
    input_ids = input_ids.cuda()
    return input_ids

def subword_tokens_to_text(draft_tokens):
    return encoder.decode(draft_tokens[0])

def text_to_byte_tokens(text_to_test):
    text_to_test = text_to_test.replace('\n', '\r\n')
    input_ids = np.frombuffer(text_to_test.encode('utf-8'), dtype=np.uint8)
    input_ids = torch.from_numpy(input_ids)[None, :].long().cuda()
    return input_ids

def byte_tokens_to_text(byte_array):
    return bytes(byte_array).decode('utf-8').replace('\r\n', '\n')

def is_boundary(token):
    # Check if the token is for space (32) or newline (10)
    return token == 32 or token == 10

# Initialize CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

input_ids = text_to_subword_tokens(prompt)
byte_ids = text_to_byte_tokens(prompt)

output = mambasubword.generate(
    input_ids=input_ids,
    max_length=2048,
    cg=False,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
    temperature=1,
    top_k=1,
    top_p=0.98,
    verify_block=10,
    verifier=mambabyte,
    text_to_draft_tokens=text_to_subword_tokens,
    draft_tokens_to_text=subword_tokens_to_text,
    text_to_verifier_tokens=text_to_byte_tokens,
    verifier_tokens_to_text=byte_tokens_to_text,
    start_verify_idx=byte_ids.shape[1],
    is_boundary=is_boundary,
)