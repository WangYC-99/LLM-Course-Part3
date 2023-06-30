#usage: python state_dict_check.py {ckpt_path} >> {output_path}

import torch
import sys

# target_path = "/zhangpai25/glm_finetune/chpt_pretrained/mp4/30000/mp_rank_00_model_states.pt"
target_path = sys.argv[1]

print("start loading")
content = torch.load(target_path)
print("loaded")

# 查看state_dict 中都存了些什么
# key_list = content.keys()
# print(len(key_list))
# print(key_list)
# 输出：21 \ dict_keys(['module', 'buffer_names', 'optimizer', 'param_shapes', 'lr_scheduler', 'sparse_tensor_module_names', 'skipped_steps', 'global_steps', 'global_samples', 'dp_world_size', 'mp_world_size', 'ds_config', 'ds_version', 'important_nonparameters', 'iteration', 'client_lr_scheduler', 'random_rng_state', 'np_rng_state', 'torch_rng_state', 'cuda_rng_state', 'rng_tracker_states'])

module = content['module']

for each in module.keys():
    print(each)
