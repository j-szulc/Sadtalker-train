from src.utils.safetensor_helper import load_x_from_safetensor
import safetensors  
import safetensors.torch


org = './checkpoints/SadTalker_V0.0.2_256.safetensors'
# new = './result_pose/ravdess2/ep334_iter1000.safetensors'
new = './result_pose/crema-d/ep475_iter9500.safetensors'

org = safetensors.torch.load_file(org)
new = safetensors.torch.load_file(new)
add = {}
for key in org:
    print(key)
    if key not in new:
        add[key] = org[key]
new.update(add)
safetensors.torch.save_file(new, 'latest.safetensors') 