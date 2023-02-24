
import torch

ckpt_path = ''
save_path = ''

checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# remove optimizer for smaller file size
if 'optimizer' in checkpoint:
    del checkpoint['optimizer']

# delete relative_position_index since we always re-init it
state_dict = checkpoint['state_dict']
relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
for k in relative_position_index_keys:
    del state_dict[k]

torch.save(checkpoint, save_path)

