import os
import json
from safetensors.torch import load_file, save_file
from tqdm import tqdm

sharded_folder = "/raid/ltnghia01/tanpm/JCo-MVTON/FLUX.1-dev/transformer/"
output_file = "/raid/ltnghia01/tanpm/JCo-MVTON/FLUX.1-dev/transformer_merged/diffusion_pytorch_model.safetensors"

print(f"Merge safetensors file: {sharded_folder}")

index_path = os.path.join(sharded_folder, "diffusion_pytorch_model.safetensors.index.json")
if not os.path.exists(index_path):
    index_path = os.path.join(sharded_folder, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Không tìm thấy file index.json tại: {sharded_folder}")

with open(index_path, 'r') as f:
    index_data = json.load(f)

shard_files = sorted(list(set(index_data["weight_map"].values())))
shard_paths = [os.path.join(sharded_folder, f) for f in shard_files]

print(f"Tìm thấy {len(shard_paths)} file sharded để gộp.")

full_state_dict = {}
for shard_path in tqdm(shard_paths, desc="Đang gộp các shard"):
    state_dict_shard = load_file(shard_path, device="cpu")
    full_state_dict.update(state_dict_shard)
    del state_dict_shard

print(f"Đã gộp xong. Tổng số tensors: {len(full_state_dict)}")

print(f"Đang lưu vào file output: {output_file}")
save_file(full_state_dict, output_file)

print("Hoàn thành! File đã được gộp thành công.")