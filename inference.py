import torch
from flux.pipeline_flux import FluxPipeline


from flux.transformer_flux import FluxTransformer2DModel
import copy
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
torch_dtype = torch.bfloat16
device = "cuda:0"
extra_branch_num=2
resolution =1024
height = resolution
width = resolution * 3 // 4 
bz = 8
guidance_scale = 3.5  
n_steps = 28
seed = 0
mode = 2

model_id = "/mnt/yixiao/FLUX.1-dev"
wt = 'ckpts/try_on_upper.pt'
transformer = FluxTransformer2DModel.from_pretrained(model_id,
                                                     torch_dtype=torch_dtype,
                                                     subfolder="transformer",
                                                     extra_branch_num=extra_branch_num,
                                                     low_cpu_mem_usage=False, 
                                                    ).to(device)
for j in range(extra_branch_num):
    if mode == 1:
        transformer.extra_embedder[j].load_state_dict(transformer.x_embedder.state_dict())
    for i in range(transformer.config.num_layers):
        transformer.transformer_blocks[i].attn.extra_to_q[j].load_state_dict(transformer.transformer_blocks[i].attn.to_q.state_dict())
        transformer.transformer_blocks[i].attn.extra_to_k[j].load_state_dict(transformer.transformer_blocks[i].attn.to_k.state_dict())
        transformer.transformer_blocks[i].attn.extra_to_v[j].load_state_dict(transformer.transformer_blocks[i].attn.to_v.state_dict())
        if mode == 1:
            transformer.transformer_blocks[i].extra_norm1[j].load_state_dict(transformer.transformer_blocks[i].norm1.state_dict())
            transformer.transformer_blocks[i].extra_norm2[j].load_state_dict(transformer.transformer_blocks[i].norm2.state_dict())
            transformer.transformer_blocks[i].extra_ff[j].load_state_dict(transformer.transformer_blocks[i].ff.state_dict())
            transformer.transformer_blocks[i].attn.extra_to_out[0][j].load_state_dict(transformer.transformer_blocks[i].attn.to_out[0].state_dict())
            transformer.transformer_blocks[i].attn.extra_to_out[1][j].load_state_dict(transformer.transformer_blocks[i].attn.to_out[1].state_dict())
            transformer.transformer_blocks[i].attn.extra_norm_q[j].load_state_dict(transformer.transformer_blocks[i].attn.norm_q.state_dict())
            transformer.transformer_blocks[i].attn.extra_norm_k[j].load_state_dict(transformer.transformer_blocks[i].attn.norm_k.state_dict())
    if mode == 2 :
        for i in range(transformer.config.num_single_layers):
            transformer.single_transformer_blocks[i].attn.extra_to_q[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_q.state_dict())
            transformer.single_transformer_blocks[i].attn.extra_to_k[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_k.state_dict())
            transformer.single_transformer_blocks[i].attn.extra_to_v[j].load_state_dict(transformer.single_transformer_blocks[i].attn.to_v.state_dict())


pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype,
    transformer=transformer,
).to(device)
transform_person = transforms.Compose([
    transforms.Resize(size=(height,width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
transform_cloth = transforms.Compose([
    transforms.Resize(size=(height,height)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
transform_output = transforms.Compose([
        transforms.ToTensor(),
])
transformer.load_state_dict(torch.load(wt)['module'], strict=False)
pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype,
    transformer=transformer,
).to(device)


person = Image.open(f'assets/ref.jpg').convert("RGB").resize((width,height))
cloth = Image.open(f'assets/upper.jpg').convert("RGB").resize((height,height))
person_tensor = transform_person(person)  # 形状为 (C, H, W)
cloth_tensor = transform_cloth(cloth)
prompt = f"A fashion model wearing stylish clothing, high-resolution 8k, detailed textures, realistic lighting, fashion photography style."

with torch.inference_mode():
    generated_image = pipe(
        generator=torch.Generator(device="cpu").manual_seed(seed),
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,  
        height=height,
        width=width,
        cloth_img=cloth_tensor,
        person_img=person_tensor,
        extra_branch_num=2,
        mode=mode,
        max_sequence_length=77,
    ).images[0]
transform_output = transforms.Compose([
    transforms.ToTensor(),
])
person_tensor = transform_output(person)  
cloth_tensor = transform_output(cloth)
generated_tensor = transform_output(generated_image)  
concatenated_tensor = torch.cat((cloth_tensor, person_tensor, generated_tensor), dim=2)
vutils.save_image(concatenated_tensor, f'output.png')


