import torch
from flux.pipeline_flux import FluxPipeline

from flux.transformer_flux import FluxTransformer2DModel
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import os
import csv

args = OmegaConf.load('config.yaml')
torch_dtype = torch.bfloat16
device = "cuda:0"

extra_branch_num = args.extra_branch_num
bz = args.bz
guidance_scale = args.guidance_scale
n_steps = args.n_steps
mode = args.mode

height = args.image_height
width = args.image_width

model_path = args.model_path
wt_dress = args.model_weight_dress
wt_upper = args.model_weight_upper
seed = args.seed
data_set_path = args.data_set_path

test_pair_path = os.path.join(data_set_path, "test_pair.csv")
images_path = os.path.join(data_set_path, "growth_truth")
garments_path = os.path.join(data_set_path, "test_garments")
result_path = args.result_path

if not os.path.exists(wt_dress):
    raise ValueError("Weight dress root not exists!")
if not os.path.exists(wt_upper):
    raise ValueError("Weight upper root not exists!")

def make_transformer(trained_weight):
    transformer = FluxTransformer2DModel.from_pretrained(model_path,
                                                         torch_dtype=torch_dtype,
                                                         subfolder="transformer",
                                                         extra_branch_num=extra_branch_num,
                                                         low_cpu_mem_usage=False).to(device)
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
    
    transformer.load_state_dict(torch.load(trained_weight)['module'], strict=False)
    
    return transformer

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

pipe_dress = FluxPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch_dtype,
    transformer=make_transformer(wt_dress),
).to(device)

pipe_upper = FluxPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch_dtype,
    transformer=make_transformer(wt_upper),
).to(device)

if __name__ == '__main__':
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"Result save at {result_path}")
    try:
        with open(test_pair_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            _ = next(csv_reader)
            for row in csv_reader:
                image_path = os.path.join(images_path, row[0])
                garment_path = os.path.join(garments_path, row[1])
                object_type = "top clothes" if row[3] == "upper" else "dress"
                
                if row[3] == 'upper':
                    pipe = pipe_upper
                else:
                    pipe = pipe_dress
                
                print(f"\nProcessing person image: {row[0]} | garment image: {row[1]} | class: {object_type}")
                
                person_image = Image.open(image_path).convert("RGB").resize((width,height))
                garment_image = Image.open(garment_path).convert("RGB").resize((height,height))
                
                person_tensor = transform_person(person_image)
                cloth_tensor = transform_cloth(garment_image)
                prompt = f"A fashion model wearing stylish clothing, detailed textures, realistic lighting, fashion photography style."
                
                with torch.inference_mode():
                    result_image = pipe(
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                        prompt=prompt,
                        num_inference_steps=n_steps,
                        guidance_scale=guidance_scale,  
                        height=height,
                        width=width,
                        cloth_img=cloth_tensor,
                        person_img=person_tensor,
                        extra_branch_num=extra_branch_num,
                        mode=mode,
                        max_sequence_length=77,
                    ).images[0]
                
                result_image_name = os.path.splitext(row[0])[0] + ".jpg"
                result_image_path = os.path.join(result_path, result_image_name)
                result_image.save(result_image_path)
                
                print(f"\nProcess completed, file saved in {result_image_path}")
                
    except FileNotFoundError:
        print(f"Error: The file at {test_pair_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


