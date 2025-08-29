# JCo-MVTON: Jointly Controllable Multi-Modal Diffusion Transformer for Mask-Free Virtual Try-on

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.17614)
[![Homepage](https://img.shields.io/badge/Homepage-Visit_My_Site-orange)](https://damocv.github.io/JCo-MVTON.github.io/)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-HuggingFace-yellow)](https://huggingface.co/Damo-vision/JCo-MVTON)
[![Demo](https://img.shields.io/badge/Demo-Link-green)](https://market.aliyun.com/apimarket/detail/cmapi00067129?spm=5176.shop.result.2.6e323934OAW8XR&innerSource=search)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>
<div align="center">

Aowen Wang¹, Wei Li¹, Hao Luo¹ ², Mengxing Ao¹, Fan Wang¹

¹DAMO Academy, Alibaba Group
²Hupan Lab

</div>

## Overview

JCo-MVTON introduces a novel framework for mask-free virtual try-on based on MM-DiT that addresses key limitations of existing systems: rigid dependencies on human body masks, limited fine-grained control over garment attributes, and poor generalization to in-the-wild scenarios.
<div align="center">
    <img src="assets/framework.jpg" alt="Overview" width="600"/>
</div>

## Quick Start

#### Clone the repository

```bash
git clone https://github.com/damo-cv/JCo-MVTON.git
cd JCo-MVTON
```

#### Create conda environment

```bash
conda create -n jco-mvton python=3.10
conda activate jco-mvton
```

#### Install dependencies

```bash
pip install -r requirements.txt
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.33.0
cp flux/modeling_utils.py   diffusers/src/diffusers/models
pip install .
```

#### Download Pre-trained Models

```bash
# Download the upper model checkpoint
wget https://huggingface.co/Damo-vision/JCo-MVTON/resolve/main/try_on_upper.pt

# Download the lower model checkpoint
wget https://huggingface.co/Damo-vision/JCo-MVTON/resolve/main/try_on_lower.pt

# Download the dress model checkpoint
wget https://huggingface.co/Damo-vision/JCo-MVTON/resolve/main/try_on_dress.pt

```

## Basic Usage

```
# Load transformer
model_id = "black-forest-labs/FLUX.1-dev"
ckpt = 'ckpts/try_on_upper.pt'
transformer = FluxTransformer2DModel.from_pretrained(
model_id,
torch_dtype=torch_dtype,
subfolder="transformer",
extra_branch_num=extra_branch_num,
low_cpu_mem_usage=False,
).to(device)
transformer.load_state_dict(torch.load(ckpt)['module'], strict=False)
pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype,
    transformer=transformer,
).to(device)
# Load and preprocess images

person = Image.open('assets/ref.jpg').convert("RGB").resize((width, height))
cloth = Image.open('assets/upper.jpg').convert("RGB").resize((height, height))

person_tensor = transform_person(person)
cloth_tensor = transform_cloth(cloth)

prompt = "A fashion model wearing stylish clothing, high-resolution 8k, detailed textures, realistic lighting, fashion photography style."

# Generate image

with torch.inference_mode():
generated_image = pipe(
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

# Save result

person_tensor = transform_output(person)
cloth_tensor = transform_output(cloth)
generated_tensor = transform_output(generated_image)

concatenated_tensor = torch.cat((cloth_tensor, person_tensor, generated_tensor), dim=2)
vutils.save_image(concatenated_tensor, 'output.png')
```



## Results
<div align="center">
    <img src="assets/overview.jpg" alt="Overview" width="600"/>
</div>

JCo-MVTON achieves state-of-the-art performance across multiple metrics:

| Methods | Paired | Paired | Paired |Paired  | Unpaired| Unpaired |
|---------|--------|-----|-------|----------|-----|-------|
|         | SSIM ↑ | FID ↓ | KID ↓ | LPIPS ↓ | FID ↓ | KID ↓ |
| MV-VTON (Wang et al., 2025b) | 0.8083 | 15.442 | 7.501 | 0.1171 | 17.900 | 3.861 |
| OOTDiffusion (Xu et al., 2024) | 0.8187 | 9.305 | 4.086 | 0.0876 | 12.408 | 4.689 |
| JCo-MVTON (Ours) | 0.8601 | 8.103 | 2.003 | 0.0891 | 9.561 | 2.700 |

## Citation

If you find our work useful, please cite:

```bibtex
@misc{wang2025jcomvtonjointlycontrollablemultimodal,
      title={JCo-MVTON: Jointly Controllable Multi-Modal Diffusion Transformer for Mask-Free Virtual Try-on}, 
      author={Aowen Wang and Wei Li and Hao Luo and Mengxing Ao and Chenyu Zhu and Xinyang Li and Fan Wang},
      year={2025},
      eprint={2508.17614},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.17614}, 
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgments

We thank the open-source community for their valuable contributions and the reviewers for their constructive feedback. Special thanks to the DAMO Academy and Hupan Lab for supporting this research.
