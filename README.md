#JCo-MVTON 

# 环境安装
```
conda create -n tryon python=3.10
conda activate tryon
pip install -r requirements.txt
git clone https://github.com/huggingface/diffusers.git
cp flux/modeling_utils.py diffusers/src/diffusers/models/
pip install -e diffusers/
```


# 推理
```
# 推理：
python inference.py
```
inference.py 中几个关键参数：
- `model_id`: FLUX-dev预训练路径
- `model`: 训练好的Tryon模型路径
- `person_dir`: 测试模特图路径
- `cloth_dir`: 测试服饰图路径
- `save_dir`: 测试结果保存路径
