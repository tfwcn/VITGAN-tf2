# VITGAN-tf2

tensorflow2 的 VITGAN 实现。

论文地址：[VITGAN: Training GANs with Vision Transformers](https://arxiv.org/pdf/2107.04589v1.pdf)

**注意：目前训练仍然有问题，欢迎纠正。**

## 实现模块：
- Mapping NetWork
- PositionalEmbedding
- MLP
- MSA多头注意力
- SLN自调制
- CoordinatesPositionalEmbedding (这部分可能有误)
- ModulatedLinear (这部分可能有误)
- Siren (这部分可能有误)
- Generator生成器
- PatchEmbedding
- ISN
- Discriminator鉴别器
- VITGAN

## 训练：
```bash
python train.py --labels_dir "./图片文件夹" --models_dir "./data/model/" --batch_size 8 --image_size 224 --patch_size 16 --overlapping 3
```