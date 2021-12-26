# T2I_CL
This is the official Pytorch implementation of the paper [Improving Text-to-Image Synthesis Using Contrastive Learning](https://arxiv.org/abs/2107.02423?context=cs)


## Requirements
* Linux
* Python ≥ 3.6

* PyTorch ≥ 1.4.0


## Prepare Data



Download the preprocessed datasets from [AttnGAN](https://github.com/taoxugit/AttnGAN)

Alternatively, another site is from [DM-GAN](https://github.com/MinfengZhu/DM-GAN)


## Training
- Pretrain DAMSM+CL:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
  - For coco dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml --gpu 0`

- Train AttnGAN+CL:
  - For bird dataset: `python main.py --cfg cfg/bird_attn2.yml --gpu 0`
  - For coco dataset: `python main.py --cfg cfg/coco_attn2.yml --gpu 0`

- Train DM-GAN+CL:
  - For bird dataset: `python main.py --cfg cfg/bird_DMGAN.yml --gpu 0`
  - For coco dataset: `python main.py --cfg cfg/coco_DMGAN.yml --gpu 0`

## Pretrained Models
- [DAMSM+CL for bird](https://drive.google.com/file/d/15w_mKV7UzmC3jMqplKyMawUEEJaJozTZ/view?usp=sharing). Download and save it to `DAMSMencoders/`
- [DAMSM+CL for coco](https://drive.google.com/file/d/1zktujHYRR4Bix7GwG9MXLtwHHv4Dayhu/view?usp=sharing). Download and save it to `DAMSMencoders/`
- [AttnGAN+CL for bird](https://drive.google.com/file/d/138g15XlWXBM_Wx-owMLkJ7dGImWJtra1/view?usp=sharing). Download and save it to `models/`
- [AttnGAN+CL for coco](https://drive.google.com/file/d/1ZnwXqe3nT0v1E-POtIKLvrPuoYLYXOkP/view?usp=sharing). Download and save it to `models/`
- [DM-GAN+CL for bird](https://drive.google.com/file/d/1QIBMz3OSPGKe5W8_dlNTcaETivVPlUtf/view?usp=sharing). Download and save it to `models/`
- [DM-GAN+CL for coco](https://drive.google.com/file/d/1nNB-MHGkVLWj1zlOcsDVGzyhkrsvw7UY/view?usp=sharing). Download and save it to `models/`

## Evaluation
- Sampling and get the R-precision:
  - `python main.py --cfg cfg/eval_bird.yml --gpu 0`
  - `python main.py --cfg cfg/eval_coco.yml --gpu 0`

- Inception score:
  - ` python inception_score_bird.py --image_folder fake_images_bird`
  - ` python inception_score_coco.py fake_images_coco`

- FID: 
  - ` python fid_score.py --gpu 0 --batch-size 50 --path1 real_images_bird --path2 fake_images_bird`
  - ` python fid_score.py --gpu 0 --batch-size 50 --path1 real_images_coco --path2 fake_images_coco`
  
### Citation
If you find this work useful in your research, please consider citing:

```
@article{ye2021improving,
  title={Improving Text-to-Image Synthesis Using Contrastive Learning},
  author={Ye, Hui and Yang, Xiulong and Takac, Martin and Sunderraman, Rajshekhar and Ji, Shihao},
  journal={The 32nd British Machine Vision Conference (BMVC)},
  year={2021}
}
```
### Acknowledge
Our work is based on the following works:
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/taoxugit/AttnGAN)
- [DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis](https://arxiv.org/abs/1904.01310) [[code]](https://github.com/MinfengZhu/DM-GAN)
