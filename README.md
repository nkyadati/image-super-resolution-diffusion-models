# Image super resolution using diffusion models
<div align="justify">
The task of image super resolution - obtaining high resolution images given their low resolution counterparts, has been a long standing challenge in the computer vision community. The applications of this task span a broad range - every day photography, enhancing low-resolution satellite images etc. Starting from the traditional computer vision techniques - interpolation based methods, frequency domain based methods, to the latest advancements in generative models - GANs, Diffusion models, the field has seen a lot of advancements. While traditional methods are compute efficient, they lack performance in case of complex textures. On the other hand, the latest generative models provide exceptional high resolution images in various scenarios, they are compute hungry. This repository will focus on the latest advancements in the field of super resolution using generative models, more specifically, diffusion models. 
</div>

<div align="justify">
Diffusion models have revolutionised the super-resolution tasks because they can effectively model the complex relationship between low-resolution and high-resolution images through a probabilistic and iterative framework. Added to it, their ability to progressively add details, recover high-frequency features, and handle diverse inputs makes them a powerful tool for generating high-quality images. Diffusion models have been able to achieve SOTA performance in both perceptual and quantitative benchmarks for image super resolution. However, there are multiple challenges with these models - higher computational cost, slower inference time. The goal of this repository is to be up-to-date on the SOTA on applying diffusion models to the task of super resolution. 
</div>

## Setup
### Machine
OS: Ubuntu 24.04
GPU: NVIDIA Tesla T4
CUDA driver version: 535.183.01
RAM: 16GB

### Datasets
Urban100, RealSR, DRealSR

### Metrics
<div align="justify">
* PSNR (Peak Signal-to-Noise Ratio): PSNR quantifies the similarity between the high-res image and the ground truth by comparing the pixel-wise differences. It is a ratio between the maximum possible signal power and the power of corrupting noise (or error). Higher PSNR indicates better similarity to the ground truth. While it is strong quantitative metric, it is uncorrelated with the perceptual quality of the output. </div>
<div align="justify">* SSIM (Structural Similarity Index Measure): SSIM measures the structural similarity between the super-resolved and ground truth images by considering luminance, contrast, and structure. Its values range from 0 to 1, where 1 indicates perfect similarity. While it is more perceptually relevant than PSNR, it is sensitive to contrast and luminance changes.</div>
<div align="justify">* LPIPS (Learned Perceptual Image Patch Similarity): LPIPS is a perceptual metric that uses deep neural networks to measure the perceptual distance between the super-resolved and ground truth images. It evaluates similarity based on human perception. A lower LPIPS indicates higher perceptual similarity. While it is strongly correlated with the human judgement, it is computationally intensive compared to the other two metrics.</div>
<div align="justify">* ClipiQA (Contrastive Language-Image Pre-training for Image Quality Assessment): It is a learning-based metric that aligns image quality assessment with human perception using CLIP's feature representations. It is effective for assessing artistic or semantic realism, which is important in generative models. </div>
<div align="justify">* MUSIQ (Multi-Scale Image Quality): MUSIQ is an IQA metric specifically designed to handle multi-scale images, including high-resolution and low-resolution images. It evaluates perceptual quality by considering content across various scales.</div>
<div align="justify">
The last two metrics complement traditional ones (PSNR, SSIM, and LPIPS) by focusing on perceptual and semantic quality, making them valuable for modern image super-resolution evaluation.
</div>

## Models
* Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C. K. Chan, and Chen Change Loy. 2024. Exploiting Diffusion Prior for Real-World Image Super-Resolution. Int. J. Comput. Vision 132, 12 (Dec 2024), 5929–5949. ([paper](https://arxiv.org/abs/2305.07015)) ([code](https://github.com/IceClear/StableSR))
* Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C. Kot, Bihan Wen. 2024. SinSR: Diffusion-Based Image Super-Resolution in a Single Step. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition ([paper](https://arxiv.org/pdf/2311.14760.pdf)) ([code](https://github.com/wyf0912/SinSR))
* Zheng Chen, Haotong Qin, Yong Guo, Xiongfei Su, Xin Yuan, Linghe Kong, and Yulun Zhang. "Binarized Diffusion Model for Image Super-Resolution", NeurIPS, 2024 ([paper](https://arxiv.org/abs/2406.05723)) ([code](https://github.com/zhengchen1999/BI-DiffSR))
* Zongsheng Yue, Jianyi Wang, and Chen Change Loy. ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting, NeurIPS 2023 ([paper](http://arxiv.org/abs/2403.07319)) ([code](https://github.com/zsyOAOA/ResShift))
* Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, Lei Zhang. "Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization", ECCV 2024([paper](https://arxiv.org/abs/2308.14469)) ([code](https://github.com/yangxy/PASD/tree/main))
* Qinpeng Cui, Yixuan Liu, Xinyi Zhang, Qiqi Bao, Qingmin Liao, Li Wang, Tian Lu, Zicheng Liu, Zhongdao Wang, Emad Barsoum. Taming Diffusion Prior for Image Super-Resolution with Domain Shift SDEs, NeurIPS 2024 ([paper](https://arxiv.org/pdf/2409.17778)) ([code](https://github.com/AMD-AIG-AIMA/DoSSR))
* Junyang Chen, Jinshan Pan, Jiangxin Dong. FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution, CVPR 2025 ([paper](https://arxiv.org/abs/2411.18824)) ([code](https://github.com/JyChen9811/FaithDiff/))
* Jinho Jeong, Jinwoo Kim, Younghyun Jo, Seon Joo Kim. Accelerating Image Super-Resolution Networks with Pixel-Level Classification, ECCV 2024 ([paper](https://arxiv.org/abs/2407.21448)) ([code](https://github.com/3587jjh/PCSR))
* Yunpeng Qu, Kun Yuan, Kai Zhao, Qizhi Xie, Jinhua Hao, Ming Sun, Chao Zhou. XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution, ECCV 2024 ([paper](https://arxiv.org/abs/2403.05049))([code](https://github.com/quyp2000/XPSR))

## Steps to run the models
1. StableSR
```bash
# git clone this repository
git clone https://github.com/IceClear/StableSR.git
cd StableSR

# Create a conda environment and activate it
conda env create --file environment.yaml
conda activate stablesr

# Install xformers
pip install xformers

# Install taming & clip
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .

# Download pre-trained models
Find the pre-trained models from OpenXLLab (https://openxlab.org.cn/models/detail/Iceclear/StableSR/tree/main): stablesr_turbo.ckpt and vqgan_cfw_00011.ckpt

# Run the inference on Urban100 dataset
python3 scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt <stablesr_turbo_path> --init-img <path_to_input_images> --outdir <path_to_save_output_images> --ddpm_steps 4 --dec_w 0.5 --seed 42 --n_samples <num_test_images> --vqgan_ckpt <vqgan_path> --colorfix_type wavelet
```

2. SinSR:
```bash
# git clone this repository
git clone https://github.com/wyf0912/SinSR.git
cd SinSR

# Create a conda environment and activate it
conda env create -n SinSR python=3.10
conda activate SinSR
pip install -r requirements.txt

# Run the inference on Urban100 dataset - models will be downloaded automatically 
python3 inference.py -i <path_to_input_images> -o <path_to_save_output_images> --ckpt weights/SinSR_v1.pth --scale 4 --one_step
```

3. ResShift:
```bash
# git clone this repository
git clone https://github.com/zsyOAOA/ResShift.git
cd ResShift

# Create a conda environment and activate it
conda create -n resshift python=3.10
conda activate resshift
pip install -r requirements.txt

# Run the inference on Urban100 dataset
python3 inference_resshift.py -i <path_to_input_images> -o <path_to_save_output_images> --task realsr --scale 4 --version v3
```

4. BI-DiffSR:
```bash
# git clone this repository
git clone https://github.com/zhengchen1999/BI-DiffSR.git
cd BI-DiffSR

# Create a conda environment and activate it
conda create -n bi_diffsr python=3.9
conda activate bi_diffsr
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Clone and install diffusers package
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"

Download the pre-trained models (https://drive.google.com/drive/folders/1hoHAG2yoLltloQ0SYv-QLxwk9Y8ZnTnH) and place them in experiments/pretrained_models/

# Run the inference on Urban100 dataset
python3 test.py -opt options/test/test_BI_DiffSR_x4.yml
```
5. PASD:
```bash
# git clone this repository
git clone https://github.com/yangxy/PASD.git
cd PASD
pip install -e .

# Download the pre-trained models
Download our pre-trained models [pasd][https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip] | [pasd_rrdb][https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_rrdb.zip] | [pasd_light][https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light.zip] | [pasd_light_rrdb][https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light_rrdb.zip], and put them into runs/

# Run inference on Urban100 dataset
pip install -r requirements-test.txt  # install additional dependencies
python test_pasd.py # --use_pasd_light --use_personalized_model
```
6. DoSSR:
```bash
# git clone this repository
git clone https://github.com/AMD-AIG-AIMA/DoSSR
cd DoSSR

# create environment
conda create -n dossr python=3.10
conda activate dossr
pip install -r requirements.txt

Download pre-trained models from [Google Drive][https://drive.google.com/drive/folders/1BTGRbAqXDSNWjyUnTLgDfmjd2xyu8GvP?usp=sharing]

# Run inference on Urban100 dataset
python inference.py \
--input [path/to/input_images] \
--config configs/model/cldm_v21.yaml \
--ckpt [path/to/model_weights(dossr_onestep.ckpt)] \
--steps 1 \
--sr_scale 4 \
--color_fix_type wavelet \
--output [path/to/output_folder] \
--device cuda
```
7. FaithDiff
```bash
# git clone this repository
git clone https://github.com/JyChen9811/FaithDiff.git
cd FaithDiff

Download pre-trained models and put them in ./checkpoints: [FaithDiff][https://huggingface.co/jychen9811/FaithDiff], [SDXL RealVisXL_V4.0][https://huggingface.co/SG161222/RealVisXL_V4.0], [SDXL VAE FP16][https://huggingface.co/madebyollin/sdxl-vae-fp16-fix], [LLaVA CLIP][https://huggingface.co/openai/clip-vit-large-patch14-336], [LLaVA v1.5 13B][https://huggingface.co/liuhaotian/llava-v1.5-13b], [BSRNet][https://drive.usercontent.google.com/download?id=1JGJLiENPkOqi39bvQYa_jlIPlMk24iKH&export=download&authuser=0&confirm=t&uuid=ebaa5d11-ac76-4f54-aabf-90fa43997dec&at=AEz70l4zk_8LTafpGtR0ZSE50F1N:1742369984793]

# Run inference on Urban100 dataset
python test_generate_caption.py --img_dir='path_input_images' --save_dir='path_output_save_caption' --load_8bit_llava
python test_wo_llava.py --img_dir='path_input_images' --json_dir='path_output_save_caption' --save_dir='path_output_images' --upscale=2 --guidance_scale=5 --num_inference_steps=20
```
8. PCSR
```bash
# git clone this repository
git clone https://github.com/3587jjh/PCSR
cd PCSR

# Install dependencies
pip install numpy opencv-python pandas tqdm fast_pytorch_kmeans torch

# Run inference on Urban100 dataset
python test.py --config <config path> --hr_data <hr foler> --lr_data <lr folder> --per_image --crop
```
9. XPSR
```bash
# git clone this repository
git clone https://github.com/quyp2000/XPSR
cd XPSR

# create an environment with python >= 3.9
conda create -n xpsr python=3.9
conda activate xpsr
pip install -r requirements.txt

Download pretrained mdoels: [SD-v1.5][https://huggingface.co/runwayml/stable-diffusion-v1-5], place them in checkpoints/stable-diffusion-v1-5; [XPSR][https://drive.google.com/drive/folders/1rzlHjp6DuiD7timULeDvmxSQignnMywS?usp=sharing], place them in runs/xpsr

Prepare testing images in the testset folder

# Run inference on Urban100 dataset
Generate high level and low level prompts
Download [llava-v1.5-7b][https://huggingface.co/liuhaotian/llava-v1.5-7b] and [MLLM][https://huggingface.co/DLight1551/internlm-xcomposer-vl-7b-qinstruct-full], place them in checkpoints/ folder
python test.py
```

## Results
| Models/Metrics | PSNR     | SSIM    | LPIPS   | ClipiQA | MUSIQ    | Inference time|
| -------------- | -------- | ------- | ------- | ------- | -------- | --------------|
| SinSR          | 24.59139 | 0.69949 | 0.15714 | 0.71062 | 71.96875 | 2.47 seconds  |
| ResShift       | 25.57122 | 0.7327  | 0.15823 | 0.59917 | 71.84375 | 3.3 seconds   |
| Bi-DiffSR      | 25.5606  | 0.73853 | 0.175   | 0.62165 | 67.59375 | 20 seconds    |
| StableSR       | \-       | \-      | \-      | 0.6463  | 70.65344 | 36 seconds    |
| PASD       |   22.1     |  0.58     |  0.35     | 0.61  | 66.0 | 8.4 seconds    |
| DoSSR       |  24.1      |  0.62     |   0.31    | 0.68  | 66.1 | 4 seconds    |
| FaithDiff       |   21.8     |   0.55    |   0.32    | 0.65  | 67.2 | 4.1 seconds  |
| PCSR       |    26.53    |   0.61    |   0.33    | 0.65  | 66.8 | 4.9 seconds    |
| XPSR       |   21.8     |  0.55     |   0.39    | 0.71  | 67.1 | 8.9 seconds    |

Results on Urban100 Test dataset: https://drive.google.com/drive/folders/1AZGn8mAOsk9hTALHym73ACQJE3NRXXSb?usp=sharing. Results on more datasets will follow. 

## ToDo:
- [ ] In-depth qualitative analysis
- [ ] Experiments on more datasets
- [x] Include results from more research papers - PASD, etc.
- [ ] Tools for visualisation of the results
- [ ] Convert the best model to enable mobile deployment (tflite/coreml/onnx)
