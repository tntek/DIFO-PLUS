# DIFOPLUS

## Preliminary

To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate difoplus
```
- **Datasets**
  - `office-31` [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA)
  - `office-home` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)
  - `VISDA-C` [VISDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
You need to download the above dataset,modify the path of images in each '.txt' under the folder './data/'.In addition, class name files for each dataset also under the folder './data/'.The prepared directory would look like:
```bash
├── data
    ├── office-home
        ├── amazon_list.txt
        ├── classname.txt
        ├── dslr_list.txt
        ├── webcam_list.txt
    ├── office-home
        ├── Art_list.txt
        ├── classname.txt
        ├── Clipart_list.txt
        ├── Product_list.txt
        ├── RealWorld_list.txt
    ...  ...
```
For the ImageNet variations, modify the `${DATA_DIR}` in the `conf.py` to your data directory where stores the ImageNet variations datasets.

## Training
We provide config files for experiments. 
### Source
- For office-31, office-home and VISDA-C, there is an example to training a source model :
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/source.yaml" SETTING.S 0
```
- For domainnet126, we follow [AdaContrast](https://github.com/DianCh/AdaContrast) to train the source model.

- For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) or [timm](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13) can be used.

- We also provide the pre-trained source models which can be downloaded from [here](https://drive.google.com/drive/folders/17n6goPXw_-ERgTK8R8nm4M_8PJPTEK1j?usp=sharing).

### Target
After obtaining the source models, modify the `${CKPT_DIR}` in the `conf.py` to your source model directory. For office-31, office-home and VISDA-C, simply run the following Python file with the corresponding config file to execute source-free domain adaptation.
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/difo.yaml" SETTING.S 0 SETTING.T 1
```
For domainnet126 and ImageNet variations.
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_in_126.py --cfg "cfgs/domainnet126/difo.yaml" SETTING.S 0 SETTING.T 1
```

## Acknowledgements
+ SHOT [official](https://github.com/tim-learn/SHOT)
+ NRC [official](https://github.com/Albert0147/NRC_SFDA)
+ COWA [official](https://github.com/Jhyun17/CoWA-JMDS)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ PLUE [official](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation)
+ CoOp [official](https://github.com/KaiyangZhou/CoOp)
+ RMT [official](https://github.com/mariodoebler/test-time-adaptation)
