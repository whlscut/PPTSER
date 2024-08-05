# [PPTSER: A Plug-and-Play Tag-guided Method for Few-shot Semantic Entity Recognition on Visually-rich Documents](https://openreview.net/forum?id=gDqdC8dGfjO)

Visually-rich document information extraction (VIE) is a vital aspect of document understanding, wherein Semantic Entity Recognition (SER) plays a significant role. However, few-shot SER on visually-rich documents remains relatively unexplored despite its considerable potential for practical applications. To address this issue, we propose a simple yet effective **P**lug-and-**P**lay **T**ag-guided method for few-shot **S**emantic **E**ntity **R**ecognition (**PPTSER**) on visually-rich documents. PPTSER is built upon off-the-shelf multi-modal pre-trained models. It leverages the semantics of the tags to guide the SER task, reformulating SER into entity typing and span detection, handling both tasks simultaneously via cross-attention. Experimental results illustrate that PPTSER outperforms existing fine-tuning and few-shot methods, especially in low-data regimes. With full training data, PPTSER achieves comparable or superior performance to fine-tuning baseline. For instance, on the FUNSD benchmark, our method improves the performance of LayoutLMv3-base in 1-shot, 3-shot and 5-shot scenarios by 15.61%, 2.13%, and 2.01%, respectively. Overall, PPTSER demonstrates promising generalizability, effectiveness, and plug-and-play nature for few-shot SER on visually-rich documents.

![](architecture.png)

## Preparation
- Download [layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)
- Download [FUNSD](https://guillaumejaume.github.io/FUNSD/dataset.zip), and put `dataset/FUNSD/split` to the unzipped FUNSD `dataset` directory


## Installation
``` bash
conda create --name layoutlmv3 python=3.7
conda activate layoutlmv3
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -r requirements.txt
pip install -e .
```

## Run
```
sh run_funsh.sh
```

## Copyright
- This repository can only be used for non-commercial research purposes.
- For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).
- Copyright 2024, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net), South China University of Technology. 

## Citation
If you find our work helpful, please cite us:
```
@inproceedings{liao2024pptser,
    title = {PPTSER: A Plug-and-Play Tag-guided Method for Few-shot Semantic Entity Recognition on Visually-rich Documents},
    author = {Wenhui Liao and Jiapeng Wang and Zening Lin and Longfei Xiong and Lianwen Jin},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2024}
    year = {2024},
}
```