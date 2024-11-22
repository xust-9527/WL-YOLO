# Experimental environment

    python: 3.8.16
    torch: 1.13.1+cu117
    torchvision: 0.14.1+cu117
    mmcv: 2.1.0
    mmengine: 0.9.0


# Environment Configuration

    1. Execute "pip uninstall ultralytics" to uninstall the ultralytics library installed in your environment.
    2. After the uninstallation is complete, execute the same again, if WARNING: Skipping ultralytics as it is not installed.
    3. Additional required package installation commands.
        pip install timm==0.9.5 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 torch-pruning==1.3.7
    4. If there are any packages missing at runtime, please install them yourself.

# Description of the document
1. train.py
    Scripts for training models

2. compress.py

    Scripts for model pruning

3. val.py
    Scripts for calculating metrics using trained models

4. detect.py
    Scripts for reasoning

# How to use

1. Load the dataset, change the file path in `dataset/data.yaml` `path:`
2. Run `xml2txt.py` to convert the xml file to txt format
3. Run `split_data.py` to split the dataset

4. 运行`python train.py`