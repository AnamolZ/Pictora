
##### Directory Structure
```
Pictora/
├── Pictora_Dataset/
│   ├── train/
│   │   ├── train_caption/
│   │   │   └── train_pseudo_caption.json
│   │   └── train_images/
│   │       ├── img1.jpg
│   │       ├── img2.jpg
│   │       └── ...
│   └── validation/
│       ├── validation_caption/
│       │   └── validation_pseudo_caption.json
│       └── validation_images/
│           ├── img1.jpg
│           ├── img2.jpg
│           └── ...
├── main.ipynb
├── Pre_Captions/ (Has unprocessed Captions and Caption Related Files)
└── readme.md
```

Installation Instruction: GPU Support for Library Use

```
conda create -n Pictora python=3.9

pip install "tensorflow<2.11"

pip install numpy<2

conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://github.com/huggingface/transformers
```