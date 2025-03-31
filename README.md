
##### Directory Structure
```
Pictora/
├── training_data/
│   ├── dataset/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── pseudo_caption/
│       └── pseudo_caption.txt
├── train_model.ipynb
├── models/
├── procedded/
├── testImage/
└── README.md
```

Installation Instruction: GPU Support for Library Use

```
conda create -n Pictora python=3.9

conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge

pip install "tensorflow<2.11"

pip install numpy<2

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install git+https://github.com/huggingface/transformers
```
