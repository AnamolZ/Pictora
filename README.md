
##### Directory Structure
```
Pictora/
├── captions/
│   ├── averageConfidence.png
│   ├── filtered_captions.json
│   └── unfilteredCaption.json
├── freemiumModel/
│   ├── caption_generator.py
│   ├── clip_scorer.py
│   ├── feature_extractor.py
│   ├── freemiumModel.py
│   ├── freemiumModelAPI.py
│   └── model_builder.py
├── modelCreation/
│   ├── data_generator.py
│   ├── description.py
│   ├── feature_extractor.py
│   ├── model_builder.py
│   ├── modelCreation.py
│   ├── pipeline.py
│   ├── sequence_generator.py
│   └── tokenizer_manager.py
├── models/
│   ├── modelV1.1.h5
│   └── yolo12n.pt
├── pre_processor/
│   ├── get_pseudo_caption.ipynb
│   └── pre_processing.ipynb
├── premiumModel/
│   ├── clip_score.py
│   ├── extract_features.py
│   ├── generate_caption.py
│   ├── model.py
│   ├── premiumModel.py
│   ├── premiumModelAPI.py
│   └── train.py
├── processed/
│   ├── descriptions.txt
│   ├── features.p
│   └── tokenizer.p
├── training_data/
│   ├── dataset/
│   └── pseudo_caption/pseudo_caption.txt
├── main.py
├── README.md
├── requirements.txt
└── testImages/

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
