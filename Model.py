# File: train_drowning_detector.py
import torch
from yolov5 import train, detect

# Install dependencies
# !pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# Dataset configuration (YAML format)
with open('drowning_dataset.yaml', 'w') as f:
    f.write('''
    path: ../datasets/drowning
    train: images/train
    val: images/val
    nc: 2
    names: ['normal_swim', 'drowning']
    ''')

# Training parameters
config = {
    'weights': 'yolov5s.pt',          # Start with pretrained Tiny model
    'data': 'drowning_dataset.yaml',
    'epochs': 50,
    'batch-size': 16,
    'img-size': 640,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Start training
train.run(**config)