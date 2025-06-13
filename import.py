import torch
from torchreid import models
from torchreid.utils import load_pretrained_weights

# Cargar modelo preentrenado
model = models.build_model(name='osnet_x1_0', num_classes=1000)
load_pretrained_weights(model, r'C:\Users\nelso\.cache\torch\checkpoints\osnet_x1_0_imagenet.pth')

 # Este archivo lo debes descargar manualmente
model.eval()