import torch
import os
from .NetworkModule import ViTForCIFAR, NetList
from .logging_utils import setup_logging

# Set save path
save_dir = "pretrained_models"
os.makedirs(save_dir, exist_ok=True)

# Initialize log
logger = setup_logging("pretrain_vit", 0, save_dir)

# Create a pretrained ViT model
model = ViTForCIFAR(input_dim=(3, 32, 32), pretrained=True, logger=logger)
model.to("cpu")

# Wrap the model in the format to be saved
models_list = NetList([model])
data_to_save = {
    "models": models_list,
    "info": "Pretrained ViT-B-16 from torchvision, modified for CIFAR binary classification"
}

# Save model
save_path = os.path.join(save_dir, "vit_pretrained.pt")
torch.save(data_to_save, save_path)
logger.info(f"Saved pretrained ViT model to {save_path}")
print(f"Saved pretrained ViT model to {save_path}")
