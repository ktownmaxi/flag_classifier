import os

import torch
import torchvision
from going_modular.going_modular.predictions import pred_and_plot_image
from going_modular.going_modular import data_setup

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ressources")
img_path = os.path.join(data_path, "big_img_dataset")

train_dir = os.path.join(img_path, "train")
test_dir = os.path.join(img_path, "test")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=auto_transforms,
                                                                                   batch_size=32)


model = torch.load("V3.pth").to(device)

torch.manual_seed(42)

for i in range(1000):
    image_path = str(input("Enter image path: "))

    label = pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_path=image_path,
        transform=auto_transforms,
        device=device
    )

    print(f"Predicted label: {label}")