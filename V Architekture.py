if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import os

    from torch import nn
    from torchvision import transforms

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ressources")
    img_path = os.path.join(data_path, "big_img_dataset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_dir = os.path.join(img_path, "train")
    test_dir = os.path.join(img_path, "test")

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    auto_transforms = weights.transforms()

    from going_modular.going_modular import data_setup

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=auto_transforms,
                                                                                   batch_size=32)

    model = torchvision.models.resnet50(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = True

    from torch import nn

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280,  # feature vector coming in
                  out_features=len(class_names))).to(device)  # how many classes do we have?

    print(model.classifier)

    ### Train

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    from going_modular.going_modular import engine

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Setup training and save the results
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=5,
                           device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    from helper_functions import plot_loss_curves

    plot_loss_curves(results)

    torch.save(model, "V3.pth")
