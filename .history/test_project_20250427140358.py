import torch
from interface import TheModel, the_trainer, TheDataset, the_dataloader, the_batch_size, total_epochs
from torchvision import transforms

def test_interface():
    print("Testing interface imports...")
    print(f"Batch size: {the_batch_size}, Total epochs: {total_epochs}")

    # Instantiate the model
    model = TheModel()
    print("Model instantiated successfully.")

    # Test dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  # Ensure this matches your config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = TheDataset(split='Training', fold=1, transform=transform)
    print(f"Dataset size (Training): {len(dataset)}")

    dataloader = the_dataloader(split='Training', fold=1, transform=transform, batch_size=the_batch_size)
    print(f"Dataloader batches: {len(dataloader)}")

    # Test a batch from the dataloader
    for images, labels in dataloader:
        print(f"Batch size: {images.size(0)}, Image shape: {images.size()}, Labels: {labels}")
        break

def test_training():
    print("\nTesting training loop...")
    # Instantiate the model
    model = TheModel()

    # Define a simple loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Get dataloaders
    train_loader = the_dataloader(split='Training', fold=1, transform=transform, batch_size=the_batch_size)

    # Run the training loop
    the_trainer(model, total_epochs, train_loader, loss_fn, optimizer)
    print("Training loop executed successfully.")

def test_prediction():
    print("\nTesting prediction function...")
    # Instantiate the model
    model = TheModel()

    # Load the model weights (ensure final_weights.pth exists in checkpoints/)
    checkpoint_path = './checkpoints/final_weights.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Define a list of image paths for testing
    image_paths = [
        './data/img01.jpg',
        './data/img02.jpg',
        './data/img03.jpg',
        './data/img04.jpg',
        './data/img05.jpg',
        './data/img06.jpg',
        './data/img07.jpg',
        './data/img08.jpg',
        './data/img09.jpg',
        './data/img10.jpg'
    ]

    # Run predictions
    predictions = the_predictor(image_paths, model=model)
    for img_path, pred in zip(image_paths, predictions):
        print(f"Image: {img_path}, Predicted Emotion: {pred}")

if __name__ == "__main__":
    print("Starting project tests...\n")
    test_interface()
    test_training()
    test_prediction()
    print("\nAll tests completed successfully.")