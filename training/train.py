import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  # This is needed for Windows

    DATA_PATH = r'training\app\data\yolo_dataset'  # Path where the dataset is stored (images and labels)
    MODEL_PATH = r'training\app/model/yolo_model.pt'  # Path to save the trained model
    EPOCHS = int(os.getenv('EPOCHS', 10))  # Number of training epochs (default: 10)
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))  # Batch size for training (default: 16)
    IMGSZ = int(os.getenv('IMGSZ', 640))  # Image size for training (default: 640)
    WORKERS = int(os.getenv('WORKERS', 4))  # Number of worker threads for data loading (default: 4)
    PRETRAINED_MODEL = os.getenv('PRETRAINED_MODEL', 'yolo11s.pt')  # Name of pretrained model

    # Ensure CUDA is available (use GPU if possible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv5 model
    model = YOLO(PRETRAINED_MODEL)

    # Train the model on the dataset
    model.train(
        data=os.path.join(DATA_PATH, 'data.yaml'),
        epochs=EPOCHS,  # Number of epochs to train
        batch=BATCH_SIZE,  # Specify batch size
        imgsz=IMGSZ,  # Specify image size
        workers=WORKERS,  # Specify number of workers
        device=device  # Specify the device (CUDA or CPU)
    )

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
