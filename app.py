import os
import zipfile
import requests
import torch
from yolov5 import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse

# Define dataset class for face images
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Function to download and extract the zip file
def download_and_extract_zip(url, extract_to='data/'):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, 'faces.zip')

    print(f"Downloading data from {url}...")
    response = requests.get(url)
    
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to download YOLOv5 model if not already downloaded
def download_yolov5_model(model_file='yolov5s.pt'):
    if not os.path.exists(model_file):
        print(f"{model_file} not found, downloading YOLOv5 model...")
        url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        response = requests.get(url)
        
        with open(model_file, 'wb') as f:
            f.write(response.content)
        
        print(f"{model_file} downloaded.")
    else:
        print(f"{model_file} already exists.")

# Function to load the pre-trained YOLO model
def load_pretrained_yolo():
    model_file = 'yolov5s.pt'
    download_yolov5_model(model_file)  # Ensure YOLO model is downloaded
    print("Loading pre-trained YOLO model...")
    model = YOLO(model_file)  # Load the downloaded YOLO model
    return model

# Freeze YOLO backbone to retain object detection capabilities
def freeze_backbone(model):
    print("Freezing YOLO backbone layers to retain object detection...")
    for param in model.model.parameters():
        param.requires_grad = False

    # Unfreeze detection head for face fine-tuning
    for param in model.model[-1].parameters():
        param.requires_grad = True

# Fine-tuning function (only fine-tuning head layers)
def fine_tune_yolo(model, dataset, num_epochs=10):
    print("Starting fine-tuning...")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.model[-1].parameters(), lr=1e-4)  # Fine-tune detection head
    model.train()

    for epoch in range(num_epochs):
        for images in dataloader:
            optimizer.zero_grad()
            results = model(images)
            loss = results.loss  
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} complete. Loss: {loss.item()}")

    print("Fine-tuning complete.")
    return model

# Save the fine-tuned model
def save_model(model, person_name):
    os.makedirs('output', exist_ok=True)
    model_path = f'output/{person_name}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# Function to perform inference using the fine-tuned model
def detect_faces_and_objects(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    all_results = []
    with torch.no_grad():
        for images in dataloader:
            results = model(images)
            all_results.append(results)
    
    return all_results

# Output the results of face and object detection
def output_results(results, person_name):
    print(f"Results for {person_name}:")
    for result in results:
        # Example: process and display results for face detection
        face_detected = False
        objects_detected = result.xyxy[0]  # xyxy format
        
        for obj in objects_detected:
            class_id = int(obj[5])  # Class ID
            label = model.names[class_id]  # Get label from class ID
            confidence = obj[4]  # Confidence score

             if label == 'face':  # Assuming you've fine-tuned to detect faces
                face_detected = True
                print(f"{person_name} detected with confidence: {confidence:.2f}")
            else:
                print(f"Object '{label}' detected with confidence: {confidence:.2f}")

        if not face_detected:
            print(f"{person_name} not detected.")

# Main function to handle the entire process
def main(url, person_name):
    # Step 1: Download and extract the data
    data_dir = 'data/faces'
    download_and_extract_zip(url, data_dir)

    # Step 2: Load the YOLO model
    model = load_pretrained_yolo()

    # Step 3: Freeze the backbone layers to retain object detection capabilities
    freeze_backbone(model)

    # Step 4: Prepare the dataset for fine-tuning
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    dataset = FaceDataset(data_dir, transform=transform)

    # Step 5: Fine-tune the YOLO model with the face data
    model = fine_tune_yolo(model, dataset, num_epochs=5)

    # Step 6: Save the fine-tuned model
    save_model(model, person_name)

    # Step 7: Perform face and object detection on the dataset using the fine-tuned model
    results = detect_faces_and_objects(model, dataset)

    # Step 8: Output the results
    output_results(results, person_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Fine-tuning for Face and Object Detection")
    parser.add_argument('url', type=str, help="URL to the zipped file containing face images")
    parser.add_argument('person_name', type=str, help="Name of the person for face recognition")

    args = parser.parse_args()

    main(args.url, args.person_name)
