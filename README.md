# YOLO Face Detection and Object Recognition

This project fine-tunes a YOLOv5 model for face detection using a provided dataset of face images and performs face and object detection. The application is containerized using Docker for ease of deployment and execution.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)

## Project Overview

This project provides a Python application that:
1. Downloads and extracts face images from a provided URL.
2. Fine-tunes a YOLOv5 model to detect faces in addition to the standard objects it was originally trained on.
3. Saves the fine-tuned model with a name based on the provided person's name.
4. Performs face and object detection and outputs the results.

## Requirements

- Docker
- Python 3.9+
- PyTorch 2.0.1+cu117
- torchvision 0.15.2+cu117
- Pillow
- requests
- numpy
- pyyaml
- tqdm

## Setup and Installation

### Docker Setup

1. **Build the Docker Image**

   Build the Docker image with the following command:

   ```bash
   docker build -t face-detection-app .

2. **Run the Docker Container**
   Run the container with the necessary arguments (URL to the zipped file and person's name):

   ```bash
   docker run --rm face-detection-app url personname
   example : docker run --rm face-detection-app https://example.com/faces.zip JohnDoe
   Replace https://example.com/faces.zip with the actual URL of the zipped file containing face images, and JohnDoe with the name of the person.

   
   
