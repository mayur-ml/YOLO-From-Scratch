# YOLO-From-Scratch

![YOLO](https://github.com/Mayur-ingole/YOLO-From-Scratch/blob/main/assets/YOLO_logo.png)


![YOLO](https://github.com/Mayur-ingole/YOLO-From-Scratch/blob/main/assets/yolo_image.png)

## Introduction

Welcome to the **YOLO-From-Scratch** repository! This project implements the YOLO (You Only Look Once) object detection algorithm from scratch, providing a deep dive into one of the most powerful and efficient real-time object detection models. Whether you’re a seasoned data scientist or a machine learning enthusiast, this project will offer valuable insights into the inner workings of YOLO.

## **Note:** This project is still in progress. Stay tuned for updates!

## Model Architecture
![YOLO](https://github.com/Mayur-ingole/YOLO-From-Scratch/blob/main/assets/yolo_arch.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Custom Implementation**: Understand YOLO’s architecture by building it from scratch.
- **PyTorch Based**: Utilizes the powerful PyTorch library for neural network construction and training.
- **Modular Design**: Clean and modular code, making it easy to extend and experiment with.
- **Comprehensive Tests**: Includes detailed unit tests to ensure correctness and reliability.
- **Visualization**: Tools for visualizing detection results on images.

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.8.0 or higher
- Additional Python libraries: numpy, matplotlib, opencv-python

### Clone the Repository

```bash
git clone https://github.com/Mayur-ingole/YOLO-From-Scratch.git
cd YOLO-From-Scratch
```
### project Structure
```bash
.
├── assets
│   ├── yolo_arch.png
│   ├── yolo_image.png
│   └── YOLO_logo.png
├── dataset.py
├── loss.py
├── model.py
├── __pycache__
│   └── iou.cpython-311.pyc
├── README.md
├── test
│   ├── iou_test.py
│   ├── map_test.py
│   └── nms_test.py
├── train.py
├── utils
│   ├── iou.py
│   ├── map.py
│   ├── nms.py
│   ├── __pycache__
│   │   ├── iou.cpython-311.pyc
│   │   ├── map.cpython-311.pyc
│   │   └── nms.cpython-311.pyc
│   └── README.md
├── utils.py
└── YOLO_Paper
    └── YOLO_Paper.pdf

6 directories, 21 files
