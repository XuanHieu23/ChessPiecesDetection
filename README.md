# YOLOv8 Chess Piece Detection

This project uses a **YOLOv8-Nano** model to detect and classify 12 types of chess pieces (6 white and 6 black) in images.

The repository includes custom scripts for both training a new model (`train_model.py`) and running inference with visualization (`inference.py`).

## 🎯 Features

* **12-Class Detection:** Identifies 12 distinct classes (White King, White Queen... Black Pawn).
* **Lightweight Model:** Based on the fast and efficient YOLOv8-Nano.
* **Custom Scripts:** Includes ready-to-use Python scripts for a full training and inference pipeline.

## 🚀 Getting Started

### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/XuanHieu23/ChessPiecesDetection.git](https://github.com/XuanHieu23/ChessPiecesDetection.git)
    cd ChessPiecesDetection
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Dataset

https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset


## 🔧 Usage

### 1. Training

The `train_model.py` script is set up to start training using `yolov8n.pt`.

```bash
# Start training
python train_model.py --data data.yaml --model yolov8n.pt --epochs 10 --imgsz 640 --batch 16
```

* This script will train for `10` epochs with `imgsz=640` and `batch=16`.
* **Important:** Update the `data` path in `train_model.py` to point to `data.yaml`. 

### 2. Inference (Detection)

#### Method 1: Custom Script (`inference.py`)

This script provides custom visualization and saves the output images to a folder.

```bash
 python inference.py --model Result/weights/best.pt --source data/Chess Pieces.yolov8-obb/test/images --output results --imgsz 1024 --conf 0.35
```
* `--model`: Path to your trained file 
* `--source`: Path to a source image or a folder of images.
* `--output`: Directory to save visualized results
* `--imgsz`: Image size for inference.
* `--conf`: Confidence threshold for detections.