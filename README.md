

<div align="center">
<h1>YOLO with Inverted Fusion Depthwise Convolution for Real-time Marine Application</h1>
</div>

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Efficient deep learning models are crucial for real-time computer vision on resource-constrained devices. This paper proposes the Inverted Fusion Depthwise Convolution (IFDWConv) module, integrating inverted bottleneck designs with structural re-parameterization for enhanced feature extraction and computational efficiency. Incorporated into YOLOv8, YOLOv11, and YOLOv12 within the Ultralytics framework, IFDWConv improves mean Average Precision (mAP) by over 1% on the COCO dataset across object detection, pose estimation, and instance segmentation. On marine datasets (FishInv and MegaFauna), our models achieve mAPs of 62% and 83%, surpassing baselines. Despite challenges with small objects and class confusion, IFDWConv offers a robust solution for efficient, high-performance vision tasks, particularly in underwater environments.
</details>


## Main Results

[**Object detection**](https://docs.ultralytics.com/tasks/detect/):
| Model (det)                                                                              | size<br><sup>(pixels) | mAP<sup>det<br>50-95 | Speed  (ms) <br><sup>AMD Ryzen 9 3900XT<br> | Speed  (ms) <br><sup>RTX 3090<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------: | :-----------------: | :-----------------: |
| [mYOLOv8n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolov8n.pt) | 640                   | 39.8                 | 30.8                           | 4.9                           | 2.6                | 7.1              |
| [mYOLO11n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo11n.pt) | 640                   | 40.2                 | 30.7                              | 5.2                           | 2.7                | 6.7              |
| [mYOLO12n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo12n.pt) | 640                   | 41.0                 | 39.0                              | 8.7                           | 2.5                | 6.3              |

[**Instance segmentation**](https://docs.ultralytics.com/tasks/segment/):
| Model (seg)                                                                              | size<br><sup>(pixels) | mAP<sup>mask<br>50-95 | Speed  (ms) <br><sup>AMD Ryzen 9 3900XT<br> | Speed  (ms) <br><sup>RTX 3090<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------: | :-----------------: | :-----------------: |
| [mYOLOv8n-seg](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolov8n-seg.pt) | 640                   | 32.1                 | 46.5                           | 6.4                           | 2.9                | 11.0              |
| [mYOLO11n-pose](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo11n-seg.pt) | 640                   | 32.5                 | 47.4                              | 8.2                           | 3.0                | 10.6              |
| [mYOLO12n-seg](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo12n-seg.pt) | 640                   | 33.0                 | 55.7                              | 13.2                           | 2.8                | 10.2              |

[**Human Pose estimation**](https://docs.ultralytics.com/tasks/pose/):
| Model (pose)                                                                              | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | Speed  (ms) <br><sup>AMD Ryzen 9 3900XT<br> | Speed  (ms) <br><sup>RTX 3090<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------: | :-----------------: | :-----------------: |
| [mYOLOv8n-pose](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolov8n-pose.pt) | 640                   | 51.4                 | 43.0                           | 5.4                           | 2.9                | 8.0              |
| [mYOLO11n-pose](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo11n-pose.pt) | 640                   | 49.9                 | 43.0                              | 5.7                           | 3.0                | 7.6              |
| [mYOLO12n-pose](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo12n-pose.pt) | 640                   | 48.7                 | 51.9                              | 10.1                           | 2.8                | 7.2              |

[**FishInv**](https://github.com/Orange-OpenSource/marine-detect):
| Model (det)                                                                              | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | Speed  (ms) <br><sup>AMD Ryzen 9 3900XT<br> | Speed  (ms) <br><sup>RTX 3090<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------: | :-----------------: | :-----------------: |
| [YOLOv8n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolov8n-FishInv.pt) | 640                   | 60.1                 | 25.3                           | 4.8                           | 3.0                | 8.1              |
| [YOLO11n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolo11n-FishInv.pt) | 640                   | 61.7                 | 32.5                           | 6.1                           | 2.6                | 6.5               |
| [YOLO12n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolo12n-FishInv.pt) | 640                   | 61.9                 | 38.1                           | 8.8                           | 2.6                | 6.5               |
| [mYOLOv8n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolov8n-FishInv.pt) | 640                   | 61.9                 | 30.8                           | 4.9                           | 2.6                | 6.9              |
| [mYOLO11n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo11n-FishInv.pt) | 640                   | 62.8                 | 30.7                              | 5.2                           | 2.7                | 6.5              |
| [mYOLO12n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo12n-FishInv.pt) | 640                   | 62.2                 | 39.0                              | 8.7                           | 2.5                | 6.1              |

[**MegaFauna**](https://github.com/Orange-OpenSource/marine-detect):
| Model (det)                                                                              | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | Speed  (ms) <br><sup>AMD Ryzen 9 3900XT<br> | Speed  (ms) <br><sup>RTX 3090<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------: | :-----------------: | :-----------------: |
| [YOLOv8n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolov8n-MegaFauna.pt) | 640                   | 83.1                 | 25.3                           | 4.8                           | 3.0                | 8.1              |
| [YOLO11n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolo11n-MegaFauna.pt) | 640                   | 83.1                 | 32.5                           | 6.1                           | 2.6                | 6.5               |
| [YOLO12n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/yolo12n-MegaFauna.pt) | 640                   | 83.4                 | 38.1                           | 8.8                           | 2.6                | 6.5               |
| [mYOLOv8n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolov8n-MegaFauna.pt) | 640                   | 83.5                 | 30.8                           | 4.9                           | 2.6                | 6.9              |
| [mYOLO11n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo11n-MegaFauna.pt) | 640                   | 83.7                 | 30.7                              | 5.2                           | 2.7                | 6.5              |
| [mYOLO12n](https://github.com/thanhhvnqb/ultralytics-myolo-c2mb/releases/download/v1.0/myolo12n-MegaFauna.pt) | 640                   | 84.1                 | 39.0                              | 8.7                           | 2.5                | 6.1              |
</details>

## Installation
```
conda create -n ultralytics python=3.11
conda activate ultralytics
pip install -r requirements.txt
pip install -e .
```

## Validation

```python
from ultralytics import YOLO

model = YOLO('myolo12n.pt')
model.val(data='coco.yaml', save_json=True)
```

## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

```

## Prediction
```python
from ultralytics import YOLO

model = YOLO('myolo12n.pt')
model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO('myolov12n.pt')
model.export(format="engine", half=True)  # or format="onnx"
```

## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!

<!-- ## Citation

```BibTeX
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
``` -->
