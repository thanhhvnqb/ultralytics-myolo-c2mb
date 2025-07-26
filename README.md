

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

[**Instance segmentation**](https://github.com/sunsmarterjie/yolov12/tree/Seg):
| Model (seg)                                                                              | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed  (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------:| :------------------: | :-----------------: |
| [YOLOv12n-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt) | 640                   | 39.9                 | 32.8                  | 1.84                           | 2.8                | 9.9              |
| [YOLOv12s-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt) | 640                   | 47.5                 | 38.6                  | 2.84                           | 9.8                | 33.4              |
| [YOLOv12m-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt) | 640                   | 52.4                 | 42.3                  | 6.27                           | 21.9               | 115.1             |
| [YOLOv12l-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt) | 640                   | 54.0                 | 43.2                  | 7.61                          | 28.8               | 137.7             |
| [YOLOv12x-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt) | 640                   | 55.2                 | 44.2                  | 15.43                          | 64.5               | 308.7             |


[**Classification**](https://github.com/sunsmarterjie/yolov12/tree/Cls):
| Model (cls)                                                                              | size<br><sup>(pixels) | Acc.<br><sup>top-1<br> | Acc.<br><sup>top-5<br> | Speed  (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(G) |
| :----------------------------------------------------------------------------------------| :-------------------: | :------------: | :------------: | :-------------------------------------:| :----------------: | :---------------: |
| [YOLOv12n-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt) | 224             | 71.7           | 90.5           | 1.27                                   | 2.9                | 0.5               |
| [YOLOv12s-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12s-cls.pt) | 224             | 76.4           | 93.3           | 1.52                                   | 7.2                | 1.5               |
| [YOLOv12m-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12m-cls.pt) | 224             | 78.8           | 94.4           | 2.03                                   | 12.7               | 4.5               |
| [YOLOv12l-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12l-cls.pt) | 224             | 79.5           | 94.5           | 2.73                                   | 16.8               | 6.2               |
| [YOLOv12x-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12x-cls.pt) | 224             | 80.1           | 95.3           | 3.64                                   | 35.5               | 13.7              |

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

model = YOLO('yolov12{n/s/m/l/x}.pt')
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

model = YOLO('yolov12{n/s/m/l/x}.pt')
model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}.pt')
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
