if [ "$1" = "pose" ]; then
    yolo pose train data=coco-pose.yaml model=$2.yaml name=trainval/$2 imgsz=640 batch=64 epochs=100 close_mosaic=0 workers=8 device=0 pretrained=true
elif  [ "$1" = "segment" ]; then
    yolo sement train data=coco.yaml model=$2.yaml name=trainval/$2 imgsz=640 batch=64 epochs=100 close_mosaic=10 workers=8 device=0 pretrained=true
else    
    yolo detect train data=coco.yaml model=$2.yaml name=trainval/$2 imgsz=640 batch=64 epochs=100 close_mosaic=10 workers=8 device=0 pretrained=true
fi