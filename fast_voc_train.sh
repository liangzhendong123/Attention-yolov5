python3 train.py --data voc/VOC.yaml --weights weights/yolov5s.pt --cfg voc/yolov5s-senet.yaml --hyp voc/hyp.scratch.yaml --batch-size 32 --cache-images --img-size 512 --device $1 --epochs 150
