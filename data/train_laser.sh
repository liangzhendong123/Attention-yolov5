#!/bin/sh

python3 train.py --weights weights/yolov5s.pt --cfg models/yolov5s.yaml --data data/laser.yaml --batch-size 4 --img-size 1080 --multi-scale --epochs 300 --cache-images
