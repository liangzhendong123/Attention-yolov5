# 在insulator数据集上训练yolov5s 500轮
python3 train.py --weights insulator/yolov5s.pt --cfg insulator/yolov5s.yaml --data insulator/insulator.yaml --hyp insulator/hyp.finetune.yaml --batch-size 16 --cache-images --epochs 500 --adam --resume

# 在insulator数据集上训练yolov5s-senet 500轮
python3 train.py --weights insulator/yolov5s.pt --cfg insulator/yolov5s-senet.yaml --data insulator/insulator.yaml --hyp insulator/hyp.finetune.yaml --batch-size 16 --cache-images --epochs 500 --adam

# 在voc2007数据集上训练yolov5s 150轮
python3 train.py --weights voc/yolov5s.pt --cfg voc/yolov5s.yaml --data voc/VOC.yaml --hyp voc/hyp.finetune.yaml --batch-size 16 --cache-images --epochs 150 --adam

# 在voc2007数据集上训练yolov5s-senet 150轮
python3 train.py --weights voc/yolov5s.pt --cfg voc/yolov5s-senet.yaml --data voc/VOC.yaml --hyp voc/hyp.finetune.yaml --batch-size 16 --cache-images --epochs 150 --adam
