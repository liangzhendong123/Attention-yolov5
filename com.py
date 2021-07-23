import torch
from models.yolo import Model
from utils.torch_utils import intersect_dicts

if __name__ == "__main__":
    ckpt = torch.load("weights/yolov5s.pt")
    model = Model("insulator/yolov5s-senet.yaml", ch=3, nc=1)
    state_dict = ckpt['model'].float().state_dict()
    #ckpt['model'] = model

    #torch.save(ckpt, "senet.pt")
    da = state_dict
    db = model.state_dict()

    for key, value in db.items():
        print(key)
