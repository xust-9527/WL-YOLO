import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('runs/train/yolov8s/weights/best.pt')
    model = YOLO('runs/prune/yolov8s/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )