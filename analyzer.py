import time
import cv2
import torch
import torch.nn as nn
import numpy as np

# https://github.com/WongKinYiu/ScaledYOLOv4 kütüphanesindeki kodlar kullanılmaktadır
from models.experimental import attempt_load
from utils.general import (check_img_size, non_max_suppression, scale_coords, plot_one_box)
from utils.torch_utils import select_device
from utils.datasets import letterbox

weight_loc = './weights/best.pt' # Modelin ağırlık dosyasının konumu
image_size = 1024 # Modelin işleyeceği görüntü boyutu
_device = '' # Kullanılacak cihaz (GPU için boş bırakın)
#save_loc = None # Hücrelerin işaretlendiği fotoğrafın kaydedileceği yer (Fotoğrafı kaydetmemek için NONE yapın)
#_augment = False # Test time augmentation
#conf_thres = 0.4 # object confidence threshold
#iou_thres = 0.5 # IOU threshold for NMS
#agnostic_nms = False # class-agnostic NMS

# Initialize
device = select_device(_device) # GPU veya CPU seçimi
half = False #device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weight_loc, map_location=device)  # load FP32 model
imgsz = check_img_size(image_size, s=model.stride.max())  # Seçilen görüntü boyutu 32'nin katı mı kontrol et

if half:
    model.half()  # to FP16

def analyze(img0, save_loc=None, _augment=False, conf_thres=0.4, iou_thres=0.5, agnostic_nms=False):
    with torch.no_grad():
        cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # TODO test if transpose blow is better
        # Read Image - LoadImages
        img = letterbox(img0, new_shape=imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        names = ["canli", "olu"]
        colors = [(0, 0, 255), (255, 0, 0)]

        # Run inference
        #t0 = time.time()

        # ! check if it brokes anything !
        #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img 
        #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        
        #for path, img, im0s, vid_cap in dataset:
        # path image path
        # img letterboxed img
        # im0s original image
        # vid_cap None
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        pred = model(img, augment=_augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
        #t2 = time_synchronized()
        cell_nums = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print(det)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique(): # class nums are here
                    cell_nums.append((det[:, -1] == c).sum())  # detections per class

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s' % (names[int(cls)])
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

        # Print time (inference + NMS)
        #print('%sDone. (%.3fs)' % (s, t2 - t1))

        # Save results (image with detections)
        if save_loc is not None:
            save_path = f"{save_loc}\\{int(time.time())}.png"
            cv2.imwrite(save_path, img0)
            img0 = save_path

        print(f"Cell num:{cell_nums}")
        if len(cell_nums) == 0:
            return img0, 0, 0
        if len(cell_nums) == 1:
            return img0, cell_nums[0].item(), 0
        if len(cell_nums) == 2:
            return img0, cell_nums[0].item(), cell_nums[1].item()
        #print('Done. (%.3fs)' % (time.time() - t0))
