from ultralytics import YOLO
import torch
import cv2
from COCO_Dataset import className
import math
from sort import *
import numpy as np
import cvzone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("../yoloWeights/yolov8n.pt") 
model = model.to(device)

wCam, hCam = 640, 360
cap = cv2.VideoCapture("../Videos/people.mp4")
cap.set(3, wCam)
cap.set(4, hCam)

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
# 計數線擺放位置
upstair_limits = [50,100,150,100] # x1,y,x2 y
downstair_limits = [300,300,420,300] # x1,y,x2 y
# max_age: 如果一個追蹤軌跡在連續 15 幀都沒有再次被檢測到相關的物體，系統就會將這個追蹤視為失效或者丟棄。
# 如果 max_age 設定太小，追蹤軌跡在沒有再次被檢測到的情況下會很快失效。這可能導致過早地結束追蹤，即使物體實際上仍在畫面中存在
# 如果 max_age 設定得太大，即使物體已經離開畫面或不再需要追蹤，但系統仍然會持續維持軌跡。增加了運算資源的浪費
# min_hits: 系統只會開始記錄追蹤某物體的軌跡，當該物體被檢測並連續出現了 3 次或更多次時。
# iou_threshold: 衡量兩個物體（通常是檢測框或者區域）之間的重疊程度， 設置的越高，要求匹配的兩個框之間的重疊度就越高
mask = cv2.imread("mask.png")
mask_resized = cv2.resize(mask, (wCam, hCam))

graphics = cv2.resize(cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED), (300,66))

upstair_id_List = []
downstair_id_List = []
# counter = 1
while True:
    success, img = cap.read()
    img = cv2.resize(img, (wCam, hCam))
    # 偵測特定位置的車輛
    detect_region_img = cv2.bitwise_and(img, mask_resized)
    # print(img.dtype,  mask_resized.dtype)
    # print(img.shape,  mask_resized.shape)
    results = model(detect_region_img, stream = True)
    
    detections=np.empty((0, 5)) # this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    
    # 計上樓人數的線
    cv2.line(img, (upstair_limits[0], upstair_limits[1]), (upstair_limits[2], upstair_limits[3]), (0,0,255), 5)
    cv2.line(img, (downstair_limits[0], downstair_limits[1]), (downstair_limits[2], downstair_limits[3]), (0,0,255), 5)

    cvzone.overlayPNG(img, graphics, pos=[340, 0])


    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            currentClass = className[int(box.cls[0])]
            conf = math.ceil(box.conf[0] * 100) / 100
            if currentClass == "person" and conf > 0.5:
                
                detections = np.vstack(
                    (detections, np.array([x1, y1, x2, y2, conf]))
                )
    # 給予每台車 獨一無二的id，id不會連續每個數字都有，因為一些偵測問題，但沒有關係
    track_bbs_ids = tracker.update(detections)
    for track in track_bbs_ids:
        x1, y1, x2, y2, id = track 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


        # 每輛車都給一個中心圓，用於當中心圓通過計數線時，計數線變色
        people_top = (x1+(x2-x1)//2, y1+10)
        cv2.circle(img, people_top, 2, (0,0,255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
        
        # 上樓通過 計數線叢紅變綠
        if (upstair_limits[0] < people_top[0] < upstair_limits[2]) and (upstair_limits[1]-15 <people_top[1] < upstair_limits[1] +15):
            # 只記錄獨一無二的id數，作為通過車輛數
            if upstair_id_List.count(id) == 0:
                upstair_id_List.append(id)
                cv2.line(img, (upstair_limits[0], upstair_limits[1]), (upstair_limits[2], upstair_limits[3]), (0,255,0), 5)
            
            # cv2.putText(img, str(id), (x1, y1), cv2.FONT_HERSHEY_PLAIN,3, (0,255,0), 2)
        
        # 下樓通過 計數線叢紅變綠
        if (downstair_limits[0] < people_top[0] < downstair_limits[2]) and (downstair_limits[1]-15 <people_top[1] < downstair_limits[1] +15):
            # 只記錄獨一無二的id數，作為通過車輛數
            if downstair_id_List.count(id) == 0:
                downstair_id_List.append(id)
                cv2.line(img, (downstair_limits[0], downstair_limits[1]), (downstair_limits[2], downstair_limits[3]), (0,255,0), 5)

            # cv2.putText(img, str(id), (x1, y1), cv2.FONT_HERSHEY_PLAIN,3, (0,255,0), 2)
        
    # 上樓計數器
    cv2.putText(img, str(len(upstair_id_List)), (440, 50), cv2.FONT_HERSHEY_PLAIN,4, (0,0,0), 5) 
    # 下樓計數器
    cv2.putText(img, str(len(downstair_id_List)), (580, 50), cv2.FONT_HERSHEY_PLAIN,4, (0,0,0), 5) 

    cv2.imshow("video", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
