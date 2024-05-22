import cv2
import numpy as np
import torch
import utils
from ultralytics import YOLO

model = YOLO("best.pt")
#Read video
cap = cv2.VideoCapture("0522.mp4")
out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (1920, 1080))

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    scale_percent = 80 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # img = utils.resize_image(img, width, height)

    # Inference
    results = utils.inference(img, model)
    # Draw results
    imgP = utils.imgProcess(img.copy(), results.boxes)

    # to BGR
    imgP = cv2.cvtColor(imgP, cv2.COLOR_RGB2BGR)

    cv2.imshow('Frame', imgP)
    out.write(imgP)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
print("Done!")
cv2.destroyAllWindows()