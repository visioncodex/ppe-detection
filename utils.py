import cv2
import numpy as np
import matplotlib.pyplot as plt

colors = {
    0: (231, 76, 60),
    1: (231, 76, 60),
    2: (230, 126, 34),
    3: (46, 204, 113),
    4: (46, 204, 113)
}
labels = {0: "himoya yo'q", 1: "nimcha yo'q", 2: "ishchi", 3: "bosh himoyasi", 4: "nimcha"}

#   0: No helmet 
#   1: No vest 
#   2: Person
#   3: helmet
#   4: vest

def drawImg(img: np.ndarray, xmin, ymin, xmax, ymax, name, label):
    start_point = (int(xmin), int(ymin))
    end_point = (int(xmax), int(ymax))
    color = colors[label]

    img = cv2.rectangle(
        img,
        start_point, 
        end_point,
        color=color,
        thickness = 3
        )
    text_size, _ = cv2.getTextSize(
        name,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.2,
        thickness=1
        )
    text_w, text_h = text_size

    img = cv2.rectangle(
        img,
        (int(xmin), int(ymin) - text_h - 5),
        (int(xmin) + text_w, int(ymin)),
        color=color,
        thickness=-1
        )
    img = cv2.putText(
        img,
        name,
        org=(int(xmin), int(ymin) - 5),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.2,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.FILLED
        )
    return img

    
def imgProcess(img, boxes):
    classes = boxes.cls
    conf = boxes.conf
    xyxy = boxes.xyxy
    for i, box in enumerate(xyxy):
        confidence = float(conf[i].cpu().detach().numpy())
        label = classes[i].cpu().detach().numpy()
        box = box.cpu().detach().numpy()
        xmin, ymin, xmax, ymax = box
        if confidence > 0.5:
            name = labels[int(label)] + ' ' + str(round(confidence, 2))
            img = drawImg(img, xmin, ymin, xmax, ymax, name, int(label))
        
    return img


def inference(img, model):
    # Load model
    # inference
    results = model.predict(img)
    return results[0]

def resize_image(image, w,h):
    """
    Image resize with opencv
    """
    image = cv2.resize(image,(w, h))
    return image