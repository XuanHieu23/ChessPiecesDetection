import os
import argparse
import cv2
from ultralytics import YOLO

def draw_boxes(img, boxes, names):
    for b in boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - tsize[1] - 6), (x1 + tsize[0] + 6, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img

def main(args):
    model = YOLO(args.model)
    names = model.model.names  # lấy tên lớp trực tiếp từ model

    os.makedirs(args.output, exist_ok=True)

    sources = []
    if os.path.isdir(args.source):
        for f in os.listdir(args.source):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                sources.append(os.path.join(args.source, f))
    elif os.path.isfile(args.source):
        sources = [args.source]
    else:
        raise FileNotFoundError(args.source)


    for src in sources:
        res = model.predict(source=src, imgsz=args.imgsz, conf=args.conf, iou=0.45, device=args.device)
        # res is list, take first
        boxes = res[0].boxes
        img = cv2.imread(src)
        img_out = draw_boxes(img, boxes, names)
        out_path = os.path.join(args.output, os.path.basename(src))
        cv2.imwrite(out_path, img_out)
        print("Saved:", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True, help="image file or folder")
    p.add_argument("--output", default="preds")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default="0", help="cuda device or cpu")
    args = p.parse_args()
    main(args)
