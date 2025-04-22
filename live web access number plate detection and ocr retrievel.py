
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import csv
import datetime

# === MODEL & OUTPUT PATHS ===
model_path = r"C:\Users\vgunt\Documents\FOLDER\weights\best.pt"  # ✅ Your local YOLOv8 model
csv_output_path = r"C:\Users\vgunt\Documents\FOLDER\plate_log.csv"

# === LOAD MODELS ===
print("🚀 Loading YOLOv8...")
yolo_model = YOLO(model_path)

print("🔤 Loading PaddleOCR...")
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# === OPEN WEBCAM ===
cap = cv2.VideoCapture(0)  # 0 is default laptop camera

detected_plates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === YOLO DETECTION ===
    results = yolo_model.predict(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        roi = frame[int(y1):int(y2), int(x1):int(x2)]

        if roi.size == 0:
            continue

        try:
            ocr_out = ocr_model.ocr(roi)

            if ocr_out and isinstance(ocr_out[0], list):
                for line in ocr_out[0]:
                    text = line[1][0]
                    conf = line[1][1]

                    if conf >= 0.6:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🧾 [{timestamp}] Plate: {text} | Conf: {round(conf, 2)}")

                        # Draw detection
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        detected_plates.append({
                            "timestamp": timestamp,
                            "text": text,
                            "confidence": round(conf, 2),
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        })

        except Exception as e:
            print(f"❌ OCR error: {e}")
            continue

    # === DISPLAY FRAME ===
    cv2.imshow("📸 License Plate Detector (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === SAVE TO CSV ===
if detected_plates:
    with open(csv_output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=detected_plates[0].keys())
        writer.writeheader()
        writer.writerows(detected_plates)
    print(f"✅ Plate log saved to: {csv_output_path}")
else:
    print("⚠️ No plates detected.")

