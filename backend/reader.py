## file only for testing!! 

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# --- Load model ---
num_classes = 29  # Adjust to your dataset
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("asl_model_best.pth", map_location="cpu"))
model.eval()

# --- Class names (example A-Z + extra) ---
class_names = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']  # A-Z + 3 extras

# --- Image preprocessing ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Skin color mask ---
def get_skin_mask(hsv):
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

# --- Start camera ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = get_skin_mask(hsv)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand = max(contours, key=cv2.contourArea) if contours else None
    bbox = None

    if hand is not None and cv2.contourArea(hand) > 1000:
        cv2.drawContours(frame, [hand], -1, (0, 255, 255), 2)

        # Fingertip dots using convexity defects
        hull = cv2.convexHull(hand, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(hand, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand[s][0])
                    end = tuple(hand[e][0])
                    far = tuple(hand[f][0])
                    cv2.circle(frame, start, 8, (0, 255, 0), -1)
                    cv2.circle(frame, end, 8, (0, 255, 0), -1)
                    cv2.circle(frame, far, 5, (0, 0, 255), -1)

        # Bounding box around hand
        x, y, w, h = cv2.boundingRect(hand)
        bbox = frame[y:y+h, x:x+w]

    # --- Predict ASL letter if bbox is valid ---
    if bbox is not None and bbox.size > 0:
        try:
            input_tensor = preprocess(bbox).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                pred_class = class_names[predicted.item()]
                cv2.putText(frame, f'Predicted: {pred_class}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            cv2.putText(frame, f'Prediction error', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Recognition (No MediaPipe)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
