from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import sqlite3


con = sqlite3.connect("./tablice.db")

cur = con.cursor()


# Define the same model class you used during training
class GrayResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=None)  # Don't re-download
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load model
OCRmodel = GrayResNet18(num_classes=37)
OCRmodel.load_state_dict(torch.load("resnet18_gray_anpr.pth", map_location="cpu"))
OCRmodel.eval()
labels = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P',  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']
# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# Load a COCO-pretrained YOLO11n model
model = YOLO("./best.pt")

# find corner points
def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],        # top-left
        pts[np.argmin(diff)],     # top-right
        pts[np.argmax(s)],        # bottom-right
        pts[np.argmax(diff)]      # bottom-left
    ], dtype="float32")

cap = cv2.VideoCapture("rtsp://admin:NemciMePokrali1@192.168.1.2:554/Streaming/channels/101")
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model.predict(frame, verbose=False, stream=False, save=False, show=False)
    result = results[0]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for box in result.boxes:
        if box.xyxy[0][1].item() == box.xyxy[0][3].item() or box.xyxy[0][0].item() == box.xyxy[0][2].item():
            continue
        croppedImage = frame[int(box.xyxy[0][1].item()):int(box.xyxy[0][3].item()),
                       int(box.xyxy[0][0].item()):int(box.xyxy[0][2].item())]
        gaussian_blur = cv2.GaussianBlur(croppedImage, (5, 5), 0, sigmaY=0)
        cannyGauss = cv2.Canny(gaussian_blur, 20, 40)
        dilategaussian = cv2.dilate(cannyGauss, None, iterations=2)
        contoursGauss, _ = cv2.findContours(dilategaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contoursGauss:
            continue
        largest_contour = max(contoursGauss, key=cv2.contourArea)

        mask = np.zeros_like(croppedImage)

        # Step 2: Draw filled contour in white
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)  # white filled

        # Step 3: Erode and dilate the mask (morphological operations)
        eroded_mask = cv2.erode(mask, None, iterations=20)
        dilated_mask = cv2.dilate(eroded_mask, None, iterations=20)

        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours detected")
            continue

        screen_cnt = None
        approx = None

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                screen_cnt = approx
                break

        if screen_cnt is None:
            # fallback to largest contour and force 4-point approximation
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Force the approximation to 4 points (convex hull)
            approx = cv2.convexHull(cnt)
            if len(approx) > 4:
                approx = cv2.approxPolyDP(approx, 0.05 * cv2.arcLength(approx, True), True)

            if len(approx) != 4:
                print("Even fallback contour does not give 4 points.")
                continue

        screen_cnt = approx

        if len(approx) != 4:
            print("Could not find 4-point contour")
            continue

        src_pts = order_points(approx)
        w, h = 400, 100  # Desired output size (adjust as needed)
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(croppedImage, M, (w, h))
        warped = warped[10:100, 0:400]
        binary_img = cv2.adaptiveThreshold(
            warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        letterContours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letterContours = [c for c in letterContours if
                          600 < cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] < 3600 and
                          75 > cv2.boundingRect(c)[2] > 8 and
                          27 < cv2.boundingRect(c)[3] < 95]

        sorted_contours = sorted(letterContours, key=lambda cnt: cv2.boundingRect(cnt)[0])
        color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        num0 = 0
        if len(sorted_contours) > 1:
            universal_bounding_box = cv2.boundingRect(sorted_contours[1])
            bounding_boxes = []
            for i in range(0, sorted_contours.__len__()):
                if (i > 0 and cv2.boundingRect(sorted_contours[i])[3] < universal_bounding_box[3] - 5 and
                        cv2.boundingRect(sorted_contours[i - 1])[3] > universal_bounding_box[3] - 5):
                    x, y, w, _ = cv2.boundingRect(sorted_contours[i])
                    h = universal_bounding_box[3]
                    bounding_boxes.append((x, universal_bounding_box[1], w, h))

                elif not (i < len(sorted_contours) - 1 and cv2.boundingRect(sorted_contours[i])[3] <
                          universal_bounding_box[3] - 5 < cv2.boundingRect(sorted_contours[i + 1])[3]):
                    bounding_boxes.append(cv2.boundingRect(sorted_contours[i]))
            initialText = ''
            for (x, y, w, h) in bounding_boxes:
                # Draw rectangle in pink (BGR)
                cv2.rectangle(color_img, (x, y), (x + w, y + h), (255, 0, 255), 1)
                img = Image.fromarray(color_img[y:y + h, x:x + w])
                img = transform(img).unsqueeze(0)
                output = OCRmodel(img)
                predicted = torch.argmax(output, dim=1).item()
                initialText = initialText + labels[predicted]

            res = cur.execute("SELECT text FROM LicensePlate WHERE expires IS NULL OR expires < CURRENT_TIMESTAMP;")
            licensePlates = res.fetchall()
            licensePlates = list(licensePlate[0] for licensePlate in licensePlates)
            initialText =  initialText.replace("-", "")
            print(licensePlates)
            print(initialText)
            if initialText in licensePlates:
                print("License plate detected")

















