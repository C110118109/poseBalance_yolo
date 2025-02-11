import time
from ultralytics import YOLO
import cv2
import numpy as np

# 加載 YOLO 模型
pose_model = YOLO('yolov8n-pose.pt')  # 用於偵測人的骨架和關鍵點
action_model = YOLO('best.pt')       # 用於偵測動作

# 設定影片來源
video_path = "data/videos/3 (2).mp4"  # 替換為你的影片路徑
output_path = "new_output.mp4"
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Pose Judgment', cv2.WINDOW_NORMAL)  # 設定視窗

# 取得影片資訊
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 原影片的每秒幀數

# 初始化影片寫入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式的編碼器
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 繪製偵測框和動作標籤
def draw_bounding_box_and_label(img, bbox, action_label, confidence, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    # 動態計算框和文字樣式
    radius = int(min(img.shape[0], img.shape[1]) * 0.005555)
    thickness = max(1, radius // 2)
    fontsize = min(img.shape[0], img.shape[1]) / 850

    # 畫出偵測框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # 顯示標籤和置信度
    label_text = f"{action_label} ({confidence:.2f})"
    cv2.putText(img, label_text, (x1, max(y2 + 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness, cv2.LINE_AA)

# 繪製關鍵點與連線
def draw_keypoints(image, keypoints, radius, thickness):
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    # 繪製關鍵點
    for x, y in keypoints:
        cv2.circle(image, (int(x), int(y)), radius, (255, 255, 255), -1)  # 白點

    # 繪製連線
    for connection in connections:
        try:
            start = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            end = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(image, start, end, (255, 0, 255), thickness)  # 紅線
        except IndexError:
            print(f"Connection {connection} is out of range for keypoints list.")

    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 步驟 1: 先用 YOLOv8 偵測人物框
    results_pose = pose_model(frame)

    for result in results_pose:
        for i in range(len(result.boxes.xyxy)):  # 遍歷所有檢測到的人
            bbox = result.boxes.xyxy[i].cpu().numpy()  # 取得人物框
            x1, y1, x2, y2 = map(int, bbox)

            # 取得圖像尺寸
            height, width, _ = frame.shape  
            radius = int(min(width, height) * 0.005555)
            thickness = max(1, radius // 2)

            # 步驟 2: 從框內裁剪人物，傳給動作模型
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:  # 確保裁切有效
                continue

            action_results = action_model(person_crop)
            if len(action_results) > 0 and len(action_results[0].boxes) > 0:
                action_box = action_results[0].boxes[0]
                class_id = int(action_box.cls[0].cpu().numpy())
                confidence = action_box.conf[0].cpu().numpy()
                action_label = action_model.names[class_id]

                # 繪製偵測框與動作標籤
                draw_bounding_box_and_label(frame, bbox, action_label, confidence)

                # 步驟 3: 如果有識別到動作，則畫出關鍵點
                keypoint_coords = result.keypoints.xy[i].cpu().numpy()  # 取得該人物的關鍵點
                frame = draw_keypoints(frame, keypoint_coords, radius, thickness)

    # 顯示處理後的畫面
    cv2.imshow("Pose and Action Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    out.write(frame)

# 釋放資源
cap.release()
cv2.destroyAllWindows()
