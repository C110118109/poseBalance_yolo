import time
from ultralytics import YOLO
import cv2
import numpy as np

# 加載模型
model = YOLO('yolov8n-pose.pt')  # 載入 YOLOv8 姿勢辨識模型

# 計算肩膀中心、骨盆中心及人體重心
def calculate_centers(keypoints):
    if None in keypoints.values():
        return None, None, None
    
    shoulder_center = ((keypoints["left_shoulder"][0] + keypoints["right_shoulder"][0]) // 2,
                       (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) // 2)

    hip_center = ((keypoints["left_hip"][0] + keypoints["right_hip"][0]) // 2,
                  (keypoints["left_hip"][1] + keypoints["right_hip"][1]) // 2)

    body_center = ((shoulder_center[0] + hip_center[0]) // 2,
                   (shoulder_center[1] + hip_center[1]) // 2)

    return body_center

# 繪製關鍵點與連線
def draw_keypoints(img, keypoints):
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    # 繪製關鍵點
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # 綠點

    # 繪製連線
    for connection in connections:
        try:
            start = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            end = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(img, start, end, (0, 0, 255), 2)  # 紅線
        except IndexError:
            print(f"Connection {connection} is out of range for keypoints list.")

    return img

# 處理影片
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)  # 設定視窗

    while cap.isOpened():
        st = time.time()  # 計時開始
        ret, frame = cap.read()
        if not ret:
            break

        # 進行姿勢辨識
        results = model(frame)

        for result in results:
            keypoints = {}
            keypoint_coords = result.keypoints.xy[0].cpu().numpy()  # 獲取關鍵點座標
            
            # 提取特定關鍵點
            keypoints["nose"] = tuple(keypoint_coords[0]) if len(keypoint_coords) > 0 else (None, None)
            keypoints["left_shoulder"] = tuple(keypoint_coords[5]) if len(keypoint_coords) > 5 else (None, None)
            keypoints["right_shoulder"] = tuple(keypoint_coords[6]) if len(keypoint_coords) > 6 else (None, None)
            keypoints["left_hip"] = tuple(keypoint_coords[11]) if len(keypoint_coords) > 11 else (None, None)
            keypoints["right_hip"] = tuple(keypoint_coords[12]) if len(keypoint_coords) > 12 else (None, None)

            # 計算重心
            body_center = calculate_centers(keypoints)

            # 繪製結果
            frame = draw_keypoints(frame, keypoint_coords)

            # 畫重心及鉛垂線
            if body_center:
                cv2.circle(frame, (int(body_center[0]), int(body_center[1])), 5, (255, 0, 0), -1)
                cv2.line(frame, (int(body_center[0]), 0), (int(body_center[0]), frame.shape[0]), (255, 0, 0), 2)

        # 顯示FPS
        et = time.time()
        FPS = round(1 / (et - st), 1)
        cv2.putText(frame, f"FPS: {FPS}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 顯示影像
        cv2.imshow('Pose Detection', frame)
        
        # ESC鍵退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 主程式
if __name__ == "__main__":
    # video_path = "data/videos/sample_video.mp4"  # 測試影片路徑
    video_path = "data/videos/sample.mp4"  # 測試影片路徑
    process_video(video_path)
