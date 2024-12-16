from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加載模型
model = YOLO('yolov8n-pose.pt')  # 載入預訓練模型

# 推論函數
def detect_pose(image_path):
    # 加載圖片
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用模型進行推論
    results = model(img_rgb)
    
    # 提取關鍵點資訊
    keypoints = {}
    for result in results:
        keypoints_result = result.keypoints  # 獲取關鍵點
        keypoint_coords = keypoints_result.xy[0].cpu().numpy()  # 轉換為 numpy array
        
        # 提取特定關鍵點座標
        keypoints["nose"] = tuple(keypoint_coords[0]) if len(keypoint_coords) > 0 else (None, None)
        keypoints["left_shoulder"] = tuple(keypoint_coords[5]) if len(keypoint_coords) > 5 else (None, None)
        keypoints["right_shoulder"] = tuple(keypoint_coords[6]) if len(keypoint_coords) > 6 else (None, None)
        keypoints["left_hip"] = tuple(keypoint_coords[11]) if len(keypoint_coords) > 11 else (None, None)
        keypoints["right_hip"] = tuple(keypoint_coords[12]) if len(keypoint_coords) > 12 else (None, None)
        
        # 輸出特定關鍵點座標 (可以根據需求儲存或進一步處理)
        print(f"Nose: {keypoints['nose']}")
        print(f"Left Shoulder: {keypoints['left_shoulder']}")
        print(f"Right Shoulder: {keypoints['right_shoulder']}")
        print(f"Left Hip: {keypoints['left_hip']}")
        print(f"Right Hip: {keypoints['right_hip']}")
        
        # 繪製關鍵點和連線
        img_with_keypoints = draw_keypoints(img_rgb, keypoint_coords)
        
        # print(f"total:{keypoints}")
        
        # 計算重心
        body_center=calculate_centers(keypoints)
        
        # 畫上重心
        cv2.circle(img_with_keypoints, (int(body_center[0]), int(body_center[1])), 5, (255, 0, 0), -1)
        
        # 畫上重心鉛錘線
        cv2.line(img_with_keypoints, (int(body_center[0]), 0), (int(body_center[0]), img.shape[0]), (255, 0, 0), 2) 

    
    # 顯示圖片
    plt.imshow(img_with_keypoints)
    plt.axis('off')
    plt.show()



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

# 主程式
if __name__ == "__main__":
    # image_path = "data/images/sample_image.jpg"  # 測試圖片路徑
    image_path = "data/images/image.jpg"  # 測試圖片路徑
    detect_pose(image_path)
