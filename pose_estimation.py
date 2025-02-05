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
        bbox = result.boxes.xyxy[0].cpu().numpy()  # 獲取檢測框座標 [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # # 獲取圖像解析度
        # height, width  , _ = img.shape  

        # # 動態計算圓的半徑和邊框粗細
        # radius = int(min(width, height) * 0.005555)  # 取最小邊長的 0.5% 作為半徑
        # thickness = max(1, radius // 2)
        # fontsize = min(width, height) / 850
    
        
        
        # 提取特定關鍵點座標
        keypoints["nose"] = tuple(keypoint_coords[0]) if len(keypoint_coords) > 0 else (None, None)
        keypoints["left_shoulder"] = tuple(keypoint_coords[5]) if len(keypoint_coords) > 5 else (None, None)
        keypoints["right_shoulder"] = tuple(keypoint_coords[6]) if len(keypoint_coords) > 6 else (None, None)
        keypoints["left_ankle"] = tuple(keypoint_coords[15]) if len(keypoint_coords) > 15 else (None, None)
        keypoints["right_ankle"] = tuple(keypoint_coords[16]) if len(keypoint_coords) > 16 else (None, None)
        
        # 輸出特定關鍵點座標 (可以根據需求儲存或進一步處理)
        print(f"Nose: {keypoints['nose']}")
        print(f"Left Shoulder: {keypoints['left_shoulder']}")
        print(f"Right Shoulder: {keypoints['right_shoulder']}")
        print(f"Left Ankle: {keypoints['left_ankle']}")
        print(f"Right Ankle: {keypoints['right_ankle']}")
        
        # 繪製關鍵點和連線
        img_with_keypoints = draw_keypoints(img_rgb, keypoint_coords)
        
        # print(f"total:{keypoints}")
        
        # 計算重心
        body_center=calculate_body_center(keypoints)
        
        # 找出肩膀中心點
        shoulder_center = find_shoulder_center(keypoints)
        
        # 畫重心
        if body_center:
            cv2.circle(img_rgb, (int(body_center[0]), int(body_center[1])), 5, (255, 0, 0), -1)
        print(f"body_center: {body_center}")
        
        # 繪製肩膀中心點
        if body_center:
            cv2.circle(img_rgb, (int(shoulder_center[0]), int(shoulder_center[1])), 5, (255, 0, 0), -1)
        
        if body_center and shoulder_center:
            # 計算斜率和截距
            slope = calculate_slope(shoulder_center, body_center)
            intercept = calculate_intercept(slope, shoulder_center)

            # 計算延長線與檢測框邊界的交點
            if slope is not None:
                # 與上邊 (y1) 的交點
                x_at_y1 = (y1 - intercept) / slope if slope != 0 else shoulder_center[0]
                # 與下邊 (y2) 的交點
                x_at_y2 = (y2 - intercept) / slope if slope != 0 else shoulder_center[0]
            else:
                # 垂直線的情況
                x_at_y1 = shoulder_center[0]
                x_at_y2 = shoulder_center[0]
            # 限制 x 範圍在 [x1, x2] 內
            x_at_y1 = max(x1, min(x2, x_at_y1))
            x_at_y2 = max(x1, min(x2, x_at_y2))
            
            start_point=int(x_at_y1), int(y1)
            end_point=int(x_at_y2), int(y2)
            cv2.line(img_rgb, start_point, end_point, (255, 0, 0),2)
            
    
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

# # 兩大重心連線並延長至檢測框
# def extend_line_to_bbox(shoulder_center, body_center, bbox):
#     """
#     根據兩點的連線，計算該線延長後與檢測框的交點
#     :param point1: 第一點的座標 (x1, y1)
#     :param point2: 第二點的座標 (x2, y2)
#     :param bbox: 檢測框的座標 (x1, y1, x2, y2)
#     :return: 延長線與檢測框的交點 (start_point, end_point)
#     """
#     x1, y1, x2, y2 = bbox

#     # 計算斜率和截距
#     slope = calculate_slope(shoulder_center, body_center)
#     intercept = calculate_intercept(slope, shoulder_center)

#     # 計算延長線與檢測框邊界的交點
#     if slope is not None:
#         # 與上邊 (y1) 的交點
#         x_at_y1 = (y1 - intercept) / slope if slope != 0 else shoulder_center[0]
#         # 與下邊 (y2) 的交點
#         x_at_y2 = (y2 - intercept) / slope if slope != 0 else shoulder_center[0]
#     else:
#         # 垂直線的情況
#         x_at_y1 = shoulder_center[0]
#         x_at_y2 = shoulder_center[0]

#     # 限制 x 範圍在 [x1, x2] 內
#     x_at_y1 = max(x1, min(x2, x_at_y1))
#     x_at_y2 = max(x1, min(x2, x_at_y2))

#     return (int(x_at_y1), int(y1)), (int(x_at_y2), int(y2))


# 計算人體重心座標
def calculate_body_center(keypoints):
    valid_keypoints = [v for v in keypoints.values() if v is not None and None not in v]
    
    if not valid_keypoints:
        return None
    
    X_c = np.mean([kp[0] for kp in valid_keypoints])  # 平均 X 座標
    Y_c = np.mean([kp[1] for kp in valid_keypoints])  # 平均 Y 座標
    
    return X_c, Y_c

def find_shoulder_center(keypoints):
    L_shoulder = keypoints.get("left_shoulder")
    R_shoulder=keypoints.get("right_shoulder")
    
    # 確保肩膀的座標有效
    if L_shoulder is None or R_shoulder is None:
        return None
    if None in L_shoulder or None in R_shoulder:
        return None
    
    # 計算中心點
    X_center = (L_shoulder[0] + R_shoulder[0]) / 2
    Y_center = (L_shoulder[1] + R_shoulder[1]) / 2
    return X_center,Y_center

# 計算交點
def calculate_intersection(slope1, intercept1, slope2, intercept2):
    """
    計算兩條直線的交點
    :param slope1: 第一條直線的斜率
    :param intercept1: 第一條直線的截距
    :param slope2: 第二條直線的斜率
    :param intercept2: 第二條直線的截距
    :return: (x, y) 交點座標
    """
    if slope1 is None:  # 第一條是垂直線
        x = -intercept1  # intercept1 在垂直線時為 x 座標
        y = slope2 * x + intercept2
    elif slope2 is None:  # 第二條是垂直線
        x = -intercept2
        y = slope1 * x + intercept1
    else:
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1
    return x, y

# 計算截距
def calculate_intercept(slope, point):
    """
    計算截距 b
    :param slope: 直線的斜率
    :param point: 直線上一點的座標 (x, y)
    :return: 截距 b，如果斜率為 None（垂直線），返回 None
    """
    if slope is None:  # 垂直線無法計算截距
        return None
    x, y = point
    return y - slope * x

# 計算斜率
def calculate_slope(point1, point2):
    """
    計算兩點之間的斜率
    :return: 斜率 m，如果是垂直線則返回 None
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:  # 垂直線的情況
        return None
    return round((y2 - y1) / (x2 - x1),1)


# 主程式
if __name__ == "__main__":
    # image_path = "data/images/sample_image.jpg"  # 測試圖片路徑
    image_path = "data/images/image.jpg"  # 測試圖片路徑
    # image_path = "data/images/court1.jpg"  # 測試圖片路徑
    detect_pose(image_path)
