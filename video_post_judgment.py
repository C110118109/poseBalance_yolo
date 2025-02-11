import time
from ultralytics import YOLO
import cv2
import numpy as np

# 加載 YOLO 模型
pose_model = YOLO('yolov8n-pose.pt')  # 用於偵測人的骨架和關鍵點
action_model = YOLO('best.pt')       # 用於偵測動作

state_counts = {
    "Balanced": 0,
    "Leaning Left": 0,
    "Leaning Right": 0,
    "Unbalanced": 0
}

# 設定影片來源
video_path = "data/videos/1 (1).mp4"  # 替換為你的影片路徑
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

# 計算人體重心座標
def calculate_body_center(keypoints):
    """
    計算人體重心座標 (X_c, Y_c)，忽略值為 [0, 0] 的關鍵點
    :param keypoints: numpy array，包含所有關鍵點的 [x, y] 座標
    :return: (X_c, Y_c) 重心座標，或 None 如果所有座標都是 [0, 0]
    """
    if len(keypoints) == 0:
        return None  # 若沒有關鍵點，回傳 None

    # 過濾掉 [0, 0] 的關鍵點
    valid_keypoints = keypoints[(keypoints[:, 0] != 0) | (keypoints[:, 1] != 0)]

    # 若過濾後沒有有效的關鍵點，回傳 None
    if len(valid_keypoints) == 0:
        return None

    # 計算 X_c 和 Y_c
    X_c = np.mean(valid_keypoints[:, 0])  # 有效 x 座標的平均值
    Y_c = np.mean(valid_keypoints[:, 1])  # 有效 y 座標的平均值

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

# 兩大重心連線並延長至檢測框
def extend_line_to_bbox(shoulder_center, body_center, bbox):
    """
    根據兩點的連線，計算該線延長後與檢測框的交點
    :param point1: 第一點的座標 (x1, y1)
    :param point2: 第二點的座標 (x2, y2)
    :param bbox: 檢測框的座標 (x1, y1, x2, y2)
    :return: 延長線與檢測框的交點 (start_point, end_point)
    """
    x1, y1, x2, y2 = bbox

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

    return (int(x_at_y1), int(y1)), (int(x_at_y2), int(y2))

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

# 判斷是否垂直
def are_lines_perpendicular(slope1, slope2):
    """
    檢查兩條線是否互相垂直
    :return: True 如果垂直，False 如果不垂直
    """
    if slope1 is None:  # 第一條線垂直
        return slope2 == 0
    if slope2 is None:  # 第二條線垂直
        return slope1 == 0
    
    num=round(slope1 * slope2,1)

    if num >= -1.2 and num <= -0.8:
        return True
    else:
        return False

# 重心偏向判斷
def determine_balance(body_center, shoulder_center, L_ankle, R_ankle):
    """
    判斷平衡狀態（偏左、偏右或完全不平衡）
    :param body_center: 人體重心 (x, y)
    :param shoulder_center: 肩膀中心點 (x, y)
    :param L_ankle: 左腳踝座標 (x, y)
    :param R_ankle: 右腳踝座標 (x, y)
    :return: 'Balanced', 'Left', 'Right', 'Unbalanced'
    """
    # 重心線斜率與截距
    center_slope = calculate_slope(shoulder_center, body_center)
    center_intercept = calculate_intercept(center_slope, shoulder_center)

    # 腳連線斜率與截距
    foot_slope = calculate_slope(L_ankle, R_ankle)
    foot_intercept = calculate_intercept(foot_slope, L_ankle)

    # 計算重心線與腳連線的交點
    intersection = calculate_intersection(center_slope, center_intercept, foot_slope, foot_intercept)

    if not intersection:
        return "Unbalanced"  # 無交點，表示完全不平衡

    x_inter, _ = intersection
    x_left = min(L_ankle[0], R_ankle[0])
    x_right = max(L_ankle[0], R_ankle[0])

    if x_left <= x_inter <= x_right:
        if x_inter < (x_left + x_right) / 2:  # 靠近左側
            return "Left"
        else:  # 靠近右側
            return "Right"
    else:
        return "Unbalanced"  # 重心線交點不在腳連線範圍內

# ---------------------------------------------------------------------------------------------------------------------------
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
            fontsize = min(width, height) / 850
            
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
                keypoints = {}
                keypoint_coords = result.keypoints.xy[i].cpu().numpy()  # 取得該人物的關鍵點
                frame = draw_keypoints(frame, keypoint_coords, radius, thickness)
                
                # 提取特定關鍵點
                keypoints["nose"] = tuple(keypoint_coords[0]) if len(keypoint_coords) > 0 else (None, None)
                keypoints["left_shoulder"] = tuple(keypoint_coords[5]) if len(keypoint_coords) > 5 else (None, None)
                keypoints["right_shoulder"] = tuple(keypoint_coords[6]) if len(keypoint_coords) > 6 else (None, None)
                keypoints["left_ankle"] = tuple(keypoint_coords[16]) if len(keypoint_coords) > 16 else (None, None)
                keypoints["right_ankle"] = tuple(keypoint_coords[15]) if len(keypoint_coords) > 15 else (None, None)

                # 計算人體重心
                body_center = calculate_body_center(keypoint_coords)
                
                # 找出肩膀中心點
                shoulder_center = find_shoulder_center(keypoints)

                # 畫重心
                if body_center:
                    cv2.circle(frame, (int(body_center[0]), int(body_center[1])), radius, (255, 0, 0), -1)
                    
                
                # 繪製肩膀中心點
                if body_center:
                    cv2.circle(frame, (int(shoulder_center[0]), int(shoulder_center[1])), radius, (255, 0, 0), -1)
                
                # 兩大重心連線
                if body_center and shoulder_center:
                    start_point, end_point = extend_line_to_bbox(shoulder_center, body_center, bbox)
                    cv2.line(frame, start_point, end_point, (255, 0, 0),thickness)
                
                # 連線雙腳
                R_anakle=keypoints.get("right_ankle")
                L_ankle=keypoints.get("left_ankle")
                cv2.line(frame, (int(L_ankle[0]),int(L_ankle[1])), (int(R_anakle[0]),int(R_anakle[1])), (76, 0, 153),thickness)
                
                # 判斷腳連線與重心線是否垂直 & balance
                if body_center and shoulder_center and L_ankle and R_anakle:

                    center_line_slope = calculate_slope(shoulder_center, body_center)
                    foot_line_slope = calculate_slope(L_ankle, R_anakle)             
                    
                    # 判斷是否平衡
                    if are_lines_perpendicular(center_line_slope, foot_line_slope):

                        cv2.rectangle(frame, (x1, y1), (x2, y2),  (0, 255,0), thickness)
                        cv2.putText(frame, "Balanced", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255,0), thickness)
                        state_counts["Balanced"] += 1
                    else:
                        balance_status = determine_balance(body_center, shoulder_center, L_ankle, R_anakle)
                        if balance_status == "Left":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 255, 255), thickness)
                            cv2.putText(frame, "Leaning Left", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (51, 255, 255), thickness)
                            state_counts["Leaning Left"] += 1
                        elif balance_status == "Right":
                            text_size = cv2.getTextSize("Leaning Right", cv2.FONT_HERSHEY_SIMPLEX, fontsize, thickness)[0]  # 返回 (width, height)
                            text_width, text_height = text_size
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 153, 255), thickness)
                            cv2.putText(frame, "Leaning Right", (x2 - text_width, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (51, 153, 255), thickness)
                            state_counts["Leaning Right"] += 1
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
                            cv2.putText(frame, "Unbalanced", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), thickness)
                            state_counts["Unbalanced"] += 1
        
    # 顯示處理後的畫面
    cv2.imshow("Pose and Action Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    out.write(frame)

# 釋放資源
cap.release()
cv2.destroyAllWindows()
