import time
from ultralytics import YOLO
import cv2
import numpy as np

# 加載模型
model = YOLO('yolov8n-pose.pt')  # 載入 YOLOv8 姿勢辨識模型

state_counts = {
    "Balanced": 0,
    "Leaning Left": 0,
    "Leaning Right": 0,
    "Unbalanced": 0
}

# 計算人體高度（基於 YOLO 檢測框）
def get_person_height(bbox):
    _, y1, _, y2 = bbox  # 取得邊界框上下兩端的 Y 座標
    return abs(y2 - y1)  # 高度為上下差值

# 繪製檢測框
def draw_bounding_box(img, bbox, label="Person", color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)  # 將浮點數座標轉換為整數
    # 獲取圖像解析度
    height, width, _ = img.shape  

    # 動態計算圓的半徑和邊框粗細
    radius = int(min(width, height) * 0.005555)  # 取最小邊長的 0.5% 作為半徑
    thickness = max(1, radius // 2) 
            
    # 繪製矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # 在框上方繪製類別名稱
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

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

    # 初始化 intersection
    intersection = None

    # 如果腳的連線可以形成有效的線段
    if foot_slope is not None and foot_intercept is not None:
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

# 繪製關鍵點與連線
def draw_keypoints(img, keypoints, radius, thickness):
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    # 繪製關鍵點
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), -1)  # 白點

    # 繪製連線
    for connection in connections:
        try:
            start = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            end = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(img, start, end, (255, 0, 255), thickness)  # 紅線
        except IndexError:
            print(f"Connection {connection} is out of range for keypoints list.")

    return img

# 處理影片
def process_video(video_path,output_path):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)  # 設定視窗

    # 取得影片資訊
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 原影片的每秒幀數

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式的編碼器
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
            bbox = result.boxes.xyxy[0].cpu().numpy()  # 獲取檢測框座標 [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # 獲取圖像解析度
            height, width, _ = frame.shape  

            # 動態計算圓的半徑和邊框粗細
            radius = int(min(width, height) * 0.005555)  # 取最小邊長的 0.5% 作為半徑
            thickness = max(1, radius // 2)
            fontsize = min(width, height) / 850
            
            # 計算人體高度（基於檢測框）
            body_height = get_person_height(bbox)
            # cv2.putText(frame, f"body_height: {body_height}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
            
            # 提取特定關鍵點
            keypoints["nose"] = tuple(keypoint_coords[0]) if len(keypoint_coords) > 0 else (None, None)
            keypoints["left_shoulder"] = tuple(keypoint_coords[5]) if len(keypoint_coords) > 5 else (None, None)
            keypoints["right_shoulder"] = tuple(keypoint_coords[6]) if len(keypoint_coords) > 6 else (None, None)
            keypoints["left_ankle"] = tuple(keypoint_coords[16]) if len(keypoint_coords) > 16 else (None, None)
            keypoints["right_ankle"] = tuple(keypoint_coords[15]) if len(keypoint_coords) > 15 else (None, None)

            # print(keypoint_coords)
            
            # 計算人體重心
            body_center = calculate_body_center(keypoint_coords)
            
            # 找出肩膀中心點
            shoulder_center = find_shoulder_center(keypoints)
                        
            # 繪製檢測框
            # draw_bounding_box(frame, bbox, label="Person")
            
            # 繪製關鍵點
            frame = draw_keypoints(frame, keypoint_coords,radius,thickness)
            
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
            
            # 判斷腳連線與重心線是否垂直
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
        
        # 顯示FPS
        et = time.time()
        FPS = round(1 / (et - st), 1)
        cv2.putText(frame, f"FPS: {FPS}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2*fontsize, (0, 255, 255), thickness, cv2.LINE_AA)
        
        y_offset = height - 200
        for state, count in state_counts.items():
            cv2.putText(frame, f"{state}: {count}", (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 2*fontsize, (255, 255, 0), thickness, cv2.LINE_AA)
            y_offset +=30
            
        # 顯示影像
        cv2.imshow('Pose Detection', frame)
        
        # ESC鍵退出
        if cv2.waitKey(1) & 0xFF == 27:
            for state, count in state_counts.items():
                print(f"{state}: {count}")
            break
        
        out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()

# 主程式
if __name__ == "__main__":
    # video_path = "data/videos/sample_video.mp4"  # 測試影片路徑
    video_path = "data/videos/1 (4).mp4"  # 測試影片路徑
    output_path = "output.mp4"
    process_video(video_path,output_path)
