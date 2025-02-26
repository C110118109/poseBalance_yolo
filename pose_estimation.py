from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 加載模型
model = YOLO('yolov8n-pose.pt')  # 載入預訓練模型

state_counts = {
    "Balanced": 0,
    "Leaning Left": 0,
    "Leaning Right": 0,
    "Unbalanced": 0
}

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
        
        # 獲取圖像解析度
        height, width  , _ = img.shape  

        # 動態計算圓的半徑和邊框粗細
        radius = int(min(width, height) * 0.008)  # 取最小邊長的 0.5% 作為半徑
        thickness = max(1, radius // 2)
        fontsize = min(width, height) / 950
    
        
        
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
        img_with_keypoints = draw_keypoints(img_rgb, keypoint_coords,radius,thickness)
        
        # print(f"total:{keypoints}")
        
        # 計算重心
        body_center=calculate_body_center(keypoints)
        
        # 找出肩膀中心點
        shoulder_center = find_shoulder_center(keypoints)
        
        # 畫重心
        if body_center:
            cv2.circle(img_rgb, (int(body_center[0]), int(body_center[1])), radius, (255, 0, 0), -1)
        print(f"body_center: {body_center}")
        
        # 繪製肩膀中心點
        if body_center:
            cv2.circle(img_rgb, (int(shoulder_center[0]), int(shoulder_center[1])), radius, (255, 0, 0), -1)
        
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
            cv2.line(img_rgb, start_point, end_point, (255, 0, 0),thickness)
            
        # 連線雙腳
        R_anakle=keypoints.get("right_ankle")
        L_ankle=keypoints.get("left_ankle")
        cv2.line(img_rgb, (int(L_ankle[0]),int(L_ankle[1])), (int(R_anakle[0]),int(R_anakle[1])), (76, 0, 153),thickness)
        

        # 判斷腳連線與重心線是否垂直
        if body_center and shoulder_center and L_ankle and R_anakle:

            center_line_slope = calculate_slope(shoulder_center, body_center)
            foot_line_slope = calculate_slope(L_ankle, R_anakle)             
            
            print(f"center_line_slope: {center_line_slope}")
            print(f"foot_line_slope: {foot_line_slope}")
            
            # 判斷是否平衡
            if are_lines_perpendicular(center_line_slope, foot_line_slope):

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2),  (0, 255,0), thickness)
                cv2.putText(img_rgb, "Balanced", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255,0), thickness)
                state_counts["Balanced"] += 1
                print(f"{"Balanced"}")
            else:
                print(f"{determine_balance(body_center, shoulder_center, L_ankle, R_anakle)}")
                balance_status = determine_balance(body_center, shoulder_center, L_ankle, R_anakle)
                
                if balance_status == "Left":
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (51, 255, 255), thickness)
                    cv2.putText(img_rgb, "Leaning Left", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (51, 255, 255), thickness)
                    state_counts["Leaning Left"] += 1
                elif balance_status == "Right":
                    text_size = cv2.getTextSize("Leaning Right", cv2.FONT_HERSHEY_SIMPLEX, fontsize, thickness)[0]  # 返回 (width, height)
                    text_width, text_height = text_size

                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (51, 153, 255), thickness)
                    cv2.putText(img_rgb, "Leaning Right", (x2 - text_width, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (51, 153, 255), thickness)
                    state_counts["Leaning Right"] += 1
                else:
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), thickness)
                    cv2.putText(img_rgb, "Unbalanced", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), thickness)
                    state_counts["Unbalanced"] += 1

        # # 計算角度
        # angle = calculate_angle(center_line_slope, foot_line_slope)
        
        # # 顯示角度
        # display_angle_on_image(img_rgb, angle,fontsize,thickness, position=(x2 + 20, y1 + 20))  # 在圖片旁顯示角度
        
    
    # 顯示圖片
    plt.imshow(img_with_keypoints)
    plt.axis('off')
    plt.show()



# 繪製關鍵點與連線
def draw_keypoints(img, keypoints,radius,thickness):
    connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ]

    # 繪製關鍵點
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)  # 綠點

    # 繪製連線
    for connection in connections:
        try:
            start = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            end = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(img, start, end, (0, 0, 255), thickness)  # 紅線
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
    
    print(f"num: {num}")
    
    if num >= -1.5 and num <= -0.5:
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

# # 計算兩條線之間的角度
# def calculate_angle(slope1, slope2):
#     if slope1 is None or slope2 is None:
#         return None  # 如果其中一條線是垂直線，無法計算角度
    
#     # 兩條直線的斜率
#     angle_radians = abs(math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2))))
#     angle_degrees = math.degrees(angle_radians)  # 轉換為度數
#     return round(angle_degrees, 2)

# # 在圖像上顯示角度
# def display_angle_on_image(img, angle, fontsize, thickness, position=(50, 50)):
#     if angle is not None:
#         cv2.putText(img, f"Angle: {angle}", position, cv2.FONT_HERSHEY_SIMPLEX, fontsize+0.3, (0, 255, 255), thickness)
#         print(f"Angle: {angle}")


# 主程式
if __name__ == "__main__":
    # image_path = "data/images/sample_image.jpg"  # 測試圖片路徑
    image_path = "data/images/sample07.jpg"  # 測試圖片路徑
    # image_path = "data/images/court1.jpg"  # 測試圖片路徑
    detect_pose(image_path)
