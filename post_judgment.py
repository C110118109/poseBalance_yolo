from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# **載入你訓練好的 YOLOv8 模型**
model = YOLO('best.pt')  # 載入訓練好的模型

# **推論函數**
def detect_pose(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換顏色

    results = model(img_rgb)  # 進行推論

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # 偵測框
            class_id = int(box.cls[0].cpu().numpy())  # 類別 ID
            conf = box.conf[0].cpu().numpy()  # 置信度
            
            # 計算框的寬高
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 如果框的大小接近圖片大小，縮小框
            img_height, img_width, _ = img_rgb.shape
            if box_width > 0.9 * img_width or box_height > 0.9 * img_height:
                margin_x = int(0.05 * img_width)  # 左右各縮5%
                margin_y = int(0.05 * img_height)  # 上下各縮5%
                x1 = max(x1 + margin_x, 0)
                y1 = max(y1 + margin_y, 0)
                x2 = min(x2 - margin_x, img_width - 1)
                y2 = min(y2 - margin_y, img_height - 1)

            
            # 獲取圖像解析度
            height, width, _ = img_rgb.shape 
            
            # 動態計算圓的半徑和邊框粗細
            radius = int(min(width, height) * 0.005555)  # 取最小邊長的 0.5% 作為半徑
            thickness = max(1, radius // 2)
            fontsize = min(width, height) / 850
            
            label = model.names[class_id]  # 取得姿勢名稱
            print(f"label: {label}")
        # 在畫面上繪製偵測框和標籤
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), thickness)  # 綠色邊框
        cv2.putText(
            img_rgb,
            f"{label} ({conf:.2f})",  # 標籤名稱與置信度
            (x1, max(y1-5 , 0)),  # 確保文字顯示在框上方且不超出圖片範圍
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize,  # 字體大小，可以調大一些
            (0, 255, 0),  # 綠色文字
            thickness,  # 文字粗細
            cv2.LINE_AA  # 抗鋸齒
        )


    # 顯示結果
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# **測試圖片**
if __name__ == "__main__":
    image_path = "data/images/sample.jpg"  # 測試圖片路徑
    detect_pose(image_path)
