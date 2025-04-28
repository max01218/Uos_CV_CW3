import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import glob
import csv
import time

def create_tiny_image(img, size=16):
    """
    創建tiny image特徵
    1. 裁剪圖像中心的正方形區域
    2. 調整大小為16x16
    3. 將像素值打包成向量
    4. 使特徵向量具有零均值和單位長度
    """
    # 獲取圖像的寬度和高度
    width, height = img.size
    
    # 裁剪中心的正方形區域
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img_cropped = img.crop((left, top, right, bottom))
    
    # 調整大小為16x16
    img_resized = img_cropped.resize((size, size), Image.LANCZOS)
    
    # 將圖像轉換為numpy數組並展平
    img_array = np.array(img_resized).flatten()
    
    # 使特徵向量具有零均值和單位長度
    img_array = img_array - np.mean(img_array)
    norm = np.linalg.norm(img_array)
    if norm > 0:
        img_array = img_array / norm
    
    return img_array

def load_training_data(training_dir):
    """
    加載訓練數據，返回特徵矩陣和標籤
    """
    print("正在加載訓練數據...")
    features = []
    labels = []
    
    # 獲取所有類別目錄
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"處理類別: {class_name}")
        # 獲取該類別下的所有圖像
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for img_file in image_files:
            # 加載圖像
            img = Image.open(img_file)
            # 創建tiny image特徵
            feature = create_tiny_image(img)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir):
    """
    加載測試數據，返回特徵矩陣和圖像文件名
    """
    print("正在加載測試數據...")
    features = []
    image_files = []
    
    # 獲取所有測試圖像並按數字順序排序
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_images = sorted(test_images, key=lambda x: float(os.path.basename(x).split('.')[0]))  # 按數字大小排序
    
    for img_file in test_images:
        # 加載圖像
        img = Image.open(img_file)
        # 創建tiny image特徵
        feature = create_tiny_image(img)
        features.append(feature)
        # 保存圖像文件名（不含路徑）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_knn_classifier(features, labels, k=3):
    """
    訓練k近鄰分類器
    """
    print(f"訓練k近鄰分類器 (k={k})...")
    classifier = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    classifier.fit(features, labels)
    return classifier

def predict_and_save(classifier, test_features, test_files, output_file):
    """
    對測試數據進行預測並保存結果
    """
    print("對測試數據進行預測...")
    predictions = classifier.predict(test_features)
    
    # 保存預測結果到TXT文件，確保按照文件名數字順序排序
    results = list(zip(test_files, predictions))
    results.sort(key=lambda x: float(x[0].split('.')[0]))  # 按文件名中的數字排序
    
    with open(output_file, 'w') as f:
        for img_file, pred_class in results:
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"預測結果已保存到 {output_file}")

def main():
    # 設置路徑
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run1.txt"  # 修改輸出文件名
    
    # 記錄開始時間
    start_time = time.time()
    
    # 加載訓練數據
    train_features, train_labels = load_training_data(training_dir)
    
    # 加載測試數據
    test_features, test_files = load_test_data(test_dir)
    
    # 訓練分類器
    k = 3  # 可以根據需要調整k值
    classifier = train_knn_classifier(train_features, train_labels, k)
    
    # 預測並保存結果
    predict_and_save(classifier, test_features, test_files, output_file)
    
    # 計算並打印運行時間
    elapsed_time = time.time() - start_time
    print(f"總運行時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
