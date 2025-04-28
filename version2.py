import os
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob
import csv
import time
from tqdm import tqdm
from joblib import dump, load
import multiprocessing
from joblib import Parallel, delayed

def extract_patches(img, patch_size=8, stride=4):
    """
    從圖像中密集採樣像素塊
    """
    # 轉換為numpy數組
    img_array = np.array(img)
    
    # 獲取圖像尺寸
    height, width = img_array.shape
    
    # 計算可以提取的塊的數量
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1
    
    # 初始化patches數組
    patches = np.zeros((num_patches_h * num_patches_w, patch_size * patch_size))
    
    # 提取patches
    idx = 0
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = img_array[i:i+patch_size, j:j+patch_size]
            patches[idx] = patch.flatten()
            idx += 1
    
    return patches

def normalize_patches(patches):
    """
    對每個patch進行均值中心化和歸一化
    """
    # 均值中心化
    patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
    
    # 歸一化
    norms = np.linalg.norm(patches_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除以零
    patches_normalized = patches_centered / norms
    
    return patches_normalized

def process_single_image(args):
    """
    處理單張圖像的輔助函數，用於並行處理
    """
    img_file, kmeans, pca, patch_size, stride = args
    img = Image.open(img_file)
    patches = extract_patches(img, patch_size, stride)
    patches_normalized = normalize_patches(patches)
    patches_pca = pca.transform(patches_normalized)
    
    # 使用批量預測代替單個預測
    labels = kmeans.predict(patches_pca)
    
    # 計算直方圖
    hist = np.bincount(labels, minlength=kmeans.n_clusters)
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, kmeans, pca, patch_size=8, stride=4):
    """
    並行加載訓練數據
    """
    print("加載訓練數據...")
    features = []
    labels = []
    
    # 獲取所有類別目錄
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    n_jobs = multiprocessing.cpu_count()
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"處理類別: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        # 準備並行處理的參數
        process_args = [(f, kmeans, pca, patch_size, stride) for f in image_files]
        
        # 並行處理該類別的所有圖像
        class_features = Parallel(n_jobs=n_jobs)(
            delayed(process_single_image)(args) for args in tqdm(process_args)
        )
        
        features.extend(class_features)
        labels.extend([class_name] * len(image_files))
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, kmeans, pca, patch_size=8, stride=4):
    """
    並行加載測試數據
    """
    print("加載測試數據...")
    
    # 獲取所有測試圖像並按數字順序排序
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_images = sorted(test_images, key=lambda x: float(os.path.basename(x).split('.')[0]))
    
    # 準備並行處理的參數
    process_args = [(f, kmeans, pca, patch_size, stride) for f in test_images]
    
    # 並行處理所有測試圖像
    n_jobs = multiprocessing.cpu_count()
    features = Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(args) for args in tqdm(process_args)
    )
    
    # 獲取文件名列表
    image_files = [os.path.basename(f) for f in test_images]
    
    return np.array(features), image_files

def build_vocabulary(training_dir, num_clusters=500, patch_size=8, stride=4, max_patches_per_image=100, pca_components=64, batch_size=1000):
    """
    從訓練圖像中構建視覺詞彙，使用並行處理加速
    """
    print("構建視覺詞彙...")
    all_patches = []
    
    # 獲取所有類別目錄
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    def process_image(img_file):
        img = Image.open(img_file)
        patches = extract_patches(img, patch_size, stride)
        return normalize_patches(patches)
    
    n_jobs = multiprocessing.cpu_count()
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"處理類別: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))[:max_patches_per_image]
        
        # 並行處理圖像
        class_patches = Parallel(n_jobs=n_jobs)(
            delayed(process_image)(f) for f in tqdm(image_files)
        )
        
        all_patches.extend(class_patches)
    
    # 將所有patches合併
    all_patches = np.vstack(all_patches)
    
    # 使用PCA降維
    print(f"使用PCA降維到 {pca_components} 維...")
    pca = PCA(n_components=pca_components, whiten=True)
    all_patches_pca = pca.fit_transform(all_patches)
    
    # 使用MiniBatchKMeans聚類
    print(f"使用MiniBatchKMeans聚類 {num_clusters} 個視覺詞...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        random_state=42,
        max_iter=300,
        n_init=3
    )
    kmeans.fit(all_patches_pca)
    
    return kmeans, pca

def train_linear_classifiers(features, labels):
    """
    訓練一組線性分類器（一對多分類器）
    """
    print("訓練線性分類器...")
    
    # 獲取唯一類別
    unique_classes = np.unique(labels)
    
    # 初始化分類器字典
    classifiers = {}
    
    # 對每個類別訓練一個一對多分類器
    for class_name in unique_classes:
        print(f"訓練類別: {class_name}")
        # 創建二分類標籤
        binary_labels = (labels == class_name).astype(int)
        
        # 訓練邏輯回歸分類器
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(features, binary_labels)
        
        # 保存分類器
        classifiers[class_name] = classifier
    
    return classifiers

def predict_and_save(classifiers, test_features, test_files, output_file):
    """
    對測試數據進行預測並保存結果
    """
    print("對測試數據進行預測...")
    
    # 獲取所有類別
    classes = list(classifiers.keys())
    
    # 初始化預測結果
    predictions = []
    
    # 對每個測試樣本進行預測
    for i, feature in enumerate(tqdm(test_features)):
        # 計算每個類別的得分
        scores = {}
        for class_name, classifier in classifiers.items():
            # 獲取正類的概率
            score = classifier.predict_proba(feature.reshape(1, -1))[0, 1]
            scores[class_name] = score
        
        # 選擇得分最高的類別
        best_class = max(scores, key=scores.get)
        predictions.append(best_class)
    
    # 保存預測結果到TXT文件，確保按照文件名數字順序排序
    results = list(zip(test_files, predictions))
    results.sort(key=lambda x: float(x[0].split('.')[0]))  # 按文件名中的數字排序
    
    with open(output_file, 'w') as f:
        for img_file, pred_class in results:
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"預測結果已保存到 {output_file}")

def main():
    # 設置參數
    patch_size = 8
    stride = 4
    num_clusters = 500
    pca_components = 64
    batch_size = 1000  # MiniBatchKMeans的批次大小
    
    # 設置路徑
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run2.txt"
    model_file = "models.joblib"  # 模型保存文件
    
    # 記錄開始時間
    start_time = time.time()
    
    # 檢查是否存在保存的模型
    if os.path.exists(model_file):
        print("加載已保存的模型...")
        models = load(model_file)
        kmeans, pca, classifiers = models['kmeans'], models['pca'], models['classifiers']
    else:
        print("訓練新模型...")
        # 構建視覺詞彙
        kmeans, pca = build_vocabulary(training_dir, num_clusters, patch_size, stride, 
                                     pca_components=pca_components, batch_size=batch_size)
        
        # 加載訓練數據
        train_features, train_labels = load_training_data(training_dir, kmeans, pca, patch_size, stride)
        
        # 訓練分類器
        classifiers = train_linear_classifiers(train_features, train_labels)
        
        # 保存模型
        print("保存模型...")
        models = {
            'kmeans': kmeans,
            'pca': pca,
            'classifiers': classifiers
        }
        dump(models, model_file)
    
    # 加載測試數據
    test_features, test_files = load_test_data(test_dir, kmeans, pca, patch_size, stride)
    
    # 預測並保存結果
    predict_and_save(classifiers, test_features, test_files, output_file)
    
    # 計算並打印運行時間
    elapsed_time = time.time() - start_time
    print(f"總運行時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 