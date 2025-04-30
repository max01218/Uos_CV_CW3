import os
import numpy as np
from PIL import Image
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import glob
import csv
import time
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import faiss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

def extract_dense_sift(img, step=4, scales=[1.0, 0.75, 0.5, 0.25]):  # 增加尺度，減小步長
    """
    提取密集SIFT特徵，增強特徵提取
    """
    # 轉換為numpy數組並確保灰度圖
    img_array = np.array(img)
    
    # 應用CLAHE對比度增強
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # 初始化增強的SIFT檢測器
    sift = cv2.SIFT_create(
        nfeatures=0,        # 不限制特徵點數量
        nOctaveLayers=5,    # 增加octave層數
        contrastThreshold=0.04,  # 降低對比度閾值以檢測更多特徵
        edgeThreshold=10,   # 增加邊緣閾值
        sigma=1.6
    )
    
    all_descriptors = []
    
    # 在不同尺度下提取特徵
    for scale in scales:
        scaled_img = cv2.resize(img_array, None, fx=scale, fy=scale)
        
        # 創建密集網格關鍵點
        height, width = scaled_img.shape
        keypoints = []
        for y in range(step, height-step, step):
            for x in range(step, width-step, step):
                keypoints.append(cv2.KeyPoint(x, y, step*2))  # 增加特徵點尺寸
        
        # 計算SIFT描述符
        _, descriptors = sift.compute(scaled_img, keypoints)
        
        if descriptors is not None:
            # 對每個描述符進行L2歸一化
            descriptors = normalize(descriptors, norm='l2', axis=1)
            all_descriptors.append(descriptors)
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    
    return all_descriptors

def build_vocabulary(training_dir, num_clusters=1000, max_patches_per_image=200, pca_components=128):  # 增加聚類數和PCA維度
    """
    從訓練圖像中構建視覺詞彙，使用PCA降維和MiniBatchKMeans
    """
    print("構建視覺詞彙...")
    all_descriptors = []
    
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"處理類別: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for img_file in tqdm(image_files[:max_patches_per_image]):
            img = Image.open(img_file)
            descriptors = extract_dense_sift(img)
            if descriptors.size > 0:
                if len(descriptors) > 200:  # 增加每張圖片的描述符數量
                    indices = np.random.choice(len(descriptors), 200, replace=False)
                    descriptors = descriptors[indices]
                all_descriptors.append(descriptors)
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        
        # 標準化
        scaler = StandardScaler()
        all_descriptors = scaler.fit_transform(all_descriptors)
        
        # PCA降維
        print(f"使用PCA降維到{pca_components}維...")
        pca = PCA(n_components=pca_components, whiten=True)
        all_descriptors_pca = pca.fit_transform(all_descriptors)
        
        # MiniBatchKMeans聚類
        print(f"使用MiniBatchKMeans聚類{num_clusters}個視覺詞...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=2000,
            random_state=42,
            max_iter=300,  # 增加迭代次數
            n_init=3
        )
        kmeans.fit(all_descriptors_pca)
        
        return scaler, pca, kmeans
    else:
        print("錯誤：沒有找到SIFT描述符")
        return None, None, None

def extract_bow_features(img, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    使用FAISS進行快速特徵匹配，並加入空間金字塔特徵
    """
    descriptors = extract_dense_sift(img, step, scales)
    
    if descriptors.size == 0:
        return np.zeros(len(vocabulary))  # 只使用全局直方圖
    
    # 標準化和PCA降維
    descriptors = scaler.transform(descriptors)
    descriptors_pca = pca.transform(descriptors)
    
    # 使用FAISS進行快速最近鄰搜索
    index = faiss.IndexFlatL2(descriptors_pca.shape[1])
    index.add(vocabulary.astype(np.float32))
    _, labels = index.search(descriptors_pca.astype(np.float32), 1)
    
    # 計算全局直方圖
    hist = np.bincount(labels.flatten(), minlength=len(vocabulary))
    
    # 歸一化
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    加載訓練數據，返回特徵矩陣和標籤
    """
    print("加載訓練數據...")
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
        
        for img_file in tqdm(image_files):
            # 加載圖像
            img = Image.open(img_file)
            # 提取視覺詞袋特徵
            feature = extract_bow_features(img, scaler, pca, vocabulary, step, scales)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    加載測試數據，返回特徵矩陣和圖像文件名
    """
    print("加載測試數據...")
    features = []
    image_files = []
    
    # 獲取所有測試圖像並按數字順序排序
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_images = sorted(test_images, key=lambda x: float(os.path.basename(x).split('.')[0]))
    
    for img_file in tqdm(test_images):
        # 加載圖像
        img = Image.open(img_file)
        # 提取視覺詞袋特徵
        feature = extract_bow_features(img, scaler, pca, vocabulary, step, scales)
        features.append(feature)
        # 保存圖像文件名（不含路徑）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_svm_classifier(features, labels):
    """
    使用網格搜索優化SVM分類器，增加驗證和參數調整
    """
    print("訓練SVM分類器...")
    
    # 標準化特徵
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 打印每個類別的樣本數量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n類別分佈:")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count}樣本")
    
    # 定義更廣泛的參數網格
    param_grid = {
        'estimator__C': [0.001, 0.01, 0.1, 1.0],  # 減小C值範圍
        'estimator__class_weight': ['balanced'],
        'estimator__max_iter': [5000],  # 顯著增加最大迭代次數
        'estimator__tol': [1e-5],  # 降低容差
        'estimator__dual': [True],  # 使用對偶優化
        'estimator__loss': ['squared_hinge']  # 使用平方鉸鏈損失
    }
    
    # 創建基礎分類器
    base_classifier = LinearSVC(
        random_state=42,
        dual=True,  # 對於特徵數小於樣本數的情況，使用對偶形式
        loss='squared_hinge',  # 使用平方鉸鏈損失函數
        tol=1e-5  # 設置更嚴格的收斂容差
    )
    classifier = OneVsRestClassifier(base_classifier)
    
    # 使用網格搜索找到最佳參數
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='balanced_accuracy'
    )
    
    # 訓練模型
    print("\n開始網格搜索...")
    grid_search.fit(features_scaled, labels)
    print(f"\n最佳參數: {grid_search.best_params_}")
    print(f"最佳交叉驗證得分: {grid_search.best_score_:.3f}")
    
    # 使用最佳參數重新訓練模型
    print("\n使用最佳參數重新訓練模型...")
    best_classifier = grid_search.best_estimator_
    best_classifier.fit(features_scaled, labels)
    
    # 打印每個類別的性能
    predictions = best_classifier.predict(features_scaled)
    
    print("\n每個類別的性能:")
    for label in unique_labels:
        mask = labels == label
        correct = np.sum((predictions == labels) & mask)
        total = np.sum(mask)
        accuracy = correct / total
        print(f"{label}: {accuracy:.3f} ({correct}/{total})")
    
    return best_classifier, scaler

def predict_and_save(classifier, scaler, test_features, test_files, output_file):
    """
    對測試數據進行預測並保存結果，增加預測概率檢查
    """
    print("對測試數據進行預測...")
    
    # 標準化測試特徵
    test_features_scaled = scaler.transform(test_features)
    
    # 獲取預測概率（如果可用）和預測標籤
    if hasattr(classifier, 'decision_function'):
        decisions = classifier.decision_function(test_features_scaled)
        predictions = classifier.classes_[np.argmax(decisions, axis=1)]
        
        # 檢查決策值的分佈
        print("\n決策值統計:")
        print(f"最小值: {np.min(decisions):.3f}")
        print(f"最大值: {np.max(decisions):.3f}")
        print(f"平均值: {np.mean(decisions):.3f}")
        print(f"標準差: {np.std(decisions):.3f}")
        
        # 統計每個類別的預測數量
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print("\n預測分佈:")
        for pred, count in zip(unique_preds, counts):
            print(f"{pred}: {count}預測")
    else:
        predictions = classifier.predict(test_features_scaled)
    
    # 保存預測結果到TXT文件，確保按照文件名數字順序排序
    results = list(zip(test_files, predictions))
    results.sort(key=lambda x: float(x[0].split('.')[0]))
    
    with open(output_file, 'w') as f:
        for img_file, pred_class in results:
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"\n預測結果已保存到 {output_file}")

def main():
    # 設置參數
    step = 4
    scales = [1.0, 0.75, 0.5, 0.25]
    num_clusters = 1000
    pca_components = 128
    
    # 設置路徑
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run3.txt"
    model_file = "model.pkl"
    
    # 記錄開始時間
    start_time = time.time()
    
    # 強制重新訓練模型
    print("重新訓練模型...")
    
    # 構建視覺詞彙
    scaler, pca, kmeans = build_vocabulary(
        training_dir, 
        num_clusters=num_clusters, 
        max_patches_per_image=300,  # 增加每張圖片的特徵點數量
        pca_components=pca_components
    )
    
    # 加載訓練數據
    train_features, train_labels = load_training_data(training_dir, scaler, pca, kmeans.cluster_centers_, step, scales)
    
    # 檢查特徵
    print("\n特徵統計:")
    print(f"特徵維度: {train_features.shape}")
    print(f"特徵均值: {np.mean(train_features):.3f}")
    print(f"特徵標準差: {np.std(train_features):.3f}")
    print(f"零元素比例: {np.mean(train_features == 0):.3f}")
    print(f"非零元素比例: {np.mean(train_features != 0):.3f}")
    
    # 訓練分類器
    classifier, feat_scaler = train_svm_classifier(train_features, train_labels)
    
    # 保存模型
    print(f"\n保存模型: {model_file}")
    with open(model_file, 'wb') as f:
        pickle.dump((scaler, pca, kmeans, classifier, feat_scaler), f)
    
    # 加載測試數據
    test_features, test_files = load_test_data(test_dir, scaler, pca, kmeans.cluster_centers_, step, scales)
    
    # 檢查測試特徵
    print("\n測試特徵統計:")
    print(f"特徵維度: {test_features.shape}")
    print(f"特徵均值: {np.mean(test_features):.3f}")
    print(f"特徵標準差: {np.std(test_features):.3f}")
    print(f"零元素比例: {np.mean(test_features == 0):.3f}")
    print(f"非零元素比例: {np.mean(test_features != 0):.3f}")
    
    # 預測並保存結果
    predict_and_save(classifier, feat_scaler, test_features, test_files, output_file)
    
    # 計算並打印運行時間
    elapsed_time = time.time() - start_time
    print(f"\n總運行時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 