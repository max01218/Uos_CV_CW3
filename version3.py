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

def extract_dense_sift(img, step=4, scales=[1.0, 0.75, 0.5, 0.25]):  # 增加尺度，减小步长
    """
    提取密集SIFT特征，增强特征提取
    """
    # 转换为numpy数组并确保灰度图
    if len(np.array(img).shape) == 3:
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    else:
        img_array = np.array(img)
    
    # 应用CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # 初始化增强的SIFT检测器
    sift = cv2.SIFT_create(
        nfeatures=0,        # 不限制特征点数量
        nOctaveLayers=5,    # 增加octave层数
        contrastThreshold=0.04,  # 降低对比度阈值以检测更多特征
        edgeThreshold=10,   # 增加边缘阈值
        sigma=1.6
    )
    
    all_descriptors = []
    
    # 在不同尺度下提取特征
    for scale in scales:
        scaled_img = cv2.resize(img_array, None, fx=scale, fy=scale)
        
        # 创建密集网格关键点
        height, width = scaled_img.shape
        keypoints = []
        for y in range(step, height-step, step):
            for x in range(step, width-step, step):
                keypoints.append(cv2.KeyPoint(x, y, step*2))  # 增加特征点尺寸
        
        # 计算SIFT描述符
        _, descriptors = sift.compute(scaled_img, keypoints)
        
        if descriptors is not None:
            # 对每个描述符进行L2归一化
            descriptors = normalize(descriptors, norm='l2', axis=1)
            all_descriptors.append(descriptors)
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    
    return all_descriptors

def build_vocabulary(training_dir, num_clusters=1000, max_patches_per_image=200, pca_components=128):  # 增加聚类数和PCA维度
    """
    从训练图像中构建视觉词汇，使用PCA降维和MiniBatchKMeans
    """
    print("构建视觉词汇...")
    all_descriptors = []
    
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"处理类别: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for img_file in tqdm(image_files[:max_patches_per_image]):
            img = Image.open(img_file)
            descriptors = extract_dense_sift(img)
            if descriptors.size > 0:
                if len(descriptors) > 200:  # 增加每张图片的描述符数量
                    indices = np.random.choice(len(descriptors), 200, replace=False)
                    descriptors = descriptors[indices]
                all_descriptors.append(descriptors)
    
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        
        # 标准化
        scaler = StandardScaler()
        all_descriptors = scaler.fit_transform(all_descriptors)
        
        # PCA降维
        print(f"使用PCA降维到{pca_components}维...")
        pca = PCA(n_components=pca_components, whiten=True)
        all_descriptors_pca = pca.fit_transform(all_descriptors)
        
        # MiniBatchKMeans聚类
        print(f"使用MiniBatchKMeans聚类{num_clusters}个视觉词...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=2000,
            random_state=42,
            max_iter=300,  # 增加迭代次数
            n_init=3
        )
        kmeans.fit(all_descriptors_pca)
        
        return scaler, pca, kmeans
    else:
        print("错误：没有找到SIFT描述符")
        return None, None, None

def extract_bow_features(img, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    使用FAISS进行快速特征匹配，并加入空间金字塔特征
    """
    descriptors = extract_dense_sift(img, step, scales)
    
    if descriptors.size == 0:
        return np.zeros(len(vocabulary))  # 只使用全局直方图
    
    # 标准化和PCA降维
    descriptors = scaler.transform(descriptors)
    descriptors_pca = pca.transform(descriptors)
    
    # 使用FAISS进行快速最近邻搜索
    index = faiss.IndexFlatL2(descriptors_pca.shape[1])
    index.add(vocabulary.astype(np.float32))
    _, labels = index.search(descriptors_pca.astype(np.float32), 1)
    
    # 计算全局直方图
    hist = np.bincount(labels.flatten(), minlength=len(vocabulary))
    
    # 归一化
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    加载训练数据，返回特征矩阵和标签
    """
    print("加载训练数据...")
    features = []
    labels = []
    
    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"处理类别: {class_name}")
        # 获取该类别下的所有图像
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for img_file in tqdm(image_files):
            # 加载图像
            img = Image.open(img_file)
            # 提取视觉词袋特征
            feature = extract_bow_features(img, scaler, pca, vocabulary, step, scales)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, scaler, pca, vocabulary, step=4, scales=[1.0, 0.75, 0.5, 0.25]):
    """
    加载测试数据，返回特征矩阵和图像文件名
    """
    print("加载测试数据...")
    features = []
    image_files = []
    
    # 获取所有测试图像并按数字顺序排序
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_images = sorted(test_images, key=lambda x: float(os.path.basename(x).split('.')[0]))
    
    for img_file in tqdm(test_images):
        # 加载图像
        img = Image.open(img_file)
        # 提取视觉词袋特征
        feature = extract_bow_features(img, scaler, pca, vocabulary, step, scales)
        features.append(feature)
        # 保存图像文件名（不含路径）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_svm_classifier(features, labels):
    """
    使用网格搜索优化SVM分类器，增加验证和参数调整
    """
    print("训练SVM分类器...")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 打印每个类别的样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count}样本")
    
    # 定义更广泛的参数网格
    param_grid = {
        'estimator__C': [0.001, 0.01, 0.1, 1.0],  # 减小C值范围
        'estimator__class_weight': ['balanced'],
        'estimator__max_iter': [5000],  # 显著增加最大迭代次数
        'estimator__tol': [1e-5],  # 降低容差
        'estimator__dual': [True],  # 使用对偶优化
        'estimator__loss': ['squared_hinge']  # 使用平方铰链损失
    }
    
    # 创建基础分类器
    base_classifier = LinearSVC(
        random_state=42,
        dual=True,  # 对于特征数小于样本数的情况，使用对偶形式
        loss='squared_hinge',  # 使用平方铰链损失函数
        tol=1e-5  # 设置更严格的收敛容差
    )
    classifier = OneVsRestClassifier(base_classifier)
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='balanced_accuracy'
    )
    
    # 训练模型
    print("\n开始网格搜索...")
    grid_search.fit(features_scaled, labels)
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.3f}")
    
    # 使用最佳参数重新训练模型
    print("\n使用最佳参数重新训练模型...")
    best_classifier = grid_search.best_estimator_
    best_classifier.fit(features_scaled, labels)
    
    # 打印每个类别的性能
    predictions = best_classifier.predict(features_scaled)
    
    print("\n每个类别的性能:")
    for label in unique_labels:
        mask = labels == label
        correct = np.sum((predictions == labels) & mask)
        total = np.sum(mask)
        accuracy = correct / total
        print(f"{label}: {accuracy:.3f} ({correct}/{total})")
    
    return best_classifier, scaler

def predict_and_save(classifier, scaler, test_features, test_files, output_file):
    """
    对测试数据进行预测并保存结果，增加预测概率检查
    """
    print("对测试数据进行预测...")
    
    # 标准化测试特征
    test_features_scaled = scaler.transform(test_features)
    
    # 获取预测概率（如果可用）和预测标签
    if hasattr(classifier, 'decision_function'):
        decisions = classifier.decision_function(test_features_scaled)
        predictions = classifier.classes_[np.argmax(decisions, axis=1)]
        
        # 检查决策值的分布
        print("\n决策值统计:")
        print(f"最小值: {np.min(decisions):.3f}")
        print(f"最大值: {np.max(decisions):.3f}")
        print(f"平均值: {np.mean(decisions):.3f}")
        print(f"标准差: {np.std(decisions):.3f}")
        
        # 统计每个类别的预测数量
        unique_preds, counts = np.unique(predictions, return_counts=True)
        print("\n预测分布:")
        for pred, count in zip(unique_preds, counts):
            print(f"{pred}: {count}预测")
    else:
        predictions = classifier.predict(test_features_scaled)
    
    # 保存预测结果到TXT文件，确保按照文件名数字顺序排序
    results = list(zip(test_files, predictions))
    results.sort(key=lambda x: float(x[0].split('.')[0]))
    
    with open(output_file, 'w') as f:
        for img_file, pred_class in results:
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"\n预测结果已保存到 {output_file}")

def main():
    # 设置参数
    step = 4
    scales = [1.0, 0.75, 0.5, 0.25]
    num_clusters = 1000
    pca_components = 128
    
    # 设置路径
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run3.txt"
    model_file = "model.pkl"
    
    # 记录开始时间
    start_time = time.time()
    
    # 强制重新训练模型
    print("重新训练模型...")
    
    # 构建视觉词汇
    scaler, pca, kmeans = build_vocabulary(
        training_dir, 
        num_clusters=num_clusters, 
        max_patches_per_image=300,  # 增加每张图片的特征点数量
        pca_components=pca_components
    )
    
    # 加载训练数据
    train_features, train_labels = load_training_data(training_dir, scaler, pca, kmeans.cluster_centers_, step, scales)
    
    # 检查特征
    print("\n特征统计:")
    print(f"特征维度: {train_features.shape}")
    print(f"特征均值: {np.mean(train_features):.3f}")
    print(f"特征标准差: {np.std(train_features):.3f}")
    print(f"零元素比例: {np.mean(train_features == 0):.3f}")
    print(f"非零元素比例: {np.mean(train_features != 0):.3f}")
    
    # 训练分类器
    classifier, feat_scaler = train_svm_classifier(train_features, train_labels)
    
    # 保存模型
    print(f"\n保存模型: {model_file}")
    with open(model_file, 'wb') as f:
        pickle.dump((scaler, pca, kmeans, classifier, feat_scaler), f)
    
    # 加载测试数据
    test_features, test_files = load_test_data(test_dir, scaler, pca, kmeans.cluster_centers_, step, scales)
    
    # 检查测试特征
    print("\n测试特征统计:")
    print(f"特征维度: {test_features.shape}")
    print(f"特征均值: {np.mean(test_features):.3f}")
    print(f"特征标准差: {np.std(test_features):.3f}")
    print(f"零元素比例: {np.mean(test_features == 0):.3f}")
    print(f"非零元素比例: {np.mean(test_features != 0):.3f}")
    
    # 预测并保存结果
    predict_and_save(classifier, feat_scaler, test_features, test_files, output_file)
    
    # 计算并打印运行时间
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 