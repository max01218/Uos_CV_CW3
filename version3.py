import os
import numpy as np
from PIL import Image
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import glob
import csv
import time
from tqdm import tqdm
import pickle

def extract_dense_sift(img, step=8, scales=[1.0, 0.75, 0.5]):
    """
    提取密集SIFT特征
    """
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 初始化特征列表
    all_descriptors = []
    
    # 在不同尺度下提取特征
    for scale in scales:
        # 调整图像大小
        scaled_img = cv2.resize(img_array, None, fx=scale, fy=scale)
        
        # 创建关键点网格
        height, width = scaled_img.shape
        keypoints = []
        for y in range(step, height-step, step):
            for x in range(step, width-step, step):
                keypoints.append(cv2.KeyPoint(x, y, step))
        
        # 计算SIFT描述符
        _, descriptors = sift.compute(scaled_img, keypoints)
        
        # 如果找到描述符，添加到列表中
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    # 合并所有描述符
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        # 如果没有找到描述符，返回空数组
        all_descriptors = np.array([])
    
    return all_descriptors

def build_vocabulary(training_dir, num_clusters=500, max_patches_per_image=100):
    """
    从训练图像中构建视觉词汇
    """
    print("构建视觉词汇...")
    all_descriptors = []
    
    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"处理类别: {class_name}")
        # 获取该类别下的所有图像
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        for img_file in tqdm(image_files[:max_patches_per_image]):  # 限制每类图像数量以加快处理速度
            # 加载图像
            img = Image.open(img_file)
            # 提取密集SIFT特征
            descriptors = extract_dense_sift(img)
            # 添加到所有描述符
            if descriptors.size > 0:
                all_descriptors.append(descriptors)
    
    # 将所有描述符合并
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
        
        # 使用K-Means聚类
        print(f"使用K-Means聚类 {num_clusters} 个视觉词...")
        kmeans = cv2.KMeans_create(num_clusters, cv2.KMEANS_PP_CENTERS)
        kmeans.train(all_descriptors.astype(np.float32), cv2.TERM_CRITERIA_MAX_ITER, 100)
        
        # 保存词汇
        vocabulary = kmeans.get_centers()
        
        return vocabulary
    else:
        print("错误：没有找到SIFT描述符")
        return None

def extract_bow_features(img, vocabulary, step=8, scales=[1.0, 0.75, 0.5]):
    """
    从图像中提取视觉词袋特征
    """
    # 提取密集SIFT特征
    descriptors = extract_dense_sift(img, step, scales)
    
    if descriptors.size == 0:
        # 如果没有找到描述符，返回零向量
        return np.zeros(len(vocabulary))
    
    # 将每个描述符映射到最近的视觉词
    kmeans = cv2.KMeans_create()
    kmeans.set_centers(vocabulary.astype(np.float32))
    _, labels, _ = kmeans.predict(descriptors.astype(np.float32))
    
    # 计算视觉词袋特征（直方图）
    hist = np.zeros(len(vocabulary))
    for label in labels:
        hist[label] += 1
    
    # 归一化直方图
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, vocabulary, step=8, scales=[1.0, 0.75, 0.5]):
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
            feature = extract_bow_features(img, vocabulary, step, scales)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, vocabulary, step=8, scales=[1.0, 0.75, 0.5]):
    """
    加载测试数据，返回特征矩阵和图像文件名
    """
    print("加载测试数据...")
    features = []
    image_files = []
    
    # 获取所有测试图像
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    for img_file in tqdm(test_images):
        # 加载图像
        img = Image.open(img_file)
        # 提取视觉词袋特征
        feature = extract_bow_features(img, vocabulary, step, scales)
        features.append(feature)
        # 保存图像文件名（不含路径）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_svm_classifier(features, labels):
    """
    训练SVM分类器
    """
    print("训练SVM分类器...")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 训练SVM分类器
    classifier = SVC(kernel='rbf', probability=True, random_state=42)
    classifier.fit(features_scaled, labels)
    
    return classifier, scaler

def predict_and_save(classifier, scaler, test_features, test_files, output_file):
    """
    对测试数据进行预测并保存结果
    """
    print("对测试数据进行预测...")
    
    # 标准化测试特征
    test_features_scaled = scaler.transform(test_features)
    
    # 预测
    predictions = classifier.predict(test_features_scaled)
    
    # 保存预测结果到TXT文件
    with open(output_file, 'w') as f:
        for img_file, pred_class in zip(test_files, predictions):
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"预测结果已保存到 {output_file}")

def main():
    # 设置参数
    step = 8
    scales = [1.0, 0.75, 0.5]
    num_clusters = 500
    
    # 设置路径
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run3.txt"  # 修改输出文件名
    vocabulary_file = "vocabulary.pkl"
    
    # 记录开始时间
    start_time = time.time()
    
    # 构建或加载视觉词汇
    if os.path.exists(vocabulary_file):
        print(f"加载视觉词汇: {vocabulary_file}")
        with open(vocabulary_file, 'rb') as f:
            vocabulary = pickle.load(f)
    else:
        # 构建视觉词汇
        vocabulary = build_vocabulary(training_dir, num_clusters)
        # 保存词汇
        print(f"保存视觉词汇: {vocabulary_file}")
        with open(vocabulary_file, 'wb') as f:
            pickle.dump(vocabulary, f)
    
    # 加载训练数据
    train_features, train_labels = load_training_data(training_dir, vocabulary, step, scales)
    
    # 加载测试数据
    test_features, test_files = load_test_data(test_dir, vocabulary, step, scales)
    
    # 训练分类器
    classifier, scaler = train_svm_classifier(train_features, train_labels)
    
    # 预测并保存结果
    predict_and_save(classifier, scaler, test_features, test_files, output_file)
    
    # 计算并打印运行时间
    elapsed_time = time.time() - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 