import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
import csv
import time
from tqdm import tqdm

def extract_patches(img, patch_size=8, stride=4):
    """
    从图像中密集采样像素块
    """
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 获取图像尺寸
    height, width = img_array.shape
    
    # 计算可以提取的块的数量
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1
    
    # 初始化patches数组
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
    对每个patch进行均值中心化和归一化
    """
    # 均值中心化
    patches_centered = patches - np.mean(patches, axis=1, keepdims=True)
    
    # 归一化
    norms = np.linalg.norm(patches_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除以零
    patches_normalized = patches_centered / norms
    
    return patches_normalized

def build_vocabulary(training_dir, num_clusters=500, patch_size=8, stride=4, max_patches_per_image=100):
    """
    从训练图像中构建视觉词汇
    """
    print("构建视觉词汇...")
    all_patches = []
    
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
            # 提取patches
            patches = extract_patches(img, patch_size, stride)
            # 归一化patches
            patches_normalized = normalize_patches(patches)
            # 添加到所有patches
            all_patches.append(patches_normalized)
    
    # 将所有patches合并
    all_patches = np.vstack(all_patches)
    
    # 使用K-Means聚类
    print(f"使用K-Means聚类 {num_clusters} 个视觉词...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_patches)
    
    return kmeans

def extract_bow_features(img, kmeans, patch_size=8, stride=4):
    """
    从图像中提取视觉词袋特征
    """
    # 提取patches
    patches = extract_patches(img, patch_size, stride)
    
    # 归一化patches
    patches_normalized = normalize_patches(patches)
    
    # 将每个patch映射到最近的视觉词
    labels = kmeans.predict(patches_normalized)
    
    # 计算视觉词袋特征（直方图）
    hist = np.zeros(kmeans.n_clusters)
    for label in labels:
        hist[label] += 1
    
    # 归一化直方图
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, kmeans, patch_size=8, stride=4):
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
            feature = extract_bow_features(img, kmeans, patch_size, stride)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, kmeans, patch_size=8, stride=4):
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
        feature = extract_bow_features(img, kmeans, patch_size, stride)
        features.append(feature)
        # 保存图像文件名（不含路径）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_linear_classifiers(features, labels):
    """
    训练一组线性分类器（一对多分类器）
    """
    print("训练线性分类器...")
    
    # 获取唯一类别
    unique_classes = np.unique(labels)
    
    # 初始化分类器字典
    classifiers = {}
    
    # 对每个类别训练一个一对多分类器
    for class_name in unique_classes:
        print(f"训练类别: {class_name}")
        # 创建二分类标签
        binary_labels = (labels == class_name).astype(int)
        
        # 训练逻辑回归分类器
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(features, binary_labels)
        
        # 保存分类器
        classifiers[class_name] = classifier
    
    return classifiers

def predict_and_save(classifiers, test_features, test_files, output_file):
    """
    对测试数据进行预测并保存结果
    """
    print("对测试数据进行预测...")
    
    # 获取所有类别
    classes = list(classifiers.keys())
    
    # 初始化预测结果
    predictions = []
    
    # 对每个测试样本进行预测
    for i, feature in enumerate(tqdm(test_features)):
        # 计算每个类别的得分
        scores = {}
        for class_name, classifier in classifiers.items():
            # 获取正类的概率
            score = classifier.predict_proba(feature.reshape(1, -1))[0, 1]
            scores[class_name] = score
        
        # 选择得分最高的类别
        best_class = max(scores, key=scores.get)
        predictions.append(best_class)
    
    # 保存预测结果到TXT文件
    with open(output_file, 'w') as f:
        for img_file, pred_class in zip(test_files, predictions):
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"预测结果已保存到 {output_file}")

def main():
    # 设置参数
    patch_size = 8
    stride = 4
    num_clusters = 500
    
    # 设置路径
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run2.txt"  # 修改输出文件名
    
    # 记录开始时间
    start_time = time.time()
    
    # 构建视觉词汇
    kmeans = build_vocabulary(training_dir, num_clusters, patch_size, stride)
    
    # 加载训练数据
    train_features, train_labels = load_training_data(training_dir, kmeans, patch_size, stride)
    
    # 加载测试数据
    test_features, test_files = load_test_data(test_dir, kmeans, patch_size, stride)
    
    # 训练分类器
    classifiers = train_linear_classifiers(train_features, train_labels)
    
    # 预测并保存结果
    predict_and_save(classifiers, test_features, test_files, output_file)
    
    # 计算并打印运行时间
    elapsed_time = time.time() - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 