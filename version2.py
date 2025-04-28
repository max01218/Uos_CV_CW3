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

def process_single_image(args):
    """
    处理单张图像的辅助函数，用于并行处理
    """
    img_file, kmeans, pca, patch_size, stride = args
    img = Image.open(img_file)
    patches = extract_patches(img, patch_size, stride)
    patches_normalized = normalize_patches(patches)
    patches_pca = pca.transform(patches_normalized)
    
    # 使用批量预测代替单个预测
    labels = kmeans.predict(patches_pca)
    
    # 计算直方图
    hist = np.bincount(labels, minlength=kmeans.n_clusters)
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def load_training_data(training_dir, kmeans, pca, patch_size=8, stride=4):
    """
    并行加载训练数据
    """
    print("加载训练数据...")
    features = []
    labels = []
    
    # 获取所有类别目录
    class_dirs = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and not d.startswith('__')]
    
    n_jobs = multiprocessing.cpu_count()
    
    for class_name in class_dirs:
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"处理类别: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        # 准备并行处理的参数
        process_args = [(f, kmeans, pca, patch_size, stride) for f in image_files]
        
        # 并行处理该类别的所有图像
        class_features = Parallel(n_jobs=n_jobs)(
            delayed(process_single_image)(args) for args in tqdm(process_args)
        )
        
        features.extend(class_features)
        labels.extend([class_name] * len(image_files))
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir, kmeans, pca, patch_size=8, stride=4):
    """
    并行加载测试数据
    """
    print("加载测试数据...")
    
    # 获取所有测试图像并按数字顺序排序
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    test_images = sorted(test_images, key=lambda x: float(os.path.basename(x).split('.')[0]))
    
    # 准备并行处理的参数
    process_args = [(f, kmeans, pca, patch_size, stride) for f in test_images]
    
    # 并行处理所有测试图像
    n_jobs = multiprocessing.cpu_count()
    features = Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(args) for args in tqdm(process_args)
    )
    
    # 获取文件名列表
    image_files = [os.path.basename(f) for f in test_images]
    
    return np.array(features), image_files

def build_vocabulary(training_dir, num_clusters=500, patch_size=8, stride=4, max_patches_per_image=100, pca_components=64, batch_size=1000):
    """
    从训练图像中构建视觉词汇，使用并行处理加速
    """
    print("构建视觉词汇...")
    all_patches = []
    
    # 获取所有类别目录
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
            
        print(f"处理类别: {class_name}")
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))[:max_patches_per_image]
        
        # 并行处理图像
        class_patches = Parallel(n_jobs=n_jobs)(
            delayed(process_image)(f) for f in tqdm(image_files)
        )
        
        all_patches.extend(class_patches)
    
    # 将所有patches合并
    all_patches = np.vstack(all_patches)
    
    # 使用PCA降维
    print(f"使用PCA降维到 {pca_components} 维...")
    pca = PCA(n_components=pca_components, whiten=True)
    all_patches_pca = pca.fit_transform(all_patches)
    
    # 使用MiniBatchKMeans聚类
    print(f"使用MiniBatchKMeans聚类 {num_clusters} 个视觉词...")
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
    
    # 保存预测结果到TXT文件，确保按照文件名数字顺序排序
    results = list(zip(test_files, predictions))
    results.sort(key=lambda x: float(x[0].split('.')[0]))  # 按文件名中的数字排序
    
    with open(output_file, 'w') as f:
        for img_file, pred_class in results:
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"预测结果已保存到 {output_file}")

def main():
    # 设置参数
    patch_size = 8
    stride = 4
    num_clusters = 500
    pca_components = 64
    batch_size = 1000  # MiniBatchKMeans的批次大小
    
    # 设置路径
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run2.txt"
    model_file = "models.joblib"  # 模型保存文件
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查是否存在保存的模型
    if os.path.exists(model_file):
        print("加载已保存的模型...")
        models = load(model_file)
        kmeans, pca, classifiers = models['kmeans'], models['pca'], models['classifiers']
    else:
        print("训练新模型...")
        # 构建视觉词汇
        kmeans, pca = build_vocabulary(training_dir, num_clusters, patch_size, stride, 
                                     pca_components=pca_components, batch_size=batch_size)
        
        # 加载训练数据
        train_features, train_labels = load_training_data(training_dir, kmeans, pca, patch_size, stride)
        
        # 训练分类器
        classifiers = train_linear_classifiers(train_features, train_labels)
        
        # 保存模型
        print("保存模型...")
        models = {
            'kmeans': kmeans,
            'pca': pca,
            'classifiers': classifiers
        }
        dump(models, model_file)
    
    # 加载测试数据
    test_features, test_files = load_test_data(test_dir, kmeans, pca, patch_size, stride)
    
    # 预测并保存结果
    predict_and_save(classifiers, test_features, test_files, output_file)
    
    # 计算并打印运行时间
    elapsed_time = time.time() - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 