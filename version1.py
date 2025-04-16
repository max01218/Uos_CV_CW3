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
    创建tiny image特征
    1. 裁剪图像中心的正方形区域
    2. 调整大小为16x16
    3. 将像素值打包成向量
    4. 使特征向量具有零均值和单位长度
    """
    # 获取图像的宽度和高度
    width, height = img.size
    
    # 裁剪中心的正方形区域
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img_cropped = img.crop((left, top, right, bottom))
    
    # 调整大小为16x16
    img_resized = img_cropped.resize((size, size), Image.LANCZOS)
    
    # 将图像转换为numpy数组并展平
    img_array = np.array(img_resized).flatten()
    
    # 使特征向量具有零均值和单位长度
    img_array = img_array - np.mean(img_array)
    norm = np.linalg.norm(img_array)
    if norm > 0:
        img_array = img_array / norm
    
    return img_array

def load_training_data(training_dir):
    """
    加载训练数据，返回特征矩阵和标签
    """
    print("正在加载训练数据...")
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
        
        for img_file in image_files:
            # 加载图像
            img = Image.open(img_file)
            # 创建tiny image特征
            feature = create_tiny_image(img)
            features.append(feature)
            labels.append(class_name)
    
    return np.array(features), np.array(labels)

def load_test_data(test_dir):
    """
    加载测试数据，返回特征矩阵和图像文件名
    """
    print("正在加载测试数据...")
    features = []
    image_files = []
    
    # 获取所有测试图像
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    for img_file in test_images:
        # 加载图像
        img = Image.open(img_file)
        # 创建tiny image特征
        feature = create_tiny_image(img)
        features.append(feature)
        # 保存图像文件名（不含路径）
        image_files.append(os.path.basename(img_file))
    
    return np.array(features), image_files

def train_knn_classifier(features, labels, k=3):
    """
    训练k近邻分类器
    """
    print(f"训练k近邻分类器 (k={k})...")
    classifier = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    classifier.fit(features, labels)
    return classifier

def predict_and_save(classifier, test_features, test_files, output_file):
    """
    对测试数据进行预测并保存结果
    """
    print("对测试数据进行预测...")
    predictions = classifier.predict(test_features)
    
    # 保存预测结果到TXT文件
    with open(output_file, 'w') as f:
        for img_file, pred_class in zip(test_files, predictions):
            f.write(f"{img_file} {pred_class}\n")
    
    print(f"预测结果已保存到 {output_file}")

def main():
    # 设置路径
    training_dir = "training/training"
    test_dir = "testing/testing"
    output_file = "run1.txt"  # 修改输出文件名
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载训练数据
    train_features, train_labels = load_training_data(training_dir)
    
    # 加载测试数据
    test_features, test_files = load_test_data(test_dir)
    
    # 训练分类器
    k = 3  # 可以根据需要调整k值
    classifier = train_knn_classifier(train_features, train_labels, k)
    
    # 预测并保存结果
    predict_and_save(classifier, test_features, test_files, output_file)
    
    # 计算并打印运行时间
    elapsed_time = time.time() - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
