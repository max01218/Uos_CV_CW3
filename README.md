# 场景识别项目

这个项目实现了场景识别任务，包括三种不同的方法：Run #1、Run #2和Run #3。

## 项目结构

- `version1.py`: 实现Run #1的程序
- `version2.py`: 实现Run #2的程序
- `version3.py`: 实现Run #3的程序
- `requirements.txt`: 项目依赖项
- `training/`: 训练数据目录
- `testing/`: 测试数据目录
- `run1.txt`: Run #1的预测结果输出文件
- `run2.txt`: Run #2的预测结果输出文件
- `run3.txt`: Run #3的预测结果输出文件
- `vocabulary.pkl`: Run #3的视觉词汇文件

## 实现方法

### Run #1: 使用k近邻分类器和"tiny image"特征

1. 使用"tiny image"特征表示图像
   - 裁剪图像中心的正方形区域
   - 调整大小为16x16
   - 将像素值打包成向量
   - 使特征向量具有零均值和单位长度
2. 使用k近邻分类器进行分类
   - 默认k值为3，可以根据需要调整

### Run #2: 使用视觉词袋特征和线性分类器

1. 使用密集采样的像素块作为特征
   - 使用8x8的像素块
   - 在x和y方向上每4个像素采样一次
2. 使用K-Means聚类创建视觉词汇
   - 默认使用500个聚类中心
   - 对每个像素块进行均值中心化和归一化
3. 使用视觉词袋特征表示图像
   - 将每个像素块映射到最近的视觉词
   - 计算视觉词袋直方图
4. 使用一组线性分类器（一对多分类器）进行分类
   - 为每个类别训练一个逻辑回归分类器

### Run #3: 使用密集SIFT特征和SVM分类器

1. 使用密集SIFT特征
   - 在多个尺度下提取SIFT特征（1.0, 0.75, 0.5）
   - 使用8像素的步长在图像上密集采样
2. 使用K-Means聚类创建视觉词汇
   - 默认使用500个聚类中心
3. 使用视觉词袋特征表示图像
   - 将每个SIFT描述符映射到最近的视觉词
   - 计算视觉词袋直方图
4. 使用SVM分类器进行分类
   - 使用RBF核函数
   - 对特征进行标准化处理

## 使用方法

1. 安装依赖项：
   ```
   pip install -r requirements.txt
   ```

2. 运行Run #1：
   ```
   python version1.py
   ```

3. 运行Run #2：
   ```
   python version2.py
   ```

4. 运行Run #3：
   ```
   python version3.py
   ```

5. 查看结果：
   程序运行完成后，预测结果将保存在相应的TXT文件中（run1.txt、run2.txt、run3.txt）。

## 输出格式

输出文件采用以下格式：
```
<image_name> <predicted_class>
```

例如：
```
0.jpg tallbuilding
1.jpg forest
2.jpg mountain
...
``` 