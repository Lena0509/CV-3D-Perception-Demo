import cv2
import matplotlib.pyplot as plt

# 读取图像
# 特征点提取（SIFT/ORB）  图像中“稳定且有区分度的点”，比如角点
# 特征描述子匹配          比较两个特征点的“描述子”，计算距离（相似度）
# 可视化匹配结果

# 基于OpenCV实现图像特征提取与匹配，采用ORB算法完成关键点检测与描述子匹配，并对匹配结果进行筛选与可视化

# SIFT：精度高，但慢（专利限制） ORB：速度快，适合实时应用

# 读取图片
img1 = cv2.imread('data/images/img1.jpg', 0)
img2 = cv2.imread('data/images/img2.jpg', 0)

# 创建ORB
orb = cv2.ORB_create()

# 检测关键点和描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 排序
matches = sorted(matches, key=lambda x: x.distance)

# 画匹配
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

# 只保留好的匹配
good_matches = [m for m in matches if m.distance < 50]

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# 显示
plt.imshow(img_matches)
plt.axis('off')
plt.show()