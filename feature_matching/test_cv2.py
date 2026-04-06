import cv2
import numpy as np
import matplotlib.pyplot as plt

# 基于OpenCV实现图像特征提取与匹配，采用ORB算法完成关键点检测，并通过Lowe’s ratio test筛选优质匹配点，结合RANSAC算法估计单应性矩阵，提高匹配精度与鲁棒性

# Lowe’s ratio test:用两个最相似匹配的距离比值筛选“可靠匹配”
# 用RANSAC？       去掉错误匹配（outliers）
# 单应性矩阵是什么？  描述两张图之间的几何变换关系


# 读取图像
img1 = cv2.imread('data/images/img1.jpg', 0)
img2 = cv2.imread('data/images/img2.jpg', 0)

# ORB特征
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# KNN匹配（比之前更高级）
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(des1, des2, k=2)

# 筛选优质匹配（Lowe's ratio test）
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# 至少需要4个点
if len(good) > 4:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    # 画匹配结果
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.figure(figsize=(12,6))
    plt.imshow(result)
    plt.axis('off')
    plt.show()

else:
    print("匹配点太少")