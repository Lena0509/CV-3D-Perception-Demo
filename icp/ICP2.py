import open3d as o3d
import numpy as np
import copy

#基于Open3D实现点云配准，利用ICP算法对点云进行迭代对齐，估计位姿变换矩阵，实现多视角点云融合基础流程

# 读取点云
source = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)

# target = source.translate((50, 0, 0))  # 人为平移一下 错误❌️，这样target==source，相当于移动了source
# 在使用Open3D时需要注意其点云操作为原地修改（in-place），若需要保留原始数据需进行深拷贝

target = copy.deepcopy(source)
# 再做平移
target.translate((0.05, 0, 0))

source.paint_uniform_color([1, 0, 0])  # 红
target.paint_uniform_color([0, 1, 0])  # 绿


# 显示初始状态（未对齐）
print("初始点云（未对齐）")
o3d.visualization.draw_geometries([source, target])

# ICP配准
threshold = 0.1  # 距离阈值
trans_init = np.eye(4)  # 初始变换矩阵

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("变换矩阵：")
print(reg_p2p.transformation)

# 应用变换
source.transform(reg_p2p.transformation)

# 显示对齐后
print("对齐后点云")
o3d.visualization.draw_geometries([source, target])