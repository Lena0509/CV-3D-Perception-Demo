import open3d as o3d
import numpy as np

# 点云配准对齐(bunny版本）

# 读取点云
source = o3d.io.read_point_cloud("data/point_clouds/bun000.ply")
target = o3d.io.read_point_cloud("data/point_clouds/bun045.ply")

o3d.visualization.draw_geometries([source])
o3d.visualization.draw_geometries([target])

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