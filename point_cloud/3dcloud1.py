import open3d as o3d

# 读取自己的ply文件

pcd1 = o3d.io.read_point_cloud("data/point_clouds/top2.ply")

print(pcd1)

o3d.visualization.draw_geometries([pcd1])

# 体素降采样
down = pcd1.voxel_down_sample(voxel_size=0.002)

o3d.visualization.draw_geometries([down])

print(down)

# 统计滤波去噪,移除离群点
cl, ind = pcd1.remove_statistical_outlier(nb_neighbors = 20, std_ratio = 0.5)

o3d.visualization.draw_geometries([cl])

# 上色
pcd1.paint_uniform_color([0, 0.9, 1])  # 红色
o3d.visualization.draw_geometries([pcd1])