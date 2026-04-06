import open3d as o3d

# 基于Open3D实现点云处理流程，包括点云可视化、降采样及去噪处理，提升三维数据质量，为后续感知/重建任务提供基础支持

# 使用Open3D自带点云
pcd = o3d.data.PCDPointCloud()
point_cloud = o3d.io.read_point_cloud(pcd.path)

# 显示点云
o3d.visualization.draw_geometries([point_cloud])

# 体素降采样
downsampled = point_cloud.voxel_down_sample(voxel_size=0.05)

o3d.visualization.draw_geometries([downsampled])

# 去噪（统计滤波）
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

clean_cloud = point_cloud.select_by_index(ind)

o3d.visualization.draw_geometries([clean_cloud])