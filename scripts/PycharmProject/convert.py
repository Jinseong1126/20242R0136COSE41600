import os
import numpy as np
import open3d as o3d

def convert_pcd_to_npy(pcd_dir, npy_dir):
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
    pcd_files.sort()  # 파일 정렬 (옵션)
    for idx, pcd_file in enumerate(pcd_files):
        pcd_path = os.path.join(pcd_dir, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        # 포인트 추출
        points = np.asarray(pcd.points)
        # 좌표계 변환
        points_transformed = np.zeros_like(points)
        points_transformed[:, 0] = points[:, 1]         # Forward(Y) → Forward(X)
        points_transformed[:, 1] = -points[:, 0]        # Right(X) → Left(-Y)
        points_transformed[:, 2] = points[:, 2]         # Up(Z) → Up(Z)
        # intensity 직접 추출 (4번째 필드가 intensity인 경우)
        intensity = np.asarray(pcd.colors)[:, 0] if pcd.has_colors() else np.zeros((points.shape[0],))
        # 포인트와 intensity 결합
        points_intensity = np.hstack((points_transformed, intensity.reshape(-1, 1)))
        # NPY 파일 저장
        npy_filename = f'{idx:06d}.npy'
        npy_path = os.path.join(npy_dir, npy_filename)
        np.save(npy_path, points_intensity)
        print(f'Converted {pcd_file} to {npy_filename}')

# 시나리오 리스트
scenarios = [
    "01_straight_walk",
    "02_straight_duck_walk",
    "03_straight_crawl",
    "04_zigzag_walk",
    "05_straight_duck_walk",
    "06_straight_crawl",
    "07_straight_walk"
]

# 각 시나리오별 PCD 파일 변환
for scenario in scenarios:
    pcd_dir = f'COSE416_HW1_tutorial/COSE416_HW1_data_v1/data/{scenario}/pcd/'
    npy_dir = f'/home/dosung/VoxelNeXt/data/custom/{scenario}/points/'
    convert_pcd_to_npy(pcd_dir, npy_dir)