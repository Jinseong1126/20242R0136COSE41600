import open3d as o3d
import numpy as np

def visualize_single_frame(pcd_file, save_camera_params='camera_params.json'):
    # numpy 배열 로드
    points = np.load(pcd_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z 좌표 사용

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Set Camera View", width=1920, height=1080)  # 창의 크기를 지정
    vis.add_geometry(pcd)
    vis.run()  # 시각화 창이 열리고, 카메라 시점을 조정할 수 있습니다.

    # 카메라 파라미터 저장
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(save_camera_params, param)
    print(f"카메라 파라미터가 {save_camera_params}에 저장되었습니다.")

    vis.destroy_window()

if __name__ == '__main__':
    sample_pcd = "/home/dosung/VoxelNeXt/data/custom_data/07_straight_walk/points/000150.npy"  # 예시 파일 경로
    visualize_single_frame(sample_pcd)
