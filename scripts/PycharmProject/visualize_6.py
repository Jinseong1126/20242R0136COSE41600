import os
import numpy as np
import open3d as o3d
import math


def load_point_cloud(pcd_path):
    points = np.load(pcd_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z 좌표
    return point_cloud


def load_detections(result_file, score_threshold=0.15):
    pred_boxes = []
    with open(result_file, 'r') as f:
        for line in f:
            elements = line.strip().split()
            if len(elements) < 11:
                continue  # 잘못된 라인은 스킵

            # x, y, z, dx, dy, dz, yaw, vx, vy, score, label 순서로 가정
            box = list(map(float, elements[:7]))  # x, y, z, dx, dy, dz, yaw
            score = float(elements[9])
            label = int(elements[10])

            # 보행자(label == 9)만 선택하며 SCORE 기준 적용
            if label == 9 and score >= score_threshold:
                pred_boxes.append(box)

    return np.array(pred_boxes)


def get_rotation_matrix(yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def visualize_and_capture(point_cloud, pred_boxes, output_path, camera_params_path='camera_params.json'):
    vis = o3d.visualization.Visualizer()

    # 카메라 파라미터 읽기
    camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)

    # 창을 띄우지 않고 백그라운드에서 실행
    vis.create_window(visible=False, width=camera_params.intrinsic.width, height=camera_params.intrinsic.height)
    vis.add_geometry(point_cloud)

    for box in pred_boxes:
        center = box[:3]
        size = box[3:6]
        yaw = box[6]

        # 바운딩 박스 생성
        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = center
        bbox.extent = size
        bbox.R = get_rotation_matrix(yaw)
        bbox.color = (1, 0, 0)  # 빨간색

        vis.add_geometry(bbox)

    vis.get_render_option().point_size = 2.0  # 포인트 크기 조절

    # 카메라 파라미터 적용
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # 렌더링
    vis.poll_events()
    vis.update_renderer()

    # 이미지 저장
    vis.capture_screen_image(output_path)
    print(f"이미지 저장 완료: {output_path}")

    vis.destroy_window()


def process_subfolder(subfolders, index):
    folder_path = subfolders[index]
    subfolder_name = os.path.basename(folder_path)
    points_path = os.path.join(folder_path, "points")
    results_path = f"/home/dosung/VoxelNeXt/tools/result/{subfolder_name[:2]}"
    output_folder = f"./visualization_results/{subfolder_name}"  # 캡처 이미지 저장 폴더
    os.makedirs(output_folder, exist_ok=True)

    point_files = sorted([f for f in os.listdir(points_path) if f.endswith('.npy')])
    for file_name in point_files:
        frame_id = file_name.split('.')[0]
        pcd_path = os.path.join(points_path, file_name)
        result_file = os.path.join(results_path, f"{frame_id}.txt")
        output_path = os.path.join(output_folder, f"{frame_id}.png")  # 이미지 저장 경로

        try:
            # 포인트 클라우드 로드
            point_cloud = load_point_cloud(pcd_path)

            # 추론 결과 로드 (보행자만)
            pred_boxes = load_detections(result_file)

            # 시각화 및 이미지 캡처
            visualize_and_capture(point_cloud, pred_boxes, output_path, camera_params_path='camera_params.json')
        except Exception as e:
            print(f"하위 폴더 {subfolder_name}의 파일 {file_name} 처리 중 오류 발생: {e}")


def main(index):
    # 하위 폴더 리스트
    base_folder = "/home/dosung/VoxelNeXt/data/custom_data"
    subfolders = sorted(
        [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

    # 하위 폴더 처리
    print(f"처리 중인 하위 폴더 {index}: {os.path.basename(subfolders[index])}")
    process_subfolder(subfolders, index)


if __name__ == '__main__':
    # 하위 폴더 인덱스를 입력하여 실행 (예: 0)
    folder_index = 5  # 원하는 하위 폴더 인덱스 입력
    main(folder_index)
