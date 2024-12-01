import os
import subprocess

def create_videos_from_folders(base_path, output_video_folder, fps=10):
    # 하위 폴더 확인
    subfolders = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    os.makedirs(output_video_folder, exist_ok=True)

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        output_video_path = os.path.join(output_video_folder, f"{subfolder_name}.mp4")

        # 이미지 파일 정렬
        images = sorted([os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith(('.png', '.jpg'))])

        # 임시 이미지 시퀀스 폴더
        if not images:
            print(f"Warning: No images found in {subfolder}. Skipping...")
            continue

        # ffmpeg 명령어 실행
        cmd = [
            "ffmpeg",
            "-y",  # 기존 파일 덮어쓰기
            "-framerate", str(fps),  # 프레임 레이트 설정
            "-i", os.path.join(subfolder, "%06d.png"),  # 이미지 이름 패턴
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]

        print(f"Processing video for folder: {subfolder_name}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Video created: {output_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error while creating video for {subfolder_name}: {e}")


if __name__ == '__main__':
    # 이미지가 있는 베이스 경로
    base_path = "/home/dosung/pythonProject/VoxelNeXt/COSE416_HW1_tutorial/visualization_results/01_straight_walk"
    # 결과 영상 저장 경로
    output_video_folder = "/home/dosung/pythonProject/VoxelNeXt/COSE416_HW1_tutorial/visualization_videos"
    # 초당 프레임 수
    fps = 10

    # 영상 생성 실행
    create_videos_from_folders(base_path, output_video_folder, fps)
