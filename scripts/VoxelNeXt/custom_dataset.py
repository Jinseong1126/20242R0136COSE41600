import os
from pathlib import Path
import numpy as np
from pcdet.datasets.dataset import DatasetTemplate

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = 'train' if self.training else 'test'
        if hasattr(self.dataset_cfg, 'DATA_SPLIT'):
            self.split = self.dataset_cfg.DATA_SPLIT.get('train' if self.training else 'test', self.split)

        # 데이터 경로 수정
        self.root_split_path = self.root_path

        self.sample_file_list = self._get_sample_file_list()
        self.num_samples = len(self.sample_file_list)

        if self.logger is not None:
            self.logger.info(f'Total samples for CUSTOM dataset: {self.num_samples}')

    def _get_sample_file_list(self):
        points_dir = self.root_split_path
        # 모든 하위 디렉토리에서 .npy 파일 검색
        sample_files = sorted(points_dir.rglob('*.npy'))
        return sample_files

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample_file = self.sample_file_list[index]
        points = np.load(sample_file)
        
        num_points = points.shape[0]
        timestamps = np.zeros((num_points, 1), dtype=points.dtype)
    	
        points = np.hstack((points, timestamps))
    
        frame_id = sample_file.stem  # 파일명에서 확장자 제거

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
