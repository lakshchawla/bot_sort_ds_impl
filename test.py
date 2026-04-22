import numpy as np
from glob import glob
import os

import sys

path_to_botsort_parent = '/home/lab314/workspace/reid/botsort-tracker'

if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

ROOT_FRAME_DIR = "/home/lab314/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output"

def _extract_embedding(tensor_meta) -> np.ndarray | None:
    try:
        vec = tensor_meta['reid_vector']
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else None
    except Exception as e:
        print(f"[ReID] embed extract error: {e}")
        return None
    

from botsort.bot_sort import BoTSORT
# type(detections)

bs = BoTSORT()

for i in range(0,3427):
    if os.path.exists(f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"):
        #iterate through
        
        frame_content = np.load(f"{ROOT_FRAME_DIR}/batch_frame_{200}.npy", allow_pickle=True)

    detections = frame_content[0]['objects']
    bs.update(detections)
        

# frame_content[0]['objects']

# ds_ids   = [d['local_track_id'] for d in detections]
# bboxes   = [d['bbox'] for d in detections]
# tracker_confidence = [d['det_confidence'] for d in detections]
# features   = [d['reid_vector'] for d in detections]



