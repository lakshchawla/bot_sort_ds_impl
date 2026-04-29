import numpy as np
from glob import glob
import os

import sys

import cv2
import numpy as np
import copy

path_to_botsort_parent = './'

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
    



args = {
    'max_batch_size' : 32,  # maximum input batch size of reid model
    'track_buffer' : 150,  # the frames for keep lost tracks
    'with_reid' : True,  # whether to use reid model's out feature map at first association
    'sct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do single-cam tracking
    'sct_euclidean_thresh' : 0.1,  # threshold of euclidean distance when do single-cam tracking

    'clt_appearance_thresh' : 0.35,  # threshold of appearance feature cosine distance when do multi-cam clustering
    'clt_euclidean_thresh' : 0.3,  # threshold of euclidean distance when do multi-cam clustering

    'mct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do cluster tracking (not important)

    'frame_rate' : 30,  # your video(camera)'s fps
    'write_vid' : False,  # write result to video
    }

tracker = BoTSORT(
    track_high_thresh=0.6, 
    track_low_thresh=0.3, 
    new_track_thresh=0.5, 
    track_buffer=30, 
    match_thresh=0.8, 
    with_reid=True, 
    proximity_thresh=0.5, 
    appearance_thresh=0.4, 
    euc_thresh=0.1, 
    fuse_score=True, 
    frame_rate=30, 
    max_batch_size=8, 
    map_len=None, 
    real_data=True
)
clustering = Clustering(appearance_thresh= 0.65, euc_thresh=args['clt_euclidean_thresh'],
                        match_thresh=0.4)
scene = 'scene_061'
mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.3, scene=scene)
id_distributor = ID_Distributor()
# type(detections)


l256 = 0
l102 = 0
le = 0


cur_frame = 0
for i in range(0,3427):
    cur_frame += 1
    if os.path.exists(f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"):
        #iterate through
        frame_content = np.load(f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy", allow_pickle=True)

    detections = frame_content[0]['objects']
    
    for d in detections:
        if len(d['reid_vector']) == 256: l256+=1
        elif len(d['reid_vector']) == 102: l102+=1
        else: le +=1
        
    # groups = clustering.update([tracker], cur_frame, scene)
    # # mc_tracker.update([tracker], groups)
    # clustering.update([tracker], cur_frame, scene)

    # if i == 600: break
    # break


print(l256, l102, le)