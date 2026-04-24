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
from multicam_tracker.clustering import Clustering, ID_Distributor
from multicam_tracker.cluster_track import MCTracker

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

tracker = BoTSORT()
clustering = Clustering(appearance_thresh= 0.75, euc_thresh=args['clt_euclidean_thresh'],
                        match_thresh=0.8)
scene = 'scene_061'
mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
id_distributor = ID_Distributor()
# type(detections)

cur_frame = 0
for i in range(0,3427):
    cur_frame += 1
    if os.path.exists(f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"):
        #iterate through
        frame_content = np.load(f"{ROOT_FRAME_DIR}/batch_frame_{200+i}.npy", allow_pickle=True)

    detections = frame_content[0]['objects']
    tracker.update(detections)

    for t in tracker.tracked_stracks:
        t.t_global_id = id_distributor.assign_id()
    
    groups = clustering.update([tracker], cur_frame, scene)
    # mc_tracker.update([tracker], groups)
    clustering.update([tracker], cur_frame, scene)

    
    if i == 5: break
        

