import os
import sys
import cv2
import numpy as np
from glob import glob

# --- Visualization Functions ---
def convert_boxes(box_data, to_format="tlbr"):
    """
    Converts a list of tuples (box, track_id) from 'tlwh' to 'tlbr' or ensures 'tlwh'.
    Input boxes are assumed to be in tlwh format: [x, y, w, h]
    """
    if not len(box_data):
        return []
        
    converted = []
    for box, track_id in box_data:
        x, y, w, h = box
        if to_format == "tlbr":
            converted.append(([x, y, x + w, y + h], track_id))
        elif to_format == "tlwh":
            converted.append(([x, y, w, h], track_id))
    return converted

def draw_boxes_on_bg(bg_image, box_data, current_format="tlwh"):
    """
    Draws thin blue boxes and their IDs on the given background.
    """
    box_color = (255, 255, 255)  # Blue for boxes
    text_color = (255, 255, 255) # Blue for text
    thickness = 2            # Low width

    for box, track_id in box_data:
        if current_format == "tlwh":
            x, y, w, h = map(int, box)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
        elif current_format == "tlbr":
            x1, y1, x2, y2 = map(int, box)
            pt1 = (x1, y1)
            pt2 = (x2, y2)

        # Draw the rectangle
        cv2.rectangle(bg_image, pt1, pt2, box_color, thickness)
        
        # Draw the ID text
        text = f"ID: {track_id}"
        x_int, y_int = pt1[0], pt1[1]
        
        # Position text slightly above the box. If too close to top margin, push it inside the box.
        text_y = y_int - 7 if y_int - 7 > 15 else y_int + 15
        
        cv2.putText(bg_image, text, (x_int, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)
        
    return bg_image

# --- Tracker Setup ---
path_to_botsort_parent = './'

if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

ROOT_FRAME_DIR = "./deepstream_npy_output"


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
    'max_batch_size' : 32,
    'track_buffer' : 600,             # FIXED: Prevents fatal memory bloat
    'with_reid' : True,
    
    # Assuming SCT/CLT/MCT refer to Single/Cross/Multi-Camera tracking
    # If these evaluate distance (lower is better), use 0.25. 
    # (If they evaluate similarity where 1.0 is best, then 0.8/0.9 is fine).
    'sct_appearance_thresh' : 0.25,  
    'sct_euclidean_thresh' : 0.1,    
    'clt_appearance_thresh' : 0.25,  
    'clt_euclidean_thresh' : 0.3,    
    'mct_appearance_thresh' : 0.25,  
    
    'frame_rate' : 30,
    'write_vid' : False,
}

tracker = BoTSORT(
    track_high_thresh=0.6,    # Standard YOLO/ByteTrack high threshold
    track_low_thresh=0.1,     # Correct: Used for second-association
    new_track_thresh=0.7,     # Strict confidence needed to spawn a new ID
    track_buffer=600,          # Standard memory retention (1 second at 30fps)
    match_thresh=0.8,         # First association matching threshold of 0.8 is standard 
    with_reid=True, 
    proximity_thresh=0.7,     # Maps to theta_iou = 0.5 
    appearance_thresh=0.25,   # Maps to theta_emb = 0.25 
    euc_thresh=0.1,           # (Vestigial if pure IoU is used, but safe to keep)
    fuse_score=True,          
    frame_rate=30, 
    max_batch_size=8, 
    map_len=None, 
    real_data=True,
    # is_deepstream_app=True
)

clustering = Clustering(appearance_thresh=0.75, euc_thresh=args['clt_euclidean_thresh'], match_thresh=0.8)
scene = 'scene_061'
mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, scene=scene)
id_distributor = ID_Distributor()

# --- Main Processing Loop ---
cur_frame = 0
ACTIVE_FORMAT = "tlwh" 

for i in range(3000):
    cur_frame += 1
    
    # Load frame data
    npy_path = f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"
    if os.path.exists(npy_path):
        frame_content = np.load(npy_path, allow_pickle=True)
    else:
        print(f"File not found: {npy_path}")
        continue

    detections = frame_content[0]['objects']
    for d in detections:
        d['obj_meta'] = None 
    
    # 1. Update Tracker
    all_tracks= tracker.update(detections)
    
    # 2. Extract bounding boxes and apply ID assignment logic
    extracted_data = [] # Will hold tuples of (box, ID)
    
    for t in tracker.tracked_stracks:
        # New ID assignment logic requested
        if t.t_global_id == 0:
            t.t_global_id = id_distributor.assign_id()
            
        # Group the bounding box array with its respective ID
        if hasattr(t, 'tlwh'):
            extracted_data.append((t.tlwh, t.t_global_id))
    
    for x in all_tracks:
        print("assigned id: ", x.t_global_id)
        
    print()

    # 3. Visualization Integration
    # Create a plain white 1920x1080 background
    bg_frame = np.zeros((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Format and draw if we have valid boxes
    if len(extracted_data) > 0:
        formatted_boxes = convert_boxes(extracted_data, to_format=ACTIVE_FORMAT)
        bg_frame = draw_boxes_on_bg(bg_frame, formatted_boxes, current_format=ACTIVE_FORMAT)
    
    # Show the frame
    print("Displaying frame. Press any key to close the window...")
    cv2.imshow("Detections", bg_frame)
    cv2.waitKey(0) 
    
cv2.destroyAllWindows()