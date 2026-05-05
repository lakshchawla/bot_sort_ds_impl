import os
import sys
import cv2
import numpy as np
from glob import glob

# --- Visualization Functions ---
def convert_boxes(box_data, to_format="tlbr"):
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
    box_color  = (255, 255, 255)
    text_color = (255, 255, 255)
    thickness  = 2
    for box, track_id in box_data:
        if current_format == "tlwh":
            x, y, w, h = map(int, box)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
        elif current_format == "tlbr":
            x1, y1, x2, y2 = map(int, box)
            pt1 = (x1, y1)
            pt2 = (x2, y2)
        cv2.rectangle(bg_image, pt1, pt2, box_color, thickness)
        text   = f"GID: {track_id}"
        text_y = pt1[1] - 7 if pt1[1] - 7 > 15 else pt1[1] + 15
        cv2.putText(bg_image, text, (pt1[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)
    return bg_image

# --- Path setup ---
path_to_botsort_parent = './'
if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

ROOT_FRAME_DIR = "./deepstream_npy_output"

from botsort.bot_sort import BoTSORT
from multicam_tracker.clustering import Clustering, ID_Distributor
from multicam_tracker.cluster_track import MCTracker

# ── NEW: import the registry ──────────────────────────────────────────────────
from botsort.global_registry import GlobalRegistry

# --- Registry init -----------------------------------------------------------
registry = GlobalRegistry(
    match_threshold=0.25,   # cosine distance — tune this: lower = stricter
    min_frames=5,           # wait 5 frames before querying (embedding stabilises)
    max_emb=50,             # rolling buffer size per person
)

# --- Tracker init ------------------------------------------------------------
tracker = BoTSORT(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    track_buffer=600,
    match_thresh=0.8,
    with_reid=True,
    proximity_thresh=0.7,
    appearance_thresh=0.25,
    euc_thresh=0.1,
    fuse_score=True,
    frame_rate=30,
    max_batch_size=8,
    map_len=None,
    real_data=True,
    registry=registry,      # ← pass registry in
)

clustering    = Clustering(appearance_thresh=0.75, euc_thresh=0.3, match_thresh=0.8)
scene         = 'scene_061'
mc_tracker    = MCTracker(appearance_thresh=0.25, match_thresh=0.8, scene=scene)
id_distributor = ID_Distributor()

# --- Main loop ---------------------------------------------------------------
cur_frame  = 0
ACTIVE_FORMAT = "tlwh"

for i in range(3000):
    cur_frame += 1

    npy_path = f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        continue

    frame_content = np.load(npy_path, allow_pickle=True)
    detections    = frame_content[0]['objects']
    for d in detections:
        d['obj_meta'] = None

    # ── 1. BoTSORT frame-to-frame tracking ───────────────────────────────────
    all_tracks = tracker.update(detections)

    # ── 2. Registry step — runs AFTER tracker.update() ───────────────────────
    #       Reads tracker.tracked_stracks, assigns/reuses t_global_id on each.
    registry.step(tracker, frame_id=cur_frame)

    # ── 3. Build display data using t_global_id (set by registry) ────────────
    extracted_data = []
    for t in tracker.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data.append((t.tlwh, t.t_global_id))

    # ── 4. Print ──────────────────────────────────────────────────────────────
    for x in all_tracks:
        print(f"  track_id={x.track_id}  global_id={x.t_global_id}")
    print(f"  registry: {registry}")
    print()

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    bg_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    if extracted_data:
        formatted = convert_boxes(extracted_data, to_format=ACTIVE_FORMAT)
        bg_frame  = draw_boxes_on_bg(bg_frame, formatted, current_format=ACTIVE_FORMAT)

    cv2.imshow("Detections", bg_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# ── Final registry dump ───────────────────────────────────────────────────────
print("\n=== Final Gallery ===")
for e in registry.get_all_entries():
    print(e)