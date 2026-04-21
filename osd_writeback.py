"""
osd_writeback.py  –  drop-in replacements for _writeback_osd and the
association check in reid_probe.py.

Two things here:
  1. _writeback_osd()        – appends "ReID:<id>" to whatever label the
                               detector already put on the object, instead
                               of replacing it entirely.
  2. _log_associations()     – prints a small per-frame table so you can
                               verify that the tracker is matching the right
                               DS objects on each update cycle.
"""

import logging
import numpy as np
import pyds

log = logging.getLogger("reid_tracker")


# ---------------------------------------------------------------------------
# 1.  OSD writeback
#     Appends the ReID global ID to the existing detector label so the OSD
#     shows e.g.  "Person  ReID:7"  instead of just overwriting everything.
# ---------------------------------------------------------------------------

def _writeback_osd(frame_meta, output_stracks):
    """
    For every DS object in the frame, find the nearest BoTSORT track by
    centroid distance and write the global ReID id back into DS metadata.

    obj_meta.obj_label  →  "<original_label>  ReID:<global_id>"
    obj_meta.object_id  →  global_id   (so Kafka / nvmsgconv sees it too)

    Returns a list of (ds_local_id, global_id, dist_px) tuples so the
    caller can pass them to _log_associations().
    """
    associations = []

    if not output_stracks:
        return associations

    track_centroids = np.array(
        [t.centroid for t in output_stracks], dtype=np.float32
    )  # shape (M, 2)

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        r  = obj_meta.rect_params
        cx = r.left + r.width  / 2.0
        cy = r.top  + r.height / 2.0

        dists    = np.linalg.norm(
            track_centroids - np.array([cx, cy], dtype=np.float32), axis=1
        )
        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        global_id = output_stracks[best_idx].track_id

        # --- DS metadata updates ---
        ds_local_id = int(obj_meta.object_id)

        # Overwrite object_id with the global ReID id so downstream plugins
        # (nvmsgconv, message broker) see the stable cross-frame identity.
        obj_meta.object_id = global_id

        # Append to the existing detector label instead of replacing it.
        # obj_label is a 128-char fixed buffer in DS; truncate safely.
        original = obj_meta.obj_label.strip() or "obj"
        new_label = f"{original}  ReID:{global_id}"
        obj_meta.obj_label = new_label[:127]   # DS label buffer is 128 bytes

        associations.append((ds_local_id, global_id, best_dist))

        l_obj = l_obj.next

    return associations


# ---------------------------------------------------------------------------
# 2.  Association check / debug log
#     Call this right after _writeback_osd() while debugging.
#     Set your logger to DEBUG or INFO to see output; silence it in prod
#     by just not calling it (zero overhead).
# ---------------------------------------------------------------------------

def _log_associations(frame_id, source_id, detections, output_stracks, associations):
    """
    Prints a compact per-frame summary of what the tracker matched.

    Example output:
        [frame 42 | src 0]  3 dets  →  3 tracks active
          DS_id=14  →  ReID=3   dist=  8.2px   feat_ok=True
          DS_id=15  →  ReID=1   dist= 12.7px   feat_ok=True
          DS_id=16  →  ReID=7   dist=  5.1px   feat_ok=False  ← no reid vector
    """
    if not log.isEnabledFor(logging.DEBUG):
        return

    n_dets   = len(detections)
    n_tracks = len(output_stracks)

    log.debug(
        "[frame %d | src %s]  %d dets  →  %d tracks active",
        frame_id, source_id, n_dets, n_tracks
    )

    # Build a quick lookup: global_id → track object
    track_map = {t.track_id: t for t in output_stracks}

    for ds_local_id, global_id, dist_px in associations:
        track    = track_map.get(global_id)
        feat_ok  = (track is not None and track.smooth_feat is not None)

        # Warn if the centroid distance is suspiciously large
        # (suggests a mismatch — you may want to tune euc_thresh)
        flag = "  ← large dist, check euc_thresh" if dist_px > 80 else ""

        log.debug(
            "  DS_id=%-4d →  ReID=%-4d  dist=%6.1fpx   feat_ok=%-5s%s",
            ds_local_id, global_id, dist_px, str(feat_ok), flag
        )

    # Also log any active tracks that got no DS object matched to them this
    # frame — these are tracks kept alive from previous frames (lost/occluded).
    matched_global_ids = {gid for _, gid, _ in associations}
    ghost_tracks = [t for t in output_stracks if t.track_id not in matched_global_ids]
    for t in ghost_tracks:
        log.debug(
            "  [ghost]           ReID=%-4d  tracklet_len=%d  (no DS object this frame)",
            t.track_id, t.tracklet_len
        )
