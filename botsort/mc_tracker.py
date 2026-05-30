import numpy as np
from collections import deque

from .bot_sort import BoTSORT
from .global_registry import GlobalRegistry
from . import matching

import pdb
import sys
import math

class MCTracker(object):
    def __init__(
        self,
        registry: GlobalRegistry,
        frame_size: tuple = (1920, 1080),
        roi_padding: tuple = (0, 0),
        exit_timeout: int = 90,
        boundary_margin: int = 20,
    ):
        self.registry = registry
        self.frame_w, self.frame_h = frame_size
        self.boundary_margin = boundary_margin

        self.tracked_stracks = set()
        self.lost_stracks = set()
        self.removed_stracks = set()

        self.frame_id = 0

        self._pending_exits: dict[int, tuple[int, object]] = {}
        self.exit_timeout = exit_timeout

    def update_global(self, trackers: list[BoTSORT]):
        self.frame_id = trackers[0].frame_id

        self.tracked_stracks = set()
        self.lost_stracks = set()
        self.removed_stracks = set()

        for tracker in trackers:
            self.tracked_stracks.update(tracker.tracked_stracks)
            self.lost_stracks.update(tracker.lost_stracks)
            self.removed_stracks.update(tracker.removed_stracks)

        edge_lost = self._get_boundary_tracks(self.lost_stracks)

        edge_entered = [t for t in self.tracked_stracks if t.is_touching_edge == True and t.t_global_id == 0]

        # print(f"[MCT] {len(edge_lost) == 0, len(edge_entered) == 0}")

        # if edge_lost:
        #     print(f"[MCT] Edge Lost:    {[t.t_global_id for t in edge_lost]}")
        # if edge_entered:
        #     print(f"[MCT] Edge Entered: {[t.track_id for t in edge_entered]}")
    
        emb_dists = matching.embedding_distance(edge_lost, edge_entered)

        if len(emb_dists) and len(edge_lost) and len(edge_entered):
            print(len(edge_lost), len(edge_entered), ' - ', [track.track_id for track in edge_entered])
        #     print(f"[EMBD]: {emb_dists}")
            
        valid_mask = np.array(
                emb_dists < 0.2
            )
        hat_emb_dists = np.ones_like(emb_dists)
        hat_emb_dists[valid_mask] = emb_dists[valid_mask]
        hat_emb_dists
        matches, u_track, u_detection = matching.linear_assignment(hat_emb_dists, thresh=0.8)

            
        for itracked, idet in matches:
            track_init = edge_lost[itracked]
            track_curr = edge_entered[idet]
            
            # mct_state: change to camera_shift.
            print(f"[MCT] Matched {track_init.track_id} with {track_curr.track_id}")
            
            # just for visualization, 
            track_curr.t_global_id = track_init.track_id
            
        
        '''
        To be added: 
            1. Directional reference.
        '''
    
        # for track in edge_lost:
        #     if track.t_global_id != 0 and track.smooth_feat is not None:
        #         self._pending_exits[track.t_global_id] = (frame_id, track)

        # # Expire exits that were never matched within the timeout window
        # stale = [
        #     gid for gid, (ef, _) in self._pending_exits.items()
        #     if frame_id - ef > self.exit_timeout
        # ]
        # for gid in stale:
        #     del self._pending_exits[gid]

        # # Re-identify entering tracks against pending boundary exits
        # if edge_entered:
        #     self._match_cross_cam_entries(edge_entered, frame_id)

    # def _match_cross_cam_entries(self, entering_tracks, frame_id: int):
    #     for track in entering_tracks:
    #         if track.smooth_feat is None:
    #             continue

    #         best_entry, best_dist = self.registry.query(track.smooth_feat)

    #         # Only accept the match if it corresponds to a confirmed boundary exit —
    #         # this prevents re-assigning an ID for same-camera occlusions/redetections.
    #         if best_entry is None or best_entry.global_id not in self._pending_exits:
    #             continue

    #         exit_frame, _ = self._pending_exits[best_entry.global_id]
    #         print(
    #             f"[MCT] Cross-cam re-ID: track_id={track.track_id} → "
    #             f"gid={best_entry.global_id} (cos_dist={best_dist:.3f}, "
    #             f"exit {frame_id - exit_frame} frames ago)"
    #         )
    #         track.t_global_id = best_entry.global_id
    #         self.registry._link_existing(best_entry, track.track_id)
    #         best_entry.add_embedding(track.smooth_feat, frame_id, track.tlwh)
    #         del self._pending_exits[best_entry.global_id]

    def _get_boundary_tracks(self, tracks):
        boundary = []
        for track in tracks:
            x, y, w, h = track.tlwh
            x2, y2 = x + w, y + h
            m = self.boundary_margin
            if (x <= m or y <= m or
                    x2 >= self.frame_w - m or
                    y2 >= self.frame_h - m):
                boundary.append(track)
        return boundary