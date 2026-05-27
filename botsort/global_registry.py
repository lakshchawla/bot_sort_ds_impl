from __future__ import annotations
import math
import numpy as np
from collections import deque
from typing import Optional
import faiss
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


CAMERA_TOPOLOGY: dict[tuple[int, str], list[tuple[int, str, float]]] = {
    (0, 'bottom'):[(1, 'bottom',  0.0)],
    (1, 'bottom'):[(0, 'bottom', 0.0)],
    (1, 'bottom'):[(0, 'left', 0.1)],
}

MIN_CROSSING_FRAMES = 10   
MAX_CROSSING_FRAMES = 30   



class GalleryEntry:
    def __init__(
        self,
        global_id:  int,
        feat:       np.ndarray,
        track_id:   int,
        frame_id:   int,
        bbox:       np.ndarray,
        cam_id:     int = 0,
        max_emb:    int = 50,
    ):
        self.global_id  = global_id
        self.active_tid = track_id
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()
        self.cam_id     = cam_id
        self.exit_side: Optional[str] = None   # set at deactivation: 'left'|'right'|'top'|'bottom'|None

        self.embeddings: deque = deque(maxlen=max_emb)
        self.embeddings.append(feat)
        self.centroid = feat.copy()

    def _recompute_centroid(self):
        c = np.mean(self.embeddings, axis=0).astype(np.float32)
        n = np.linalg.norm(c)
        self.centroid = c / n if n > 1e-9 else c

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray,
                      cam_id: Optional[int] = None):
        self.embeddings.append(feat)
        self._recompute_centroid()
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()
        if cam_id is not None:
            self.cam_id = cam_id

    def similarity(self, feat: np.ndarray) -> float:
        n = np.linalg.norm(feat)
        if n < 1e-9 or np.linalg.norm(self.centroid) < 1e-9:
            return 0.0
        return float(np.dot(self.centroid, feat / n))

    def cosine_distance(self, feat: np.ndarray) -> float:
        return 1.0 - self.similarity(feat)

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, "
                f"tid={self.active_tid}, "
                f"cam={self.cam_id}, "
                f"exit={self.exit_side}, "
                f"n_emb={len(self.embeddings)}, "
                f"last_frame={self.last_frame})")

class GlobalRegistry:
    def __init__(
        self,
        match_threshold:       float = 0.35,
        cross_match_threshold: float = 0.50,
        min_frames:            int   = 5,
        max_emb:               int   = 50,
        emb_dim:               int   = 256,
        min_gap_frames:        int   = 10,
        intra_cam_gap:         int   = 30,
        w_appearance:          float = 0.5,
        w_spatial:             float = 0.2,
        w_time:                float = 0.2,
        frame_w:               int   = 1920,
        frame_h:               int   = 1080,
        edge_margin:           int   = 20,
    ):
        assert abs(w_appearance + w_spatial + w_time - 1.0) < 1e-6, \
            "w_appearance + w_spatial + w_time must sum to 1.0"

        self.match_threshold       = match_threshold
        self.cross_match_threshold = cross_match_threshold
        self.min_frames            = min_frames
        self.max_emb               = max_emb
        self.min_gap_frames        = min_gap_frames
        self.intra_cam_gap         = intra_cam_gap
        self.w_appearance          = w_appearance
        self.w_spatial             = w_spatial
        self.w_time                = w_time
        self.frame_w               = frame_w
        self.frame_h               = frame_h
        self.edge_margin           = edge_margin

        self._entries:       list[GalleryEntry] = []
        self._tid_to_gid:    dict[int, int]     = {}
        # per-camera counters: cam_idx → number of identities born on that camera
        self._cam_id_ctrs:   dict[int, int]     = {}

        self._emb_dim   = emb_dim
        self._index_cpu = faiss.IndexFlatIP(emb_dim)

        res             = faiss.StandardGpuResources()
        self._index     = faiss.index_cpu_to_gpu(res, 0, self._index_cpu)

        self._faiss_pos_to_gid: list[int] = []
        self._gid_to_entry: dict[int, GalleryEntry] = {}
        self._gid_to_pos:   dict[int, int]          = {}   # gid → index in _entries

    def _new_global_id(self, cam_idx: int = 0) -> int:
        """Return a stable global ID scoped to cam_idx.
        Single-cam (cam_idx=0):  1, 2, 3, ...
        Multi-cam source N:      N*1000+1, N*1000+2, ...
        """
        self._cam_id_ctrs[cam_idx] = self._cam_id_ctrs.get(cam_idx, 0) + 1
        return (cam_idx + 1) * 1_000 + self._cam_id_ctrs[cam_idx]

    def query(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        if self._index.ntotal == 0 or feat is None:
            return None, 1.0

        vec = feat.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)

        sims, idxs = self._index.search(vec, k=1)

        sim       = float(sims[0, 0])
        faiss_pos = int(idxs[0, 0])

        if faiss_pos < 0:          
            return None, 1.0

        cos_dist = 1.0 - sim       

        if cos_dist < self.match_threshold:
            gid   = self._faiss_pos_to_gid[faiss_pos]
            entry = self._get_entry_by_gid(gid)
            return entry, cos_dist

        return None, cos_dist

    def query_batch(
        self,
        feats: list[np.ndarray],
    ) -> list[tuple[Optional[GalleryEntry], float]]:
        results: list[tuple[Optional[GalleryEntry], float]] = [
            (None, 1.0)
        ] * len(feats)

        if self._index.ntotal == 0 or not feats:
            return results

        valid_feats = []
        valid_idx   = []
        for i, f in enumerate(feats):
            if f is not None and np.linalg.norm(f) > 1e-9:
                valid_feats.append(f.astype(np.float32) / np.linalg.norm(f))
                valid_idx.append(i)

        if not valid_feats:
            return results

        query_mat = np.ascontiguousarray(
            np.stack(valid_feats, axis=0).astype(np.float32)
        )

        sims, idxs = self._index.search(query_mat, k=1)

        for qi, orig_i in enumerate(valid_idx):
            sim       = float(sims[qi, 0])
            faiss_pos = int(idxs[qi, 0])

            if faiss_pos < 0:
                continue

            cos_dist = 1.0 - sim

            if cos_dist < self.match_threshold:
                gid   = self._faiss_pos_to_gid[faiss_pos]
                entry = self._get_entry_by_gid(gid)
                results[orig_i] = (entry, cos_dist)
            else:
                results[orig_i] = (None, cos_dist)

        return results

    def _register_new(self, track_id, feat, frame_id, bbox, cam_idx: int = 0):
        gid = self._new_global_id(cam_idx)
        entry = GalleryEntry(
            global_id=gid, feat=feat, track_id=track_id,
            frame_id=frame_id, bbox=bbox, cam_id=cam_idx, max_emb=self.max_emb,
        )
        pos = len(self._entries)
        self._entries.append(entry)
        self._gid_to_entry[gid] = entry
        self._gid_to_pos[gid]   = pos
        self._tid_to_gid[track_id] = gid

        vec = entry.centroid.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)
        self._index.add(vec)
        self._faiss_pos_to_gid.append(gid)

        return gid


    def _link_existing(self, entry: GalleryEntry, track_id: int) -> bool:
        """Link track_id to an existing gallery entry.

        Returns False (no-op) if the entry is already active with a *different*
        track — caller should treat this track as a new entry instead.
        """
        if entry.active_tid is not None and entry.active_tid != track_id:
            return False
        old_tid = entry.active_tid
        if old_tid is not None and old_tid in self._tid_to_gid:
            del self._tid_to_gid[old_tid]
        entry.active_tid = track_id
        self._tid_to_gid[track_id] = entry.global_id
        return True

    def deactivate_track(self, track_id: int):
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        entry = self._gid_to_entry.get(gid)  # O(1)
        if entry is not None:
            entry.active_tid = None


    def step(self, tracker, frame_id: int):
        current_tids = {t.track_id for t in tracker.tracked_stracks}

        linked_tids = set(self._tid_to_gid.keys())
        for gone_tid in linked_tids - current_tids:
            self.deactivate_track(gone_tid)

        for track in tracker.tracked_stracks:
            tid  = track.track_id
            feat = track.smooth_feat   
            bbox = track.tlwh

            if track.t_global_id != 0:
                gid = track.t_global_id
                entry = self._get_entry_by_gid(gid)
                if entry is not None and feat is not None:
                    entry.add_embedding(feat, frame_id, bbox)
                continue

            if track.tracklet_len < self.min_frames:
                continue   

            if feat is None:
                gid = self._register_new(tid, np.zeros(self._emb_dim, dtype=np.float32), frame_id, bbox)
                track.t_global_id = gid
                continue

            best_entry, best_dist = self.query(feat)

            if best_entry is not None:
                print(
                    f"[REGISTRY] Re-entry detected: "
                    f"track_id={tid} → global_id={best_entry.global_id} "
                    f"(cos_dist={best_dist:.3f})"
                )
                self._link_existing(best_entry, tid)
                best_entry.add_embedding(feat, frame_id, bbox)
                track.t_global_id = best_entry.global_id
            else:
                gid = self._register_new(tid, feat, frame_id, bbox)
                track.t_global_id = gid
                print(
                    f"[REGISTRY] New entry: "
                    f"track_id={tid} → global_id={gid} "
                    f"(closest_dist={best_dist:.3f})"
                )
        
        if any(t.t_global_id != 0 for t in tracker.tracked_stracks):
            self._rebuild_faiss_index()

    def step_source(self, tracker, cam_idx: int, frame_id: int):
        """
        Per-source registry update for multi-camera DeepStream pipelines.

        Use this instead of step() when sources arrive asynchronously
        (separate batches or mixed batches where each source has its own
        frame_num).  Track IDs are namespaced as cam_base = cam_idx * 100_000
        inside _tid_to_gid so two cameras with the same BoTSORT track_id never
        collide.  Global IDs follow cam_idx * 1_000 + per-cam-counter.

        The existing step() method is unchanged and still works for single-cam.
        """
        cam_base = cam_idx * 100_000

        # ── Deactivate tracks that left this camera's tracked_stracks ──────
        current_cam_tids = {cam_base + t.track_id for t in tracker.tracked_stracks}
        cam_linked       = {t for t in self._tid_to_gid
                            if cam_base <= t < cam_base + 100_000}
        for gone_tid in cam_linked - current_cam_tids:
            self.deactivate_track(gone_tid)

        # ── Assign / update global IDs ────────────────────────────────────
        for track in tracker.tracked_stracks:
            cam_tid = cam_base + track.track_id
            feat    = track.smooth_feat
            bbox    = track.tlwh

            # Track already has a registered identity — just update embedding
            if track.t_global_id != 0:
                entry = self._get_entry_by_gid(track.t_global_id)
                if entry is not None:
                    # Re-link if the entry's active_tid drifted (e.g. lost→tracked)
                    if entry.active_tid != cam_tid:
                        if entry.active_tid is not None and entry.active_tid in self._tid_to_gid:
                            del self._tid_to_gid[entry.active_tid]
                        entry.active_tid = cam_tid
                        self._tid_to_gid[cam_tid] = track.t_global_id
                    if feat is not None:
                        entry.add_embedding(feat, frame_id, bbox)
                continue

            # Not yet confirmed long enough — skip
            if track.tracklet_len < self.min_frames:
                continue

            # No feature available — register with a zero vector placeholder
            if feat is None:
                gid = self._register_new(
                    cam_tid, np.zeros(self._emb_dim, dtype=np.float32),
                    frame_id, bbox, cam_idx=cam_idx,
                )
                track.t_global_id = gid
                continue

            # Query the shared gallery (cross-cam re-ID happens here)
            best_entry, best_dist = self.query(feat)

            if best_entry is not None:
                # Guard: don't steal an identity that belongs to a live track
                if (best_entry.active_tid is not None
                        and best_entry.active_tid in self._tid_to_gid):
                    gid = self._register_new(cam_tid, feat, frame_id, bbox,
                                             cam_idx=cam_idx)
                    track.t_global_id = gid
                    print(
                        f"[REGISTRY] Occupied entry blocked: "
                        f"cam_tid={cam_tid} → new gid={gid} "
                        f"(matched gid={best_entry.global_id} held by "
                        f"tid={best_entry.active_tid})"
                    )
                else:
                    self._link_existing(best_entry, cam_tid)
                    best_entry.add_embedding(feat, frame_id, bbox)
                    track.t_global_id = best_entry.global_id
                    print(
                        f"[REGISTRY] Re-entry cam{cam_idx}: "
                        f"cam_tid={cam_tid} → gid={best_entry.global_id} "
                        f"(cos_dist={best_dist:.3f})"
                    )
            else:
                gid = self._register_new(cam_tid, feat, frame_id, bbox,
                                         cam_idx=cam_idx)
                track.t_global_id = gid
                print(
                    f"[REGISTRY] New entry cam{cam_idx}: "
                    f"cam_tid={cam_tid} → gid={gid} "
                    f"(closest_dist={best_dist:.3f})"
                )

        if any(t.t_global_id != 0 for t in tracker.tracked_stracks):
            self._rebuild_faiss_index()

    # ── Hungarian helpers ──────────────────────────────────────────────────────

    def _can_link(self, entry: GalleryEntry, cam_id: int, frame_id: int) -> bool:
        """Return True only when it is safe to re-link a new tracklet to entry.

        Guards:
          1. Hard block if entry is still active (has a live track_id).
          2. Temporal gate: entry must have been inactive for min_gap_frames.
          3. Intra-camera suppression: if the query cam and gallery entry share the
             same camera, require a longer gap before re-ID is allowed.
        """
        if entry.active_tid is not None:
            return False
        gap = frame_id - entry.last_frame
        if gap < self.min_gap_frames:
            return False
        if entry.cam_id == cam_id and gap < self.intra_cam_gap:
            return False
        return True

    @staticmethod
    def _compute_exit_side(
        bbox: np.ndarray,
        frame_w: int = 1920,
        frame_h: int = 1080,
        edge_margin: int = 50,
    ) -> Optional[str]:
        """Determine which frame edge a bbox is closest to / exiting from.

        Args:
            bbox: [left, top, width, height] in pixel coordinates (tlwh).
        Returns:
            'left' | 'right' | 'top' | 'bottom' | None
            Left/right take priority (horizontal alley between cameras).
        """
        left, top, w, h = bbox
        right  = left + w
        bottom = top  + h

        near_left   = left   <= edge_margin
        near_right  = right  >= frame_w - edge_margin
        near_top    = top    <= edge_margin
        near_bottom = bottom >= frame_h - edge_margin

        # Horizontal priority (cameras are side-by-side along x-axis)
        if near_left and near_right:
            # Ambiguous wide bbox — pick the closest edge
            return 'left' if left < (frame_w - right) else 'right'
        if near_right:
            return 'right'
        if near_left:
            return 'left'
        if near_bottom:
            return 'bottom'
        if near_top:
            return 'top'
        return None

    def _spatial_cost(
        self,
        entry: GalleryEntry,
        to_cam_idx: int,
        new_track_bbox: np.ndarray,
    ) -> float:
        """Spatial cost based on camera topology and detected exit/entry sides.

        Returns a value in [0, 1]:
          0.0 → expected cross-camera transition
          0.5 → no topology information (neutral)
          1.0 → impossible / same camera
        """
        if entry.cam_id == to_cam_idx:
            return 1.0   # should only be called for cross-camera candidates

        transitions = CAMERA_TOPOLOGY.get((entry.cam_id, entry.exit_side))
        if transitions is None:
            # exit_side unknown — return neutral cost so appearance+time decide
            return 0.5

        # Find the tuple matching to_cam_idx
        match = next((t for t in transitions if t[0] == to_cam_idx), None)
        if match is None:
            return 0.9   # camera not in topology → unlikely transition

        _, expected_entry_side, base_cost = match

        # Refine with where in the new camera the track actually appears
        bbox_cx = new_track_bbox[0] + new_track_bbox[2] / 2.0
        entry_x_frac = float(bbox_cx) / max(self.frame_w, 1)

        if expected_entry_side == 'left':
            side_penalty = entry_x_frac           # low x → low penalty
        elif expected_entry_side == 'right':
            side_penalty = 1.0 - entry_x_frac     # high x → low penalty
        else:
            side_penalty = 0.5                     # vertical entries — neutral

        return 0.5 * base_cost + 0.5 * side_penalty

    def _time_cost(self, entry: GalleryEntry, frame_id: int) -> float:
        """Time-based cost for cross-camera matching.

        Returns:
          1.0  if gap < MIN_CROSSING_FRAMES  (physically impossible — too early)
          0.0  if MIN ≤ gap ≤ MAX            (ideal crossing window)
          0..1 if gap > MAX                  (exponential decay toward 1.0)
        """
        gap = frame_id - entry.last_frame
        if gap < MIN_CROSSING_FRAMES:
            return 1.0
        if gap <= MAX_CROSSING_FRAMES:
            return 0.0
        excess = gap - MAX_CROSSING_FRAMES
        return min(1.0 - math.exp(-excess / MAX_CROSSING_FRAMES), 1.0)

    def _compute_cross_camera_cost(
        self,
        unlinked_tracks: list,
        cross_entries: list[GalleryEntry],
        cam_idx: int,
        frame_id: int,
    ) -> np.ndarray:
        """Build combined [n_tracks × n_entries] cost matrix for cross-camera matching.

        Each cell = w_a * appearance + w_s * spatial + w_t * time.
        Cells that are physically impossible (time=1.0) or have an active tid
        are set to INF so the Hungarian solver never selects them.
        """
        INF = 1.0
        n_t = len(unlinked_tracks)
        n_e = len(cross_entries)

        # ── Appearance sub-matrix ────────────────────────────────────────────
        track_feats = np.stack([
            t.smooth_feat.astype(np.float32) / max(np.linalg.norm(t.smooth_feat), 1e-9)
            for t in unlinked_tracks
        ], axis=0)   # [n_t, emb_dim]

        entry_centroids = np.stack([
            e.centroid.astype(np.float32)
            for e in cross_entries
        ], axis=0)   # [n_e, emb_dim]

        appearance_mat = cdist(track_feats, entry_centroids, metric='cosine').astype(np.float32)

        # ── Spatial sub-matrix ───────────────────────────────────────────────
        spatial_mat = np.empty((n_t, n_e), dtype=np.float32)
        for ti, track in enumerate(unlinked_tracks):
            for ei, entry in enumerate(cross_entries):
                spatial_mat[ti, ei] = self._spatial_cost(entry, cam_idx, track.tlwh)

        # ── Time sub-matrix (per-entry, broadcast across tracks) ─────────────
        time_vec = np.array(
            [self._time_cost(e, frame_id) for e in cross_entries],
            dtype=np.float32,
        )  # [n_e]
        time_mat = np.tile(time_vec, (n_t, 1))   # [n_t, n_e]

        # ── Combined cost ────────────────────────────────────────────────────
        combined = (self.w_appearance * appearance_mat
                    + self.w_spatial  * spatial_mat
                    + self.w_time     * time_mat)

        # Hard block: active entries or impossible time windows → INF
        for ei, entry in enumerate(cross_entries):
            if entry.active_tid is not None:
                combined[:, ei] = INF
            elif time_vec[ei] >= 1.0:
                # time_cost==1.0 means physically impossible; the combined score
                # would be inflated but set it to hard INF for clarity
                combined[:, ei] = INF

        return combined

    def step_source_hungarian(self, tracker, cam_idx: int, frame_id: int):
        """Two-pass Hungarian cross-camera re-association.

        Pass 1 — Intra-camera: appearance-only Hungarian assignment
                 (same-camera re-entry after occlusion/brief loss)
        Pass 2 — Cross-camera: combined cost matrix
                 (appearance + spatial topology prior + temporal gate)
        Pass 3 — Register remaining unlinked tracks as new gallery entries.

        Drop-in replacement for step_source(); call with identical arguments.
        """
        INF = 1.0
        cam_base = cam_idx * 100_000

        # ── 1. Deactivate tracks that left this camera ───────────────────────
        current_cam_tids = {cam_base + t.track_id for t in tracker.tracked_stracks}
        cam_linked       = {t for t in self._tid_to_gid
                            if cam_base <= t < cam_base + 100_000}
        for gone_tid in cam_linked - current_cam_tids:
            # Record exit side before deactivation so it's preserved in the entry
            gid   = self._tid_to_gid.get(gone_tid)
            entry = self._gid_to_entry.get(gid) if gid is not None else None
            if entry is not None:
                entry.exit_side = self._compute_exit_side(
                    entry.last_bbox, self.frame_w, self.frame_h, self.edge_margin
                )
                entry.cam_id = cam_idx
            self.deactivate_track(gone_tid)

        # ── 2. Rebuild FAISS with fresh centroids BEFORE any queries ─────────
        self._rebuild_faiss_index()

        # ── 3. Update already-linked tracks ──────────────────────────────────
        for track in tracker.tracked_stracks:
            cam_tid = cam_base + track.track_id
            feat    = track.smooth_feat
            bbox    = track.tlwh

            if track.t_global_id != 0:
                entry = self._get_entry_by_gid(track.t_global_id)
                if entry is not None:
                    if entry.active_tid != cam_tid:
                        if entry.active_tid is not None and entry.active_tid in self._tid_to_gid:
                            del self._tid_to_gid[entry.active_tid]
                        entry.active_tid = cam_tid
                        self._tid_to_gid[cam_tid] = track.t_global_id
                    if feat is not None:
                        entry.add_embedding(feat, frame_id, bbox, cam_id=cam_idx)
                continue

        # ── 4. Collect unlinked tracks that are old enough ───────────────────
        unlinked = [
            t for t in tracker.tracked_stracks
            if t.t_global_id == 0 and t.tracklet_len >= self.min_frames
        ]
        if not unlinked:
            return

        matched_indices: set[int] = set()

        # ── 5. Pass 1: Intra-camera Hungarian ────────────────────────────────
        intra_entries = [e for e in self._entries
                         if e.cam_id == cam_idx and e.active_tid is None]

        if intra_entries:
            n_t = len(unlinked)
            n_e = len(intra_entries)
            cost_intra = np.full((n_t, n_e), INF, dtype=np.float32)

            for ti, track in enumerate(unlinked):
                feat = track.smooth_feat
                if feat is None:
                    continue
                for ei, entry in enumerate(intra_entries):
                    if not self._can_link(entry, cam_idx, frame_id):
                        continue
                    cost_intra[ti, ei] = entry.cosine_distance(feat)

            row_ind, col_ind = linear_sum_assignment(cost_intra)
            for ri, ci in zip(row_ind, col_ind):
                if ci >= n_e or cost_intra[ri, ci] >= self.match_threshold:
                    continue
                track = unlinked[ri]
                entry = intra_entries[ci]
                cam_tid = cam_base + track.track_id
                if not self._link_existing(entry, cam_tid):
                    continue
                feat = track.smooth_feat
                if feat is not None:
                    entry.add_embedding(feat, frame_id, track.tlwh, cam_id=cam_idx)
                track.t_global_id = entry.global_id
                matched_indices.add(ri)
                print(
                    f"[REGISTRY] Intra-cam re-entry cam{cam_idx}: "
                    f"cam_tid={cam_tid} → gid={entry.global_id} "
                    f"(cos_dist={cost_intra[ri, ci]:.3f})"
                )

        # ── 6. Pass 2: Cross-camera Hungarian ────────────────────────────────
        still_unlinked = [t for i, t in enumerate(unlinked) if i not in matched_indices]
        still_indices  = [i for i in range(len(unlinked))  if i not in matched_indices]

        cross_entries = [e for e in self._entries
                         if e.cam_id != cam_idx and e.active_tid is None]

        if still_unlinked and cross_entries:
            combined = self._compute_cross_camera_cost(
                still_unlinked, cross_entries, cam_idx, frame_id
            )
            # Apply _can_link gate on top
            for ei, entry in enumerate(cross_entries):
                if not self._can_link(entry, cam_idx, frame_id):
                    combined[:, ei] = INF

            row_ind, col_ind = linear_sum_assignment(combined)
            for ri, ci in zip(row_ind, col_ind):
                if ci >= len(cross_entries) or combined[ri, ci] >= self.cross_match_threshold:
                    continue
                track = still_unlinked[ri]
                entry = cross_entries[ci]
                cam_tid = cam_base + track.track_id
                if not self._link_existing(entry, cam_tid):
                    continue
                feat = track.smooth_feat
                if feat is not None:
                    entry.add_embedding(feat, frame_id, track.tlwh, cam_id=cam_idx)
                track.t_global_id = entry.global_id
                matched_indices.add(still_indices[ri])
                print(
                    f"[REGISTRY] Cross-cam re-entry cam{entry.cam_id}→cam{cam_idx}: "
                    f"cam_tid={cam_tid} → gid={entry.global_id} "
                    f"(combined_cost={combined[ri, ci]:.3f}, "
                    f"exit_side={entry.exit_side})"
                )

        # ── 7. Register remaining unlinked tracks as new identities ──────────
        for ti, track in enumerate(unlinked):
            if ti in matched_indices:
                continue
            cam_tid = cam_base + track.track_id
            feat    = track.smooth_feat
            if feat is None:
                feat = np.zeros(self._emb_dim, dtype=np.float32)
            gid = self._register_new(cam_tid, feat, frame_id, track.tlwh, cam_idx=cam_idx)
            track.t_global_id = gid
            print(
                f"[REGISTRY] New entry cam{cam_idx}: "
                f"cam_tid={cam_tid} → gid={gid}"
            )

        # ── 8. Rebuild FAISS with updated centroids ───────────────────────────
        if matched_indices or any(t.t_global_id != 0 for t in tracker.tracked_stracks):
            self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        if not self._entries:
            self._index.reset()
            self._faiss_pos_to_gid = []
            self._gid_to_pos       = {}
            return

        centroids = np.stack(
            [e.centroid.astype(np.float32) for e in self._entries], axis=0
        )
        centroids = np.ascontiguousarray(centroids)

        self._index.reset()
        self._index.add(centroids)
        self._faiss_pos_to_gid = [e.global_id for e in self._entries]
        self._gid_to_pos       = {e.global_id: i for i, e in enumerate(self._entries)}

    def _get_entry_by_gid(self, gid: int) -> Optional[GalleryEntry]:
        return self._gid_to_entry.get(gid)

    def get_all_entries(self) -> list[GalleryEntry]:
        return list(self._entries)

    def size(self) -> int:
        return len(self._entries)

    def __repr__(self):
        active = sum(1 for e in self._entries if e.active_tid is not None)
        return (f"GlobalRegistry("
                f"total={len(self._entries)}, "
                f"active={active}, "
                f"threshold={self.match_threshold})")