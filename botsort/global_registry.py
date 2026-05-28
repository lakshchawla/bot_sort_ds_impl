from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional
import faiss



class GalleryEntry:
    def __init__(
        self,
        global_id:  int,
        feat:       np.ndarray,
        track_id:   int,
        frame_id:   int,
        bbox:       np.ndarray,
        max_emb:    int = 50,
    ):
        self.global_id  = global_id
        self.active_tid = track_id        
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()

        self.embeddings: deque = deque(maxlen=max_emb)
        self.embeddings.append(feat)
        self.centroid = feat.copy()       

    def _recompute_centroid(self):
        c = np.mean(self.embeddings, axis=0).astype(np.float32)
        n = np.linalg.norm(c)
        self.centroid = c / n if n > 1e-9 else c

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray):
        self.embeddings.append(feat)
        self._recompute_centroid()
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()

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
                f"n_emb={len(self.embeddings)}, "
                f"last_frame={self.last_frame})")

class GlobalRegistry:
    def __init__(
        self,
        match_threshold: float = 0.35,
        min_frames: int = 5,
        max_emb: int = 50,
        emb_dim: int = 256,
    ):
        self.match_threshold = match_threshold
        self.min_frames      = min_frames
        self.max_emb         = max_emb

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
            frame_id=frame_id, bbox=bbox, max_emb=self.max_emb,
        )
        self._entries.append(entry)
        self._gid_to_entry[gid] = entry
        self._tid_to_gid[track_id] = gid

        vec = entry.centroid.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)
        self._index.add(vec)                          
        self._faiss_pos_to_gid.append(gid)            

        return gid


    def _link_existing(self, entry: GalleryEntry, track_id: int):
        old_tid = entry.active_tid
        if old_tid is not None and old_tid in self._tid_to_gid:
            del self._tid_to_gid[old_tid]
        entry.active_tid = track_id
        self._tid_to_gid[track_id] = entry.global_id

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

    
    def step_reid(self, trackers: list, frame_id: int):
        combined_lost_stracks = []
        combined_tracked_stracks = []
        touching_edge = True
        
        for tracker in trackers: 
            combined_lost_stracks.extend(tracker.lost_stracks)
            combined_tracked_stracks.extend(tracker.tracked_stracks)

        #    current_tids = {t.track_id for t in tracker.tracked_stracks} # in current frame
        #    linked_tids = set(self._tid_to_gid.keys()) # owning a global id by the wrapper
        #    unlinked_tids = linked_tids - current_tids # not owning one, ()

        #    lost stracks edge filter
        #    directional_exit: occlusion specific check.
        #    query matching time_complexity

        edge_lost = self._get_boundary_tracks(combined_lost_stracks) 
        

    @staticmethod
    def _get_boundary_tracks(tracks, frame_w = 1920, frame_h = 1080, margin=20):
      """
      Returns tracks whose bounding box is within `margin` pixels of any frame edge.
      tlwh = [top-left-x, top-left-y, width, height]
      """
      boundary = []
      for track in tracks:
          x, y, w, h = track.tlwh
          x2, y2 = x + w, y + h

          touching = (
              x  <= margin        or   # left edge
              y  <= margin        or   # top edge
              x2 >= frame_w - margin or  # right edge
              y2 >= frame_h - margin    # bottom edge
          )
          if touching:
              boundary.append(track)
      return boundary


    
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
        cam_base = (cam_idx + 1) * 1_000

        # Deactivate tracks that left this camera's tracked_stracks 
        current_cam_tids = {cam_base + t.track_id for t in tracker.tracked_stracks}
        cam_linked       = {t for t in self._tid_to_gid
                            if cam_base <= t < cam_base + 100_000}
        for gone_tid in cam_linked - current_cam_tids:
            self.deactivate_track(gone_tid)

        # Assign / update global IDs 
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

    def _rebuild_faiss_index(self):
        if not self._entries:
            self._index.reset()
            self._faiss_pos_to_gid = []
            return

        centroids = np.stack(
            [e.centroid.astype(np.float32) for e in self._entries], axis=0
        )   
        centroids = np.ascontiguousarray(centroids)

        self._index.reset()                        
        self._index.add(centroids)                 
        self._faiss_pos_to_gid = [e.global_id for e in self._entries]

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