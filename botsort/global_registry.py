"""
global_registry.py
──────────────────
A persistent identity gallery that sits above BoTSORT.

Responsibilities:
  1. Maintain a list of GalleryEntry objects (one per real-world person).
  2. When a new tracklet is about to be confirmed, query the gallery first.
     - Match found (cosine dist < threshold) → reuse old global_id.
     - No match                               → mint a new global_id.
  3. Update a gallery entry's centroid every time its track is seen.
  4. Archive entries when tracks are permanently removed (for future FAISS swap-in).

Data structure (simple Python list now, FAISS-ready later):
  self._entries : List[GalleryEntry]

Each GalleryEntry stores:
  global_id     : int       – stable identity across re-entries
  centroid      : np.array  – L2-normalised mean of all collected embeddings
  embeddings    : deque     – rolling buffer of raw embeddings (max_size)
  last_frame    : int       – frame when this entry was last updated
  last_bbox     : np.array  – tlwh bbox at last sighting (for ghost init)
  active_tid    : int | None– the current BoTSORT track_id linked to this entry
                              None when person is out of frame
"""

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
        self._global_id_ctr: int                = 0

        self._emb_dim   = emb_dim        
        self._index_cpu = faiss.IndexFlatIP(emb_dim)   

        res             = faiss.StandardGpuResources()
        self._index     = faiss.index_cpu_to_gpu(res, 0, self._index_cpu)

        self._faiss_pos_to_gid: list[int] = []
        self._gid_to_entry: dict[int, GalleryEntry] = {}

    def _new_global_id(self) -> int:
        self._global_id_ctr += 1
        return self._global_id_ctr

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

    def _register_new(self, track_id, feat, frame_id, bbox):
        gid = self._new_global_id()
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


    def step(self, tracker, frame_id: int, tid_offset: int = 0, tid_range: int = 100_000):
        current_tids = {t.track_id + tid_offset for t in tracker.tracked_stracks}

        # Only consider tids that belong to this camera's range so that a
        # second call (camera 2) does not deactivate camera 1's linked tracks.
        cam_linked_tids = {t for t in self._tid_to_gid
                           if tid_offset <= t < tid_offset + tid_range}
        for gone_tid in cam_linked_tids - current_tids:
            self.deactivate_track(gone_tid)

        for track in tracker.tracked_stracks:
            tid  = track.track_id + tid_offset
            feat = track.smooth_feat
            bbox = track.tlwh


            # if global id is already assigned to a tracklet, still need to
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
                # Guard: if the matched entry is already occupied by a live
                # track, do NOT steal the global_id — that would give two
                # active tracks the same identity.  Register a fresh entry.
                if best_entry.active_tid is not None and best_entry.active_tid in self._tid_to_gid:
                    gid = self._register_new(tid, feat, frame_id, bbox)
                    track.t_global_id = gid
                    print(
                        f"[REGISTRY] Occupied entry blocked: "
                        f"track_id={tid} → new global_id={gid} "
                        f"(matched gid={best_entry.global_id} already held by tid={best_entry.active_tid})"
                    )
                else:
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