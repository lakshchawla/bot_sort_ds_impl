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


# ──────────────────────────────────────────────────────────────────────────────
# GalleryEntry
# ──────────────────────────────────────────────────────────────────────────────

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
        self.active_tid = track_id        # BoTSORT track_id currently linked
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()

        self.embeddings: deque = deque(maxlen=max_emb)
        self.embeddings.append(feat)
        self.centroid = feat.copy()       # starts as single embedding

    # ── internal ──────────────────────────────────────────────────────────────

    def _recompute_centroid(self):
        c = np.mean(self.embeddings, axis=0).astype(np.float32)
        n = np.linalg.norm(c)
        self.centroid = c / n if n > 1e-9 else c

    # ── public ────────────────────────────────────────────────────────────────

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray):
        """Called every frame the person is actively tracked."""
        self.embeddings.append(feat)
        self._recompute_centroid()
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()

    def similarity(self, feat: np.ndarray) -> float:
        """Cosine similarity against this entry's centroid. Range [−1, 1]."""
        n = np.linalg.norm(feat)
        if n < 1e-9 or np.linalg.norm(self.centroid) < 1e-9:
            return 0.0
        return float(np.dot(self.centroid, feat / n))

    def cosine_distance(self, feat: np.ndarray) -> float:
        """1 − similarity. Range [0, 2], lower = more similar."""
        return 1.0 - self.similarity(feat)

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, "
                f"tid={self.active_tid}, "
                f"n_emb={len(self.embeddings)}, "
                f"last_frame={self.last_frame})")


# ──────────────────────────────────────────────────────────────────────────────
# GlobalRegistry
# ──────────────────────────────────────────────────────────────────────────────

class GlobalRegistry:
    """
    Usage pattern (in test.py / probe):

        registry = GlobalRegistry(match_threshold=0.35)

        # after tracker.update():
        registry.step(tracker, frame_id=cur_frame)

        # read global_id off each active track:
        for t in tracker.tracked_stracks:
            print(t.t_global_id)
    """

    def __init__(
        self,
        match_threshold: float = 0.35,   # cosine distance below which → same person
        min_frames:      int   = 5,       # track must be confirmed for N frames before
                                          # registering, avoids polluting gallery with ghosts
        max_emb:         int   = 50,      # rolling embedding buffer per entry
    ):
        self.match_threshold = match_threshold
        self.min_frames      = min_frames
        self.max_emb         = max_emb

        self._entries:       list[GalleryEntry] = []
        self._tid_to_gid:    dict[int, int]     = {}   # BoTSORT track_id → global_id
        self._global_id_ctr: int                = 0

    # ── ID minting ────────────────────────────────────────────────────────────

    def _new_global_id(self) -> int:
        self._global_id_ctr += 1
        return self._global_id_ctr

    # ── Core query ────────────────────────────────────────────────────────────

    def query(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        """
        Find the closest gallery entry by cosine distance.

        Returns:
            (best_entry, best_distance)
            best_entry is None if gallery is empty or no entry is close enough.
        """
        if not self._entries or feat is None:
            return None, 1.0

        # Build centroid matrix  (N, D)
        centroids = np.stack([e.centroid for e in self._entries], axis=0)
        feat_norm = feat / (np.linalg.norm(feat) + 1e-9)

        cosine_sims  = centroids @ feat_norm          # (N,)
        cosine_dists = 1.0 - cosine_sims              # (N,)

        best_idx  = int(np.argmin(cosine_dists))
        best_dist = float(cosine_dists[best_idx])

        if best_dist < self.match_threshold:
            return self._entries[best_idx], best_dist
        return None, best_dist

    # ── Linear assignment over a batch of detections ─────────────────────────

    def query_batch(
        self,
        feats: list[np.ndarray],
    ) -> list[tuple[Optional[GalleryEntry], float]]:
        """
        Run gallery matching for a batch of embedding vectors at once.
        Returns list of (best_entry | None, best_distance) per feat.

        This is the FAISS swap-in point later:
            replace the np.stack + matmul with a faiss index search.
        """
        if not self._entries or not feats:
            return [(None, 1.0)] * len(feats)

        valid_feats = []
        valid_idx   = []
        for i, f in enumerate(feats):
            if f is not None and np.linalg.norm(f) > 1e-9:
                valid_feats.append(f / np.linalg.norm(f))
                valid_idx.append(i)

        results: list[tuple[Optional[GalleryEntry], float]] = [
            (None, 1.0)
        ] * len(feats)

        if not valid_feats:
            return results

        # (N_gallery, D)
        centroids = np.stack([e.centroid for e in self._entries], axis=0)
        # (N_query, D)
        query_mat = np.stack(valid_feats, axis=0)

        # (N_gallery, N_query)
        cos_sims  = centroids @ query_mat.T
        cos_dists = 1.0 - cos_sims   # lower = better

        for qi, orig_i in enumerate(valid_idx):
            col       = cos_dists[:, qi]
            best_idx  = int(np.argmin(col))
            best_dist = float(col[best_idx])
            if best_dist < self.match_threshold:
                results[orig_i] = (self._entries[best_idx], best_dist)
            else:
                results[orig_i] = (None, best_dist)

        return results

    # ── Register / update / deactivate ───────────────────────────────────────

    def _register_new(
        self,
        track_id: int,
        feat:     np.ndarray,
        frame_id: int,
        bbox:     np.ndarray,
    ) -> int:
        """Mint a new global_id and create a gallery entry."""
        gid = self._new_global_id()
        entry = GalleryEntry(
            global_id=gid,
            feat=feat,
            track_id=track_id,
            frame_id=frame_id,
            bbox=bbox,
            max_emb=self.max_emb,
        )
        self._entries.append(entry)
        self._tid_to_gid[track_id] = gid
        return gid

    def _link_existing(self, entry: GalleryEntry, track_id: int):
        """Link an existing gallery entry to a (new) BoTSORT track_id."""
        # Deactivate any previous track that was linked to this entry
        old_tid = entry.active_tid
        if old_tid is not None and old_tid in self._tid_to_gid:
            del self._tid_to_gid[old_tid]
        entry.active_tid = track_id
        self._tid_to_gid[track_id] = entry.global_id

    def deactivate_track(self, track_id: int):
        """
        Call when a BoTSORT track is permanently removed.
        The gallery entry is kept; only the active_tid link is cleared.
        """
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        for e in self._entries:
            if e.global_id == gid:
                e.active_tid = None
                break

    # ── Main step — call once per frame after tracker.update() ───────────────

    def step(self, tracker, frame_id: int):
        """
        Process all active BoTSORT tracks for this frame.

        For each track in tracker.tracked_stracks:
          - If track already has a global_id → just update the gallery entry.
          - If track is new (t_global_id == 0):
              · Wait until tracklet_len >= min_frames (embedding is stable).
              · Query gallery.
              · Match found  → reuse old global_id, link entry to this track.
              · No match     → register new entry, mint new global_id.
          - Assign t_global_id on the STrack object.

        For removed tracks (tracks no longer in tracked_stracks):
          - Deactivate their registry link.
        """
        current_tids = {t.track_id for t in tracker.tracked_stracks}

        # ── deactivate disappeared tracks ────────────────────────────────────
        linked_tids = set(self._tid_to_gid.keys())
        for gone_tid in linked_tids - current_tids:
            self.deactivate_track(gone_tid)

        # ── process active tracks ─────────────────────────────────────────────
        for track in tracker.tracked_stracks:
            tid  = track.track_id
            feat = track.smooth_feat   # L2-normalised EMA feature from STrack
            bbox = track.tlwh

            # ── already has a global_id: just update gallery centroid ─────────
            if track.t_global_id != 0:
                gid = track.t_global_id
                entry = self._get_entry_by_gid(gid)
                if entry is not None and feat is not None:
                    entry.add_embedding(feat, frame_id, bbox)
                continue

            # ── new track: wait for embedding to stabilise ────────────────────
            if track.tracklet_len < self.min_frames:
                continue   # not ready yet — no global_id assigned

            if feat is None:
                # No embedding available (SGIE not running) — assign new ID
                gid = self._register_new(tid, np.zeros(1), frame_id, bbox)
                track.t_global_id = gid
                continue

            # ── gallery query ─────────────────────────────────────────────────
            best_entry, best_dist = self.query(feat)

            if best_entry is not None:
                # ── Re-entry: person seen before ─────────────────────────────
                print(
                    f"[REGISTRY] Re-entry detected: "
                    f"track_id={tid} → global_id={best_entry.global_id} "
                    f"(cos_dist={best_dist:.3f})"
                )
                self._link_existing(best_entry, tid)
                best_entry.add_embedding(feat, frame_id, bbox)
                track.t_global_id = best_entry.global_id
            else:
                # ── New person: register ──────────────────────────────────────
                gid = self._register_new(tid, feat, frame_id, bbox)
                track.t_global_id = gid
                print(
                    f"[REGISTRY] New entry: "
                    f"track_id={tid} → global_id={gid} "
                    f"(closest_dist={best_dist:.3f})"
                )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_entry_by_gid(self, gid: int) -> Optional[GalleryEntry]:
        for e in self._entries:
            if e.global_id == gid:
                return e
        return None

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