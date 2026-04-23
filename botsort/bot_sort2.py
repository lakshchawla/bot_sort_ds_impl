"""
botsort.py  –  simplified for DeepStream with no ReID embeddings.

Since embed=NO (SGIE not producing vectors yet), we use pure IOU matching
for all associations.  This is identical to SORT and is extremely stable.
Once your SGIE produces embeddings, flip with_reid=True and the embedding
branch re-activates automatically.
"""

import numpy as np
import math

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


class ID_Assigner:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def next_id(self):
        self.cur_id += 1
        return self.cur_id


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.last_known_mean = None
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated  = False
        self.score         = score
        self.tracklet_len  = 0
        self.smooth_feat   = None
        self.curr_feat     = None
        self.alpha         = 0.9
        self.matched_det_idx = -1

        if feat is not None:
            self.update_features(feat)

    def activate(self, kalman_filter, frame_id, id_assigner=None):
        self.kalman_filter = kalman_filter
        self.track_id = id_assigner.next_id() if id_assigner else self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )
        self.last_known_mean = self.mean.copy()   # ← ADD
        self.tracklet_len  = 0
        self.state         = TrackState.Tracked
        self.is_activated  = frame_id == 1
        self.frame_id      = frame_id
        self.start_frame   = frame_id

    def update_features(self, feat):
        feat = feat.astype(np.float32)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        norm = np.linalg.norm(self.smooth_feat)
        if norm > 0:
            self.smooth_feat /= norm

    def predict(self):
        if self.state != TrackState.Tracked:
            # Lost track — freeze at last known position, zero all velocity
            # so the centroid used for re-association is exactly where
            # the person was last seen, not where Kalman drifted to
            self.mean = self.last_known_mean.copy()
            self.mean[4] = 0   # vx
            self.mean[5] = 0   # vy
            self.mean[6] = 0   # vw
            self.mean[7] = 0   # vh
            # Still run predict so covariance grows (uncertainty increases)
            # but the position stays anchored
            _, self.covariance = self.kalman_filter.predict(
                self.mean, self.covariance
            )
        else:
            self.mean, self.covariance = self.kalman_filter.predict(
                self.mean, self.covariance
            )   

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_cov  = np.asarray([st.covariance   for st in stracks])

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                # Freeze position at last known, zero velocity
                multi_mean[i] = st.last_known_mean.copy()
                multi_mean[i][4] = 0
                multi_mean[i][5] = 0
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0

        multi_mean, multi_cov = STrack.shared_kalman.multi_predict(
            multi_mean, multi_cov
        )

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_cov)):
            if stracks[i].state != TrackState.Tracked:
                # Restore frozen position — don't let predict() move it
                mean[:4] = stracks[i].last_known_mean[:4]
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def mark_lost(self):
        self.state = TrackState.Lost
        # Snap velocity to zero immediately so the first predict() call
        # after this doesn't move the centroid at all
        if self.mean is not None:
            self.mean[4] = 0   # vx
            self.mean[5] = 0   # vy
            self.mean[6] = 0   # vw
            self.mean[7] = 0   # vh

    def re_activate(self, new_track, frame_id, new_id=False, id_assigner=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        self.last_known_mean = self.mean.copy()   # ← ADD
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len  = 0
        self.state         = TrackState.Tracked
        self.is_activated  = True
        self.frame_id      = frame_id
        self.score         = new_track.score
        if new_id:
            self.track_id = id_assigner.next_id() if id_assigner else self.next_id()

    def update(self, new_track, frame_id):
        self.last_known_mean = self.mean.copy() 
        self.frame_id      = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state         = TrackState.Tracked
        self.is_activated  = True
        self.score         = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"
    
class ProbationTrack:
    """
    Holds a new unmatched detection for `confirm_frames` frames.
    Accumulates an averaged embedding before attempting gallery match.
    """
    def __init__(self, strack, frame_id, confirm_frames=60):
        self.strack         = strack          # underlying STrack (no ID yet)
        self.start_frame    = frame_id
        self.confirm_frames = confirm_frames
        self.feat_buffer    = []              # list of raw embedding arrays
        if strack.curr_feat is not None:
            self.feat_buffer.append(strack.curr_feat.copy())

    def update(self, strack, frame_id):
        """Feed a new matched detection into this probation slot."""
        self.strack = strack
        if strack.curr_feat is not None:
            self.feat_buffer.append(strack.curr_feat.copy())

    @property
    def age(self):
        return len(self.feat_buffer)

    @property
    def mean_feat(self):
        if not self.feat_buffer:
            return None
        feat = np.mean(self.feat_buffer, axis=0).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def is_ready(self, frame_id):
        return (frame_id - self.start_frame) >= self.confirm_frames


class BoTSORT:
    def __init__(
        self,
        track_high_thresh  = 0.5,   # lowered – DS tracker_conf often 0.5-0.8
        track_low_thresh   = 0.1,
        new_track_thresh   = 0.4,
        track_buffer       = 15000,
        match_thresh       = 0.7,   # IOU threshold for first association
        with_reid          = True, # flip True once SGIE produces embeddings
        appearance_thresh  = 0.35,
        frame_rate         = 30,
        map_len            = None,
    ):
        self.tracked_stracks  = []
        self.lost_stracks     = []
        self.removed_stracks  = []
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh

        self.buffer_size   = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = 30000
        self.kalman_filter = KalmanFilter()

        self.match_thresh      = match_thresh
        self.with_reid         = with_reid
        self.appearance_thresh = appearance_thresh
        self.max_len = map_len if map_len else math.sqrt(1920**2 + 1080**2)
        self.id_assigner = ID_Assigner()

        self.confirm_frames  = int(frame_rate * 1.0)  # 1 second probation window
        self.probation_pool  = []   # list[ProbationTrack] – new tracklets being evaluated

    def update(self, output_results):
        """
        Parameters
        ----------
        output_results : List[dict]
            local_track_id  – DS object_id (ignored, unstable)
            bbox            – [x1,y1,x2,y2] float32 tlbr
            det_confidence  – float 0-1
            reid_vector     – np.ndarray or None

        Returns
        -------
        List[STrack] with .track_id and .matched_det_idx set.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks    = []
        lost_stracks      = []
        removed_stracks   = []

        have_reid = False

        if output_results:
            scores   = np.array([d["det_confidence"] for d in output_results], dtype=np.float32)
            bboxes   = np.array([d["bbox"]           for d in output_results], dtype=np.float32)

            have_reid = any(d["reid_vector"] is not None for d in output_results)
            if have_reid:
                feat_dim = next(len(d["reid_vector"]) for d in output_results
                                if d["reid_vector"] is not None)
                features = np.array(
                    [d["reid_vector"] if d["reid_vector"] is not None
                     else np.zeros(feat_dim, dtype=np.float32)
                     for d in output_results], dtype=np.float32
                )
            else:
                features = None

            valid     = scores > self.track_low_thresh
            valid_idx = np.where(valid)[0]
            bboxes_v  = bboxes[valid]
            scores_v  = scores[valid]
            feats_v   = features[valid] if features is not None else None

            high_mask   = scores_v > self.track_high_thresh
            high_orig   = valid_idx[high_mask]
            low_orig    = valid_idx[~high_mask]

            dets_high   = bboxes_v[high_mask]
            scores_high = scores_v[high_mask]
            feats_high  = feats_v[high_mask] if feats_v is not None else None

            dets_low    = bboxes_v[~high_mask]
            scores_low  = scores_v[~high_mask]
        else:
            have_reid = False
            dets_high = scores_high = np.empty((0,))
            high_orig = low_orig = np.empty((0,), dtype=int)
            dets_low  = scores_low = np.empty((0,))
            feats_high = None

        detections = []
        for i, (tlbr, s, orig_i) in enumerate(zip(dets_high, scores_high, high_orig)):
            feat = feats_high[i] if feats_high is not None else None
            t = STrack(STrack.tlbr_to_tlwh(tlbr), float(s), feat)
            t.matched_det_idx = int(orig_i)
            detections.append(t)

        unconfirmed     = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            track.matched_det_idx = -1
            (tracked_stracks if track.is_activated else unconfirmed).append(track)
        for track in self.lost_stracks:
            track.matched_det_idx = -1

        # ── Step 2: first association – active tracked vs high-conf dets ────────
        # Only tracked (not lost) tracks here — lost tracks get their own
        # dedicated step with a relaxed threshold below.
        STrack.multi_predict(tracked_stracks)
        STrack.multi_predict(self.lost_stracks)

        iou_dists = matching.iou_distance(tracked_stracks, detections).astype(np.float64)

        print(len(tracked_stracks) > 0)

        """
        check err this part
        """
        if self.with_reid and have_reid and len(tracked_stracks) > 0 and len(detections) > 0:
            print(self.with_reid and have_reid and len(tracked_stracks) > 0 and len(detections) > 0)
            emb_dists = matching.embedding_distance(tracked_stracks, detections) / 2
            dists     = 0 * iou_dists + 2 * emb_dists
            dists[emb_dists > self.appearance_thresh] = 1.0
        else:
            dists = iou_dists

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det   = detections[idet]
            track.matched_det_idx = det.matched_det_idx
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # ── Step 2b: re-associate LOST tracks against still-unmatched dets ────
        # Lost tracks have drifted Kalman state so IOU with the real bbox is
        # often 0. Use centroid distance instead — much more forgiving when
        # the person reappears near where they disappeared.
        unmatched_dets_for_lost = [detections[i] for i in u_detection]

        if self.lost_stracks and unmatched_dets_for_lost:
            # Compute centroid distance: cx/cy of Kalman-predicted bbox vs det bbox
            lost_centroids = np.array(
                [t.tlwh_to_xywh(t.tlwh)[:2] for t in self.lost_stracks],
                dtype=np.float32
            )  # (L, 2)
            det_centroids = np.array(
                [t.tlwh_to_xywh(t.tlwh)[:2] for t in unmatched_dets_for_lost],
                dtype=np.float32
            )  # (D, 2)

            # Pairwise L2 distance, normalised by frame diagonal
            diff        = lost_centroids[:, None, :] - det_centroids[None, :, :]  # (L,D,2)
            cdist       = np.linalg.norm(diff, axis=2) / self.max_len             # (L,D)

            # Also mix in embedding distance if available
            if self.with_reid and have_reid:
                emb_lost = matching.embedding_distance(
                    self.lost_stracks, unmatched_dets_for_lost
                ) / 2
                cdist = 0.4 * cdist + 0.6 * emb_lost
                cdist[emb_lost > self.appearance_thresh] = 1.0

            # Relaxed threshold: 0.3 normalised ≈ 660px on 1080p – wide enough
            # to handle re-entries after a few seconds of occlusion
            matches_lost, u_lost, u_det_after_lost = matching.linear_assignment(
                cdist.astype(np.float64), thresh=0.3
            )

            for ilost, idet in matches_lost:
                track = self.lost_stracks[ilost]
                det   = unmatched_dets_for_lost[idet]
                track.matched_det_idx = det.matched_det_idx
                track.re_activate(det, self.frame_id, new_id=False,
                                   id_assigner=self.id_assigner)
                refind_stracks.append(track)

            # Remaining unmatched after lost re-association
            u_detection = [u_detection[i] for i in u_det_after_lost]
            # Tracks still lost after this step
            newly_lost_ids = {self.lost_stracks[i].track_id for i in u_lost}
        else:
            newly_lost_ids = {t.track_id for t in self.lost_stracks}

        # Tracks from step 2 that weren't matched
        for it in u_track:
            track = tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── Step 3: second association – low-conf dets ────────────────────────
        detections_low = []
        for tlbr, s, orig_i in zip(dets_low, scores_low, low_orig):
            t = STrack(STrack.tlbr_to_tlwh(tlbr), float(s))
            t.matched_det_idx = int(orig_i)
            detections_low.append(t)

        r_tracked = [tracked_stracks[i] for i in u_track
                     if tracked_stracks[i].state == TrackState.Tracked]

        if r_tracked and detections_low:
            dists2 = matching.iou_distance(r_tracked, detections_low).astype(np.float64)
            matches2, u_track2, _ = matching.linear_assignment(dists2, thresh=0.5)
            for itracked, idet in matches2:
                track = r_tracked[itracked]
                det   = detections_low[idet]
                track.matched_det_idx = det.matched_det_idx
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            for it in u_track2:
                track = r_tracked[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for track in r_tracked:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        # ── Step 4: unconfirmed tracks ────────────────────────────────────────
        detections_remain = [detections[i] for i in u_detection]
        if unconfirmed and detections_remain:
            dists_u = matching.iou_distance(unconfirmed, detections_remain).astype(np.float64)
            matches3, u_unconfirmed, u_det2 = matching.linear_assignment(dists_u, thresh=0.7)
            for itracked, idet in matches3:
                unconfirmed[itracked].matched_det_idx = detections_remain[idet].matched_det_idx
                unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                unconfirmed[it].mark_removed()
                removed_stracks.append(unconfirmed[it])
        else:
            u_det2 = list(range(len(detections_remain)))
            for t in unconfirmed:
                t.mark_removed()
                removed_stracks.append(t)

        # ── Step 5: init new tracks ───────────────────────────────────────────
        remaining_new = []
        matched_prob_dets = set()

        if self.probation_pool and u_det2:
            prob_stracks = [p.strack for p in self.probation_pool]
            new_det_list = [detections_remain[i] for i in u_det2]

            prob_iou = matching.iou_distance(prob_stracks, new_det_list).astype(np.float64)
            m_prob, u_prob, u_new = matching.linear_assignment(prob_iou, thresh=0.7)

            for ip, id_ in m_prob:
                self.probation_pool[ip].update(new_det_list[id_], self.frame_id)
                matched_prob_dets.add(id_)

            # Probation tracks with no match this frame – keep alive (person briefly missed)
            # Drop only if unseen for > confirm_frames (they likely never stabilised)
            surviving = []
            for ip, p in enumerate(self.probation_pool):
                if ip not in [x for x, _ in m_prob]:
                    if (self.frame_id - p.start_frame) <= self.confirm_frames * 2:
                        surviving.append(p)
                    # else: silently drop, never got a stable embedding
                else:
                    surviving.append(p)
            self.probation_pool = surviving

            unmatched_new_dets = [u_det2[i] for i in u_new]
        else:
            unmatched_new_dets = list(u_det2)

        # 5b. Try to graduate ready probation tracks → match vs lost gallery
        graduated = []
        still_on_probation = []

        for p in self.probation_pool:
            if not p.is_ready(self.frame_id):
                still_on_probation.append(p)
                continue

            mean_feat = p.mean_feat
            if mean_feat is None or not self.lost_stracks:
                # No embedding or no lost tracks – assign new ID directly
                p.strack.activate(self.kalman_filter, self.frame_id,
                                id_assigner=self.id_assigner)
                activated_stracks.append(p.strack)
                graduated.append(p)
                continue

            # Compare averaged embedding against all lost track smooth_feats
            lost_feats = np.array(
                [t.smooth_feat if t.smooth_feat is not None
                else np.zeros_like(mean_feat)
                for t in self.lost_stracks],
                dtype=np.float32
            )  # (L, D)

            # Cosine distance
            dots   = lost_feats @ mean_feat                          # (L,)
            norms  = np.linalg.norm(lost_feats, axis=1) * np.linalg.norm(mean_feat)
            norms  = np.where(norms > 0, norms, 1.0)
            cos_dist = 1.0 - (dots / norms)                         # (L,) lower = better

            best_idx  = int(np.argmin(cos_dist))
            best_dist = float(cos_dist[best_idx])

            if best_dist < self.appearance_thresh:
                # Strong appearance match → re-activate the lost track
                lost_track = self.lost_stracks[best_idx]
                lost_track.matched_det_idx = p.strack.matched_det_idx
                lost_track.re_activate(p.strack, self.frame_id, new_id=False,
                                    id_assigner=self.id_assigner)
                refind_stracks.append(lost_track)
                print(f"[PROB] Graduated probation → matched lost ReID={lost_track.track_id}  cos_dist={best_dist:.3f}")
            else:
                # No gallery match – this is genuinely a new person
                p.strack.activate(self.kalman_filter, self.frame_id,
                                id_assigner=self.id_assigner)
                activated_stracks.append(p.strack)
                print(f"[PROB] Graduated probation → NEW ReID={p.strack.track_id}  best_lost_dist={best_dist:.3f}")

            graduated.append(p)

        self.probation_pool = still_on_probation

        # 5c. Add genuinely new unmatched dets to probation (don't assign ID yet)
        for inew in unmatched_new_dets:
            track = detections_remain[inew]
            if track.score < self.new_track_thresh:
                continue
            self.probation_pool.append(
                ProbationTrack(track, self.frame_id, self.confirm_frames)
            )

        # ── Step 6: expire lost tracks ────────────────────────────────────────
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── Merge ─────────────────────────────────────────────────────────────
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks    = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks    = _sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        if self.tracked_stracks and self.lost_stracks:
            self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks
            )

        return list(self.tracked_stracks)


def _joint_stracks(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def _sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def _remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb).astype(np.float64)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb