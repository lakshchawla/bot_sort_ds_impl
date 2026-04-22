import sys
import os
import math
import platform
import yaml
import ctypes

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

import numpy as np
import time
import threading
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import logging


import ctypes
import numpy as np
import pyds
from gi.repository import Gst

path_to_botsort_parent = '/home/lab314/workspace/reid/botsort-tracker'

if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

 
from botsort.bot_sort2 import BoTSORT


PERF_MODE = os.environ.get("NVDS_TEST3_PERF_MODE") == "1"

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write(f"WARNING from element {message.src.get_name()}: {err.message}\n")
        sys.stderr.write(f"Warning: {err.message}\n")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"ERROR from element {message.src.get_name()}: {err.message}\n")
        if debug:
            sys.stderr.write(f"Error details: {debug}\n")
        loop.quit()
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct and struct.get_name() == "nvmsg-stream-eos":
            stream_id = struct.get_value("stream-id")
            sys.stdout.write(f"Got EOS from stream {stream_id}\n")
    return True

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps(None)
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    sys.stdout.write(f"Decodebin child added: {name}\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    sys.stdout.write(f"{uri}\n")
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    if PERF_MODE:
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", True)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")

    if not nbin or not uri_decode_bin:
        sys.stderr.write("One element in source bin could not be created.\n")
        return None

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None

    return nbin


DIAG = True   # flip False once tracklet_len is stable and incrementing
 
_tracker = BoTSORT()
 
 
def _extract_embedding(tensor_meta):
    layer     = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    embed_len = 1
    for i in range(layer.inferDims.numDims):
        embed_len *= layer.inferDims.d[i]
    ptr = ctypes.cast(
        pyds.get_ptr(layer.buffer),
        ctypes.POINTER(ctypes.c_float)
    )
    return np.ctypeslib.as_array(ptr, shape=(embed_len,)).copy().astype(np.float32)
 
 
def reid_pad_buffer_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
 
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
 
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
 
        # ── 1. Collect detections as an ordered list ──────────────────────────
        # Index in this list == matched_det_idx returned by BoTSORT.
        # DS object_id is NOT used for matching – it is unstable.
        det_list = []   # [(obj_meta, det_dict), ...]
 
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
 
            r    = obj_meta.rect_params
            conf = float(obj_meta.tracker_confidence) or float(obj_meta.confidence)
 
            embed = None
            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break
                if user_meta.base_meta.meta_type == \
                        pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    try:
                        embed = _extract_embedding(
                            pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        )
                    except Exception:
                        pass
                try:
                    l_user = l_user.next
                except StopIteration:
                    break
 
            det_dict = {
                "local_track_id": int(obj_meta.object_id),
                "bbox": np.array(
                    [r.left, r.top, r.left + r.width, r.top + r.height],
                    dtype=np.float32
                ),
                "det_confidence": conf,
                "reid_vector":    embed,
            }
 
            if DIAG:
                print(
                    f"[DS]  frame={frame_meta.frame_num:5d}  "
                    f"ds_id={int(obj_meta.object_id):4d}  "
                    f"bbox=[{r.left:.0f},{r.top:.0f},"
                    f"{r.left+r.width:.0f},{r.top+r.height:.0f}]  "
                    f"conf={conf:.3f}  embed={'YES' if embed is not None else 'NO'}"
                )
 
            det_list.append((obj_meta, det_dict))
 
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
 
        # ── 2. Run tracker ────────────────────────────────────────────────────
        output_stracks = _tracker.update([d for _, d in det_list])
 
        # ── 3. Writeback via matched_det_idx – O(1), no centroid search ───────
        for t in output_stracks:
            idx = t.matched_det_idx
            if idx < 0 or idx >= len(det_list):
                continue   # ghost track surviving from previous frames
 
            obj_meta, _ = det_list[idx]
            global_id   = t.track_id
 
            obj_meta.object_id = global_id
 
            conf     = float(obj_meta.tracker_confidence or obj_meta.confidence)
            obj_meta.text_params.display_text = f"ReID:{global_id}\n{conf:.2f}"
            obj_meta.text_params.x_offset     = int(max(0, obj_meta.rect_params.left))
            obj_meta.text_params.y_offset     = int(max(0, obj_meta.rect_params.top - 30))
 
            obj_meta.rect_params.border_width       = 1
            obj_meta.rect_params.border_color.red   = 0.0
            obj_meta.rect_params.border_color.green = 1.0
            obj_meta.rect_params.border_color.blue  = 0.0
            obj_meta.rect_params.border_color.alpha = 1.0
 
        if DIAG:
            print(
                f"[TR]  frame={frame_meta.frame_num:5d}  "
                f"active={len(output_stracks)}  "
                f"lost={len(_tracker.lost_stracks)}"
            )
            for t in output_stracks:
                status = "REID" if t.tracklet_len == 0 and t.is_activated else "OK"
                print(
                    f"      ReID={t.track_id:4d}  "
                    f"det_idx={t.matched_det_idx:3d}  "
                    f"tracklet_len={t.tracklet_len:3d}  "
                    f"activated={t.is_activated}  [{status}]"
                )
 
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
 
    return Gst.PadProbeReturn.OK

def main():
    Gst.init(None)

    yaml_file = "/home/lab314/workspace/reid/ds_backend_reid/MCDPT/ds_include/app_config.yml"
    with open(yaml_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.stderr.write(f"Error in parsing configuration file: {exc}\n")
            return -1

    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline.new("dstest3-pipeline")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pipeline.add(streammux)

    source_list_config = config.get('source-list', {})
    sources = []
    for key, value in source_list_config.items():
        if key.startswith('list'):
            if isinstance(value, str):
                sources.extend(value.split(';'))
            elif isinstance(value, list):
                sources.extend(value)
    sources = [s for s in sources if s]

    num_sources = len(sources)
    for i, uri in enumerate(sources):
        sys.stdout.write(f"Now playing : {uri}\n")
        source_bin = create_source_bin(i, uri)
        if not source_bin:
            sys.stderr.write("Failed to create source bin. Exiting.\n")
            return -1

        pipeline.add(source_bin)
        pad_name = f"sink_{i}"
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            sys.stderr.write("Streammux request sink pad failed. Exiting.\n")
            return -1

        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Failed to get src pad of source bin. Exiting.\n")
            return -1

        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            sys.stderr.write("Failed to link source bin to stream muxer. Exiting.\n")
            return -1

    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine-1")
    nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")

    nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")

    is_aarch64 = platform.uname().machine == 'aarch64'
    
    if PERF_MODE:
        sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    else:
        if is_aarch64:
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not (pgie and sgie1 and nvdslogger and tiler and nvvidconv and nvosd and sink and nvtracker):
        sys.stderr.write("One element could not be created. Exiting.\n")
        return -1

    streammux_config = config.get('streammux', {})
    if 'width' in streammux_config: streammux.set_property('width', streammux_config['width'])
    if 'height' in streammux_config: streammux.set_property('height', streammux_config['height'])
    if 'batch-size' in streammux_config: streammux.set_property('batch-size', streammux_config['batch-size'])
    if 'batched-push-timeout' in streammux_config: streammux.set_property('batched-push-timeout', streammux_config['batched-push-timeout'])

    pgie_config = config.get('primary-gie', {})
    pgie_config_path = pgie_config.get('config-file') or pgie_config.get('config-file-path')
    if pgie_config_path:
        pgie.set_property('config-file-path', pgie_config_path)

    sgie1_config = config.get('secondary-gie-1', {})
    sgie1_config_path = sgie1_config.get('config-file') or sgie1_config.get('config-file-path')
    if sgie1_config_path:
        sgie1.set_property('config-file-path', sgie1_config_path)

    # Batch size override
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != num_sources:
        sys.stderr.write(f"WARNING: Overriding infer-config batch-size ({pgie_batch_size}) with number of sources ({num_sources})\n")
        pgie.set_property("batch-size", num_sources)
        sgie1.set_property("batch-size", num_sources)

    tracker_config = config.get('tracker', {})
    if 'll-config-file' in tracker_config: nvtracker.set_property('ll-config-file', tracker_config['ll-config-file'])
    if 'll-lib-file' in tracker_config: nvtracker.set_property('ll-lib-file', tracker_config['ll-lib-file'])

    nvosd.set_property("display-text", 1)
    nvosd.set_property("process-mode", 1)

    tiler_rows = int(math.sqrt(num_sources))
    tiler_columns = int(math.ceil(1.0 * num_sources / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    
    tiler_config = config.get('tiler', {})
    if 'width' in tiler_config: tiler.set_property('width', tiler_config['width'])
    if 'height' in tiler_config: tiler.set_property('height', tiler_config['height'])

    if PERF_MODE:
        if is_aarch64:
            streammux.set_property("nvbuf-memory-type", 4)
        else:
            streammux.set_property("nvbuf-memory-type", 2)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline_flow = [queue1, pgie, queue2, nvtracker, queue3, sgie1, nvdslogger, tiler, queue4, nvvidconv, queue5, nvosd, queue6, sink]

    for x in pipeline_flow: pipeline.add(x)
    streammux.link(pipeline_flow[0])
    for i, ds_element in enumerate(pipeline_flow):
        if i == len(pipeline_flow) - 1: break
        ds_element.link(pipeline_flow[i+1])

    reid_sgie_pad = nvtracker.get_static_pad("src")
    if not reid_sgie_pad:
        sys.stderr.write("Could not get nvdslogger src pad. Exiting.\n")
        return -1
    reid_sgie_pad.add_probe(Gst.PadProbeType.BUFFER, reid_pad_buffer_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)

    sys.stdout.write("Running...\n")
    try:
        loop.run()
    except BaseException:
        pass


        
    sys.stdout.write("Returned, stopping playback\n")
    pipeline.set_state(Gst.State.NULL)
    sys.stdout.write("Deleting pipeline\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())