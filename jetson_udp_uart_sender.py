# jetson_udp_uart_sender.py
# ReID 再取得強化版：
# - グローバル再取得（位置ゲート解除＋厳しめ外観一致＋連続一致で確定）
# - HSVヒスト併用、特徴ギャラリーEMA、端から退出検知
# - --embedder-model/--embedder-weights に対応（TorchReIDに学習済み重みを指定可）
# - 既定は追従安定寄り：conf=0.45, max_age=30, n_init=3, bbox-ema=0.6

import argparse
import socket
import cv2
import numpy as np
from collections import deque
import math
import time

# --- NumPy 2.x 互換パッチ ---
for _alias, _target in (('float', float), ('int', int), ('bool', bool), ('object', object), ('long', int)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import pyrealsense2 as rs
import serial
from serial import SerialException
from ultralytics import YOLO
import torch, torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- TorchVision NMS フォールバック ---
def _tv_nms_available():
    try:
        _ = torchvision.ops.nms(torch.zeros((1, 4)), torch.zeros(1), 0.5)
        return True
    except Exception as e:
        print(f"[WARN] torchvision.ops.nms unavailable: {e}")
        return False

if not _tv_nms_available():
    def _box_iou(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2 - inter + 1e-7
        return inter / union

    def _nms_fallback(boxes, scores, iou_thres: float):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        keep = []
        idxs = scores.argsort(descending=True)
        while idxs.numel() > 0:
            i = idxs[0].item()
            keep.append(i)
            if idxs.numel() == 1:
                break
            rest = idxs[1:]
            ious = _box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            rest = rest[ious <= iou_thres]
            idxs = rest
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    torchvision.ops.nms = _nms_fallback
    print("[INFO] Using pure-PyTorch NMS fallback (slower but compatible).")
# --- end patch ---


def parse_args():
    p = argparse.ArgumentParser()

    # 出力先（そのまま）
    p.add_argument("--ip", default="192.168.222.100")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--uart", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)

    # YOLO
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--show", action="store_true")

    # DeepSORT
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--n-init", type=int, default=3)

    # ReID（TorchReID を既定に）
    p.add_argument("--embedder", type=str, default="torchreid", help="mobilenet|torchreid|clip|...")
    p.add_argument("--embedder-model", type=str, default="osnet_x0_25", help="TorchReID model name（例: osnet_x0_25）")
    p.add_argument("--embedder-weights", type=str, default="", help="TorchReID学習済み重みのパス（.pth/.pt）")

    # 再取得チューニング（既定を “再取得強め” に）
    p.add_argument("--reid-th", type=float, default=0.35)        # 基本閾値（cos距離。小さいほど厳しい）
    p.add_argument("--reid-th-step", type=float, default=0.03)   # SUSPECT/LOST で少し緩む
    p.add_argument("--reid-th-max", type=float, default=0.55)
    p.add_argument("--global-reid-th", type=float, default=0.33) # グローバル再取得時の厳しめ上限（-1で自動=より厳しい）

    # 位置・スケールのゲート（広めに）
    p.add_argument("--gate-ratio", type=float, default=0.30)
    p.add_argument("--gate-ratio-suspect", type=float, default=0.45)
    p.add_argument("--gate-ratio-lost", type=float, default=0.65)
    p.add_argument("--scale-ratio", type=float, default=1.6)

    # LOST 遷移タイミング（早めに global へ）
    p.add_argument("--lost1", type=int, default=8)    # → SUSPECT
    p.add_argument("--lost2", type=int, default=20)   # → LOST（長期；≈0.7秒@30fps）

    # EMA / 制御スムース（既定で有効、--no-smooth-control で無効化可）
    p.add_argument("--bbox-ema", type=float, default=0.60)
    p.add_argument("--smooth-control", action=argparse.BooleanOptionalAction, default=True)

    # 色ヒスト（通しやすく）
    p.add_argument("--use-hist", action="store_true", default=True)
    p.add_argument("--hist-ema", type=float, default=0.5)
    p.add_argument("--hist-th", type=float, default=0.65, help="Bhattacharyya距離（小さいほど近い）")

    # 端からの退出検知（判定を得やすく）
    p.add_argument("--edge-margin", type=int, default=40)
    p.add_argument("--reacquire-confirm", type=int, default=2, help="連続一致フレーム数")

    # RealSense 露出ロック（任意）
    p.add_argument("--rs-exposure", type=float, default=-1.0)
    p.add_argument("--rs-gain", type=float, default=-1.0)

    return p.parse_args()



def safe_imshow(win, img):
    try:
        cv2.imshow(win, img)
    except cv2.error:
        pass

def safe_destroy_all_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

def tlbr_from_track(track):
    if hasattr(track, "to_tlbr"):
        l, t, r, b = map(int, track.to_tlbr());  return l, t, r, b
    if hasattr(track, "to_ltrb"):
        l, t, r, b = map(int, track.to_ltrb());  return l, t, r, b
    if hasattr(track, "to_tlwh"):
        x, y, w, h = map(int, track.to_tlwh());  return x, y, x+w, y+h
    return None

def _center(bb):
    x1,y1,x2,y2 = bb
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _area(bb):
    x1,y1,x2,y2 = bb
    return max(1.0, (x2-x1)*(y2-y1))

def _diag(w,h):
    return math.hypot(w,h)

def _within_gate(prev_bbox, cand_bbox, frame_shape, state, scale_ratio, base_gate, gate_suspect, gate_lost, global_mode=False):
    if prev_bbox is None:
        return True
    if global_mode:  # 位置制限を外す
        return True
    H, W = frame_shape[:2]
    diag = _diag(W, H)
    cxp, cyp = _center(prev_bbox)
    cxc, cyc = _center(cand_bbox)
    dist = math.hypot(cxc-cxp, cyc-cyp)
    if state == "LOCKED":
        gate = base_gate * diag
    elif state == "SUSPECT":
        gate = gate_suspect * diag
    else:
        gate = gate_lost * diag
    ap = _area(prev_bbox)
    ac = _area(cand_bbox)
    ratio = max(ac/ap, ap/ac)
    return (dist <= gate) and (ratio <= scale_ratio)

def _adaptive_reid_threshold(base, step, maxv, lost_frames, lost1, lost2):
    if lost_frames < lost1:
        relax = 0.0
    elif lost_frames < lost2:
        relax = step * 1
    else:
        relax = step * 2
    return min(maxv, base + relax)

def get_depth_center(depth_frame, cx, cy, size=2):
    vals = []
    w, h = depth_frame.get_width(), depth_frame.get_height()
    for dx in range(-size, size+1):
        for dy in range(-size, size+1):
            px, py = cx+dx, cy+dy
            if 0 <= px < w and 0 <= py < h:
                d = depth_frame.get_distance(px, py)
                if 0.1 <= d <= 5.0:  # m
                    vals.append(d)
    return round(float(np.median(vals)), 2) if len(vals) >= 5 else None

class BBoxEMA:
    def __init__(self, alpha: float):
        self.a = float(alpha)
        self.state = {}  # tid -> (cx,cy,w,h)
    def reset(self, tid):
        self.state.pop(tid, None)
    def update(self, tid, ltrb):
        l,t,r,b = ltrb
        cx,cy = 0.5*(l+r), 0.5*(t+b)
        w,h = max(1.0, r-l), max(1.0, b-t)
        if self.a <= 0.0 or tid not in self.state:
            self.state[tid] = (cx,cy,w,h);  return (cx,cy,w,h)
        px,py,pw,ph = self.state[tid]
        a = self.a
        nx = a*px + (1-a)*cx
        ny = a*py + (1-a)*cy
        nw = a*pw + (1-a)*w
        nh = a*ph + (1-a)*h
        self.state[tid] = (nx,ny,nw,nh)
        return (nx,ny,nw,nh)

def compute_hs_hist(frame, l, t, r, b):
    H, W = frame.shape[:2]
    l = max(0, l); t = max(0, t); r = min(W-1, r); b = min(H-1, b)
    if r <= l+1 or b <= t+1:
        return None
    crop = frame[t:b, l:r]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180, 0,256])
    cv2.normalize(hist, hist)  # L2
    return hist

def hist_distance(h1, h2):
    # Bhattacharyya（0に近いほど近い）
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

def lock_realsense_exposure(pipeline, exposure, gain):
    if exposure < 0 and gain < 0:
        return
    try:
        profile = pipeline.get_active_profile()
        for s in profile.get_device().query_sensors():
            if s.supports(rs.option.enable_auto_exposure):
                if exposure >= 0:
                    s.set_option(rs.option.enable_auto_exposure, 0)
                    s.set_option(rs.option.exposure, float(exposure))
                if gain >= 0:
                    s.set_option(rs.option.gain, float(gain))
                print(f"[INFO] RealSense exposure locked: exposure={exposure}, gain={gain}")
                break
    except Exception as e:
        print(f"[WARN] Failed to set RealSense exposure: {e}")

def init_deepsort(args):
    kw = dict(max_age=args.max_age, n_init=args.n_init,
              embedder_gpu=torch.cuda.is_available())
    emb = (args.embedder or "").lower()
    if emb in ("torchreid","reid","osnet"):
        try:
            # 学習済み重みを直接指定可能
            extra = {}
            if args.embedder_weights:
                extra["embedder_model_path"] = args.embedder_weights
            tracker = DeepSort(embedder="torchreid",
                               embedder_model_name=args.embedder_model,
                               **extra, **kw)
            print(f"[INFO] DeepSORT embedder=torchreid model={args.embedder_model} weights={args.embedder_weights or 'builtin'}")
            return tracker
        except Exception as e:
            print(f"[WARN] torchreid unavailable ({e}); falling back to mobilenet")
    try:
        tracker = DeepSort(embedder=args.embedder, **kw)
        print(f"[INFO] DeepSORT embedder={args.embedder}")
        return tracker
    except Exception as e:
        print(f"[WARN] embedder={args.embedder} not supported ({e}); using mobilenet")
        return DeepSort(embedder="mobilenet", **kw)

def main():
    args = parse_args()

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # UART
    ser = None
    try:
        ser = serial.Serial(args.uart, args.baud, timeout=0)
        print(f"[INFO] UART open {args.uart} @ {args.baud}")
    except Exception as e:
        print(f"[WARN] UART open failed: {e}")

    # YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()
    model = YOLO(args.model)

    # DeepSORT
    tracker = init_deepsort(args)

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    lock_realsense_exposure(pipeline, args.rs_exposure, args.rs_gain)

    # 追跡状態
    target_id = None
    recent_features = deque(maxlen=5)  # 直近特徴（平均化に使用）
    hist_ref = None                     # HSVヒストの参照（EMA）
    lost_frames = 0
    track_state = "LOCKED"
    last_bbox = None
    ema = BBoxEMA(alpha=args.bbox_ema)
    edge_exit = False                   # 端から退出した可能性
    pending_tid, pending_cnt = None, 0  # グローバル再取得の連続一致判定

    # FPS
    t_prev = time.time()
    fps_smooth = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            H, W = frame.shape[:2]

            # YOLO（personのみ）
            results = model.predict(frame, classes=0, device=device,
                                    conf=args.conf, iou=args.iou,
                                    half=half, verbose=False)

            detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                xyxy = results[0].boxes.xyxy
                confs = results[0].boxes.conf
                if xyxy.is_cuda: xyxy = xyxy.detach().cpu()
                if confs.is_cuda: confs = confs.detach().cpu()
                xyxy, confs = xyxy.numpy(), confs.numpy()
                for box, c in zip(xyxy, confs):
                    x1,y1,x2,y2 = box[:4]
                    w,h = x2-x1, y2-y1
                    detections.append(([float(x1), float(y1), float(w), float(h)], float(c), "person"))

            # DeepSORT update
            try:
                tracks = tracker.update_tracks(detections, frame=frame)
            except Exception:
                tracks = []

            # 既存ターゲットの可視判定
            seen_target = False
            for tr in tracks:
                if not tr.is_confirmed() or not getattr(tr, "features", None):
                    continue
                if target_id is not None and tr.track_id == target_id:
                    seen_target = True
                    rect = tlbr_from_track(tr)
                    if rect is not None:
                        last_bbox = rect
                        # 端から退出の可能性をリセット
                        edge_exit = False
                    # 特徴更新
                    feat = tr.features[-1]
                    if isinstance(feat, np.ndarray):
                        feat = feat.astype(np.float32)
                    recent_features.append(feat)
                    # ヒスト更新（EMA）
                    if args.use_hist and rect is not None:
                        h = compute_hs_hist(frame, *rect)
                        if h is not None:
                            if hist_ref is None:
                                hist_ref = h
                            else:
                                hist_ref = args.hist_ema * hist_ref + (1.0 - args.hist_ema) * h
                    break

            # 状態遷移
            if seen_target:
                lost_frames = 0
                track_state = "LOCKED"
                pending_tid, pending_cnt = None, 0
            else:
                # 直前に端に近かったら「端退出」フラグを立てる
                if last_bbox is not None:
                    l,t,r,b = last_bbox
                    if (l <= args.edge_margin) or (t <= args.edge_margin) or \
                       (r >= W-1-args.edge_margin) or (b >= H-1-args.edge_margin):
                        edge_exit = True

                lost_frames += 1
                if lost_frames < args.lost1:
                    track_state = "LOCKED"
                elif lost_frames < args.lost2:
                    track_state = "SUSPECT"
                else:
                    track_state = "LOST"

                # 再取得（候補選定）
                # global_mode: 長期ロスト or 端退出 → 位置ゲート解除
                global_mode = (lost_frames >= args.lost2) or edge_exit

                # 閾値設定
                base_th = _adaptive_reid_threshold(args.reid_th, args.reid_th_step, args.reid_th_max,
                                                   lost_frames, args.lost1, args.lost2)
                if global_mode:
                    strict = (args.global_reid_th if args.global_reid_th > 0
                              else max(0.25, args.reid_th - 0.07))
                    base_th = min(base_th, strict)  # グローバルではむしろ厳しめ

                ref_feat = None
                if len(recent_features) > 0:
                    ref_feat = np.mean(np.stack(list(recent_features), axis=0), axis=0)
                    ref_feat = ref_feat / (np.linalg.norm(ref_feat) + 1e-6)

                best_track, best_score = None, -1e9
                for tr in tracks:
                    if not tr.is_confirmed() or not getattr(tr, "features", None):
                        continue
                    if target_id is not None and tr.track_id == target_id:
                        continue
                    tb = tlbr_from_track(tr)
                    if tb is None:
                        continue

                    # 位置・スケールのゲート（global_mode なら解除）
                    if not _within_gate(last_bbox, tb, frame.shape, track_state,
                                        args.scale_ratio, args.gate_ratio, args.gate_ratio_suspect,
                                        args.gate_ratio_lost, global_mode=global_mode):
                        continue

                    # ReID（cos距離）
                    cos_ok, cos_sim = True, 0.0
                    if ref_feat is not None:
                        cur = tr.features[-1]
                        cur = cur / (np.linalg.norm(cur) + 1e-6)
                        cos_dist = 1.0 - float(np.dot(ref_feat, cur))
                        cos_ok = (cos_dist <= base_th)
                        cos_sim = 1.0 - cos_dist
                    if not cos_ok:
                        continue

                    # ヒスト併用
                    hist_ok = True
                    if args.use_hist and hist_ref is not None:
                        h = compute_hs_hist(frame, *tb)
                        if h is not None:
                            d = hist_distance(hist_ref, h)  # 小さいほど近い
                            hist_ok = (d <= args.hist_th)
                            if not hist_ok:
                                continue
                            # スコアに色一致も加点（d=0が最高）
                            color_bonus = 1.0 - d  # 0..1
                        else:
                            color_bonus = 0.0
                    else:
                        color_bonus = 0.0

                    # スコア：外観＋近さ（global時は外観をより重視）
                    c_prev = _center(last_bbox) if last_bbox is not None else _center(tb)
                    c_cur  = _center(tb)
                    dist = np.hypot(c_cur[0]-c_prev[0], c_cur[1]-c_prev[1]) + 1e-6
                    if global_mode:
                        score = 2.0*cos_sim + 0.3*color_bonus + 0.2*(1.0/dist)
                    else:
                        score = 1.5*cos_sim + 0.2*color_bonus + 0.5*(1.0/dist)
                    if score > best_score:
                        best_score, best_track = score, tr

                # 候補確定
                if best_track is not None:
                    if global_mode:
                        # 連続一致で確定
                        if pending_tid == best_track.track_id:
                            pending_cnt += 1
                        else:
                            pending_tid, pending_cnt = best_track.track_id, 1
                        if pending_cnt >= args.reacquire_confirm:
                            target_id = best_track.track_id
                            last_bbox = tlbr_from_track(best_track)
                            # 特徴/ヒスト初期化
                            feat = best_track.features[-1]
                            if isinstance(feat, np.ndarray):
                                feat = feat.astype(np.float32)
                            recent_features.clear()
                            recent_features.append(feat)
                            if args.use_hist and last_bbox is not None:
                                h = compute_hs_hist(frame, *last_bbox)
                                hist_ref = h if h is not None else hist_ref
                            lost_frames = 0
                            track_state = "LOCKED"
                            edge_exit = False
                            pending_tid, pending_cnt = None, 0
                    else:
                        # 位置付き再取得は即確定
                        target_id = best_track.track_id
                        last_bbox = tlbr_from_track(best_track)
                        feat = best_track.features[-1]
                        if isinstance(feat, np.ndarray):
                            feat = feat.astype(np.float32)
                        recent_features.append(feat)
                        if args.use_hist and last_bbox is not None:
                            h = compute_hs_hist(frame, *last_bbox)
                            hist_ref = h if h is not None else hist_ref
                        pending_tid, pending_cnt = None, 0

            # 描画・UART
            sent = False
            if target_id is None:
                # 初回：最も信頼できるトラックを採用
                for tr in tracks:
                    if tr.is_confirmed() and getattr(tr, "features", None):
                        target_id = tr.track_id
                        last_bbox = tlbr_from_track(tr)
                        feat = tr.features[-1]
                        if isinstance(feat, np.ndarray):
                            feat = feat.astype(np.float32)
                        recent_features.append(feat)
                        if args.use_hist and last_bbox is not None:
                            h = compute_hs_hist(frame, *last_bbox)
                            hist_ref = h if h is not None else hist_ref
                        ema.reset(target_id)
                        break

            for tr in tracks:
                if not tr.is_confirmed() or not getattr(tr, "features", None):
                    continue
                if tr.track_id != target_id:
                    continue
                rect = tlbr_from_track(tr)
                if rect is None:
                    continue
                last_bbox = rect

                # EMA
                cx, cy, w, h = ema.update(tr.track_id, rect)
                l = int(cx - w/2); r = int(cx + w/2)
                t = int(cy - h/2); b = int(cy + h/2)

                depth = get_depth_center(depth_frame, int(cx), int(cy))
                dist_text = f"{depth:.2f} m" if depth else "N/A"

                try:
                    color = (0,0,255) if track_state!="LOCKED" else (0,255,0)
                    cv2.rectangle(frame, (l,t), (r,b), color, 2)
                    cv2.putText(frame, f"{dist_text}", (l, max(0,t-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"STATE:{track_state} LOST:{lost_frames} G:{'Y' if (lost_frames>=args.lost2 or edge_exit) else 'N'}",
                                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                except cv2.error:
                    pass

                if depth is not None:
                    if args.smooth_control:
                        dx = int(cx - W/2)
                        dy = int(cy - H/2)
                    else:
                        l0,t0,r0,b0 = rect
                        cx0, cy0 = 0.5*(l0+r0), 0.5*(t0+b0)
                        dx = int(cx0 - W/2); dy = int(cy0 - H/2)
                    msg = f"{dx},{dy},{depth:.2f}\n"
                    if ser:
                        try:
                            ser.write(msg.encode())
                        except SerialException as e:
                            print(f"[WARN] UART write failed: {e}")
                    print(f"[UART → Pico] {msg.strip()}")
                    sent = True

            # AndroidへJPEG
            try:
                # FPS
                now = time.time()
                dt = now - t_prev
                t_prev = now
                inst = 1.0/dt if dt>0 else 0.0
                fps_smooth = inst if fps_smooth is None else 0.9*fps_smooth + 0.1*inst
                try:
                    cv2.putText(frame, f"FPS:{fps_smooth:.1f}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                except cv2.error:
                    pass

                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                sock.sendto(jpeg.tobytes(), (args.ip, args.port))
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

            if args.show:
                safe_imshow("view", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    except KeyboardInterrupt:
        print("[INFO] 停止要求")
    finally:
        try: pipeline.stop()
        except Exception: pass
        try:
            if ser: ser.close()
        except Exception: pass
        try: sock.close()
        except Exception: pass
        safe_destroy_all_windows()

if __name__ == "__main__":
    main()
