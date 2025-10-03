# jetson_udp_uart_sender.py (NumPy2互換 + NMSフォールバック + GUIガード + UART例外) 完全修正版

import argparse
import socket
import cv2
import numpy as np

# --- NumPy 2.x 互換パッチ（古いコードが np.float/np.int 等を使っても落ちないように） ---
for _alias, _target in (('float', float), ('int', int), ('bool', bool), ('object', object), ('long', int)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import pyrealsense2 as rs
import serial
from serial import SerialException
from ultralytics import YOLO
from numpy.linalg import norm

# DeepSORT は NumPy の上書きを効かせた後で import
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- TorchVision NMS fallback patch (C++ opsが壊れてても動かす) ---
import torch, torchvision

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
            ious = _box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
            idxs = idxs[1:][ious <= iou_thres]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    torchvision.ops.nms = _nms_fallback
    print("[INFO] Using pure-PyTorch NMS fallback (slower but compatible).")
# --- end patch ---

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="192.168.222.113", help="UDP destination IP (Android)")
    p.add_argument("--port", type=int, default=5005, help="UDP destination port")
    p.add_argument("--uart", default="/dev/ttyACM0", help="UART device to Pico")
    p.add_argument("--baud", type=int, default=115200, help="UART baudrate")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model path/name")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold")
    p.add_argument("--show", action="store_true", help="Show window (requires GUI-enabled OpenCV)")
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

def get_depth_center(depth_frame, cx, cy, size=2):
    """近傍( (2*size+1)^2 ) の有効距離の中央値[m]を返す。"""
    values = []
    w, h = depth_frame.get_width(), depth_frame.get_height()
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            px = cx + dx
            py = cy + dy
            if 0 <= px < w and 0 <= py < h:
                d = depth_frame.get_distance(px, py)
                if 0.1 < d < 5.0:
                    values.append(d)
    return round(float(np.median(values)), 2) if len(values) >= 5 else None

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def tlbr_from_track(track):
    """DeepSort の Track から (l,t,r,b) を頑健に取得"""
    if hasattr(track, "to_tlbr"):
        l, t, r, b = map(int, track.to_tlbr());  return l, t, r, b
    if hasattr(track, "to_ltrb"):
        l, t, r, b = map(int, track.to_ltrb());  return l, t, r, b
    if hasattr(track, "to_ltwh"):
        l, t, w, h = map(int, track.to_ltwh());  return l, t, l + w, t + h
    return None

def main():
    args = parse_args()

    # --- UDP ソケット ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # --- UART 初期化（未接続/権限NGでも継続） ---
    ser = None
    try:
        ser = serial.Serial(args.uart, args.baud, timeout=1)
        print(f"[INFO] UART opened: {args.uart} @ {args.baud}")
    except SerialException as e:
        print(f"[WARN] UART open failed: {e} (UART送信は無効化)")

    # --- YOLO 初期化（CUDA/FP16自動） ---
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()
    model = YOLO(args.model)

    # --- DeepSORT 初期化（GPUで埋め込み計算可能なら使用） ---
    tracker = DeepSort(max_age=30, embedder_gpu=torch.cuda.is_available())

    # --- RealSense 初期化 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # --- ターゲット追跡状態 ---
    target_id = None
    target_features = []
    target_mean_feat = None
    distance_threshold = 0.25  # ReID 類似度しきい値（小さいほど厳しい）

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # YOLO 推論（人物のみ）
            results = model.predict(
                frame, classes=0, device=device, conf=args.conf, half=half, verbose=False
            )

            detections = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                xyxy = results[0].boxes.xyxy
                confs = results[0].boxes.conf
                if xyxy.is_cuda: xyxy = xyxy.detach().cpu()
                if confs.is_cuda: confs = confs.detach().cpu()
                xyxy = xyxy.numpy();  confs = confs.numpy()
                for box, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = box[:4]
                    w, h = x2 - x1, y2 - y1
                    detections.append(([float(x1), float(y1), float(w), float(h)], float(conf), "person"))

            # 検出0件フレームは DeepSORT をスキップ（旧版の内部assert回避）
            if len(detections) == 0:
                tracks = []
            else:
                tracks = tracker.update_tracks(detections, frame=frame)

            # ReID: ターゲットの自動乗り換え
            if target_mean_feat is not None and tracks:
                for track in tracks:
                    if not track.is_confirmed() or not getattr(track, "features", None):
                        continue
                    if track.track_id == target_id:
                        continue
                    dist = cosine_distance(track.features[-1], target_mean_feat)
                    if dist < distance_threshold:
                        target_id = track.track_id
                        target_features.append(track.features[-1])
                        target_features = target_features[-5:]
                        target_mean_feat = np.mean(target_features, axis=0)
                        break

            # ターゲットの描画・UART送信
            for track in tracks:
                if not track.is_confirmed() or not getattr(track, "features", None):
                    continue

                tid = track.track_id
                if target_id is None:
                    target_id = tid
                    target_features.append(track.features[-1])
                    if len(target_features) >= 5:
                        target_mean_feat = np.mean(target_features, axis=0)
                    continue

                if tid != target_id:
                    continue

                target_features.append(track.features[-1])
                target_features = target_features[-5:]
                target_mean_feat = np.mean(target_features, axis=0)

                rect = tlbr_from_track(track)
                if rect is None:
                    continue
                l, t, r, b = rect
                w, h = r - l, b - t
                cx = int(l + w / 2)
                cy = int(t + h / 2)

                depth = get_depth_center(depth_frame, cx, cy)
                dist_text = f"{depth:.2f} m" if depth else "N/A"

                # 描画（headlessでもOK）
                try:
                    cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(frame, dist_text, (l, max(0, t - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except cv2.error:
                    pass

                # UART 送信
                if depth is not None:
                    dx = cx - 320
                    dy = cy - 240
                    msg = f"{dx},{dy},{depth:.2f}\n"
                    if ser:
                        try:
                            ser.write(msg.encode())
                        except SerialException as e:
                            print(f"[WARN] UART write failed: {e}")
                    print(f"[UART → Pico] {msg.strip()}")

            # Android へ MJPEG（UDP）
            try:
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                sock.sendto(jpeg.tobytes(), (args.ip, args.port))
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

            if args.show:
                safe_imshow("YOLO+Depth", frame)
                if cv2.waitKey(1) & 0xFF == 27:
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
