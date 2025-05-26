import cv2
import numpy as np
import pyrealsense2 as rs
import socket
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from numpy.linalg import norm

UDP_IP = "192.168.222.113"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

target_id = None
target_features = []
target_mean_feat = None
distance_threshold = 0.25  # 精度を上げるため閾値を厳しく

def get_depth_center(depth_frame, cx, cy, size=2):
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
    return round(np.median(values), 2) if len(values) >= 5 else None

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        frame = np.asanyarray(color_frame.get_data())

        results = model(frame, classes=0)
        detections = []
        if results[0].boxes is not None:
            for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], float(conf), "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        # 再ロック用比較
        if target_mean_feat is not None:
            for track in tracks:
                if not track.is_confirmed() or not track.features:
                    continue
                if track.track_id == target_id:
                    continue
                dist = cosine_distance(track.features[-1], target_mean_feat)
                if dist < distance_threshold:
                    print(f"[INFO] 再ロック成功: 新ID {track.track_id}, 距離 {dist:.3f}")
                    target_id = track.track_id
                    target_features.append(track.features[-1])
                    if len(target_features) > 5:
                        target_features = target_features[-5:]
                    target_mean_feat = np.mean(target_features, axis=0)
                    break

        for track in tracks:
            if not track.is_confirmed() or not track.features:
                continue

            track_id = track.track_id

            if target_id is None:
                # 初期ターゲット取得：連続して5件特徴量が取れてからロック
                target_id = track_id
                target_features.append(track.features[-1])
                print(f"[INFO] ターゲット候補検出: ID {target_id}")
                if len(target_features) >= 5:
                    target_mean_feat = np.mean(target_features, axis=0)
                    print("[INFO] ターゲットロック完了（平均特徴量確定）")
                continue

            if track_id != target_id:
                continue

            # 特徴更新
            target_features.append(track.features[-1])
            target_features = target_features[-5:]
            target_mean_feat = np.mean(target_features, axis=0)

            l, t, w, h = map(int, track.to_ltrb())
            cx = int(l + w / 2)
            cy = int(t + h / 2)
            depth = get_depth_center(depth_frame, cx, cy)
            dist_text = f"{depth:.2f} m" if depth else "N/A"

            # 描画（赤枠、距離のみ）
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 0, 255), 2)
            cv2.putText(frame, dist_text, (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        sock.sendto(jpeg.tobytes(), (UDP_IP, UDP_PORT))

finally:
    pipeline.stop()
