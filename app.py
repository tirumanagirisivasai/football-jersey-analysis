import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from rfdetr import RFDETRMedium
import sports

# ==========================CONFIGURATION & HYPERPARAMETERS================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLAYER_CLASS_ID = 3
BALL_CLASS_ID = 1
NET_CLASS_ID = 2

# Tracking Parameters
MAX_MISSES = 25
IOU_TH = 0.3
MATCH_TH = 0.45
SMOOTH_ALPHA = 0.7  # Higher = smoother, Lower = more responsive

# Team & OCR Parameters
TEAM_MIN_AGE = 15
TEAM_LOCK_FRAMES = 20
OCR_INTERVAL = 30
JERSEY_LOCK_CONF = 2.5
OCR_CONF_TH = 0.89
GOAL_CONFIRM_FRAMES = 3

TEAM_COLORS = {
    0: (255, 0, 0),       # Team A: Red
    1: (0, 0, 255),       # Team B: Blue
    None: (0, 165, 255)   # Unknown: Orange
}

# ==========================HELPER FUNCTIONS====================================

def get_jersey_torso(player_img, alpha=1.6, beta=20):
    """
    Standardizes the torso crop for team classification to avoid 
    background noise and non-jersey colors.
    """
    if player_img is None or player_img.size == 0:
        return None
        
    h, w, _ = player_img.shape
    y1, y2 = int(0.25 * h), int(0.60 * h)
    x1, x2 = int(0.20 * w), int(0.80 * w)
    
    crop = player_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    return cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)

# ==========================MODELS & ENGINES=================================
class JerseyOCR:
    def __init__(self, model_path, num_classes=100):
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval().to(DEVICE)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, crop):
        if crop is None or crop.size == 0:
            return -1, 0.0
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        x = self.tf(img).unsqueeze(0).to(DEVICE)
        p = torch.softmax(self.model(x), 1)
        conf, idx = torch.max(p, 1)
        return int(idx.item()), float(conf.item())

# Global Model Initialization
det_model = RFDETRMedium(pretrain_weights='/kaggle/input/7bbr4894nr/player-ball-net-model_3.pth', device=DEVICE)
jersey_area_model = YOLO('/kaggle/input/7bbr4894nr/jersey_ocr_best.pt')

# Re-ID Model Setup
reid_backbone = models.resnet50(weights="IMAGENET1K_V1")
reid_backbone.fc = nn.Identity()
reid_backbone.eval().to(DEVICE)
reid_tf = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================TRACKING LOGIC=============================

class Track:
    def __init__(self, tid, box, emb):
        self.id = tid
        self.box = np.array(box, dtype=float)
        self.smooth_box = self.box.copy()
        self.emb = emb
        self.misses = 0
        self.age = 1

        # Team Voting
        self.team_votes = defaultdict(int)
        self.team_id = None
        self.team_locked = False

        # Jersey OCR Voting
        self.jersey_votes = defaultdict(float)
        self.locked_jersey = None
        self.last_ocr_frame = -1

    def update(self, box, emb):
        # EMA Smoothing for bounding box flicker
        self.smooth_box = (SMOOTH_ALPHA * self.smooth_box) + ((1 - SMOOTH_ALPHA) * np.array(box))
        self.box = self.smooth_box.astype(int)

        if emb is not None:
            self.emb = 0.8 * self.emb + 0.2 * emb
            self.emb /= (np.linalg.norm(self.emb) + 1e-6)

        self.age += 1
        self.misses = 0

class HybridTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, boxes, embs):
        used = set()
        for b, e in zip(boxes, embs):
            best_i, best_score = -1, MATCH_TH
            for i, t in enumerate(self.tracks):
                if i in used: continue
                iou_val = self._iou(b, t.box)
                if iou_val < IOU_TH: continue

                cos_sim = np.dot(e, t.emb)
                score = 0.5 * iou_val + 0.5 * cos_sim
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0:
                self.tracks[best_i].update(b, e)
                used.add(best_i)
            else:
                self.tracks.append(Track(self.next_id, b, e))
                self.next_id += 1

        for i, t in enumerate(self.tracks):
            if i not in used: t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

#---------MAIN CODE FROM HERE-----------
def run_pipeline(video_in, video_out, ocr_engine):
    cap = cv2.VideoCapture(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("🚀 Stage 1: Calibrating Teams...")
    player_crops_raw = []
    brightened_jersey_crops = []
    sample_stride = max(1, total_frames // 300)

    for fidx in range(0, total_frames, sample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue

        res = det_model.predict(frame, threshold=0.4)
        for i, cls in enumerate(res.class_id):
            if int(cls) == PLAYER_CLASS_ID:
                bx = res.xyxy[i].astype(int)
                bx[0], bx[1] = max(0, bx[0]), max(0, bx[1])
                bx[2], bx[3] = min(w, bx[2]), min(h, bx[3])
                
                crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
                if crop.size > 5000:
                    player_crops_raw.append(crop)
        
        if len(player_crops_raw) >= 200: break

    if len(player_crops_raw) < 20:
        print("❌ Not enough players for calibration. Exiting.")
        return

    for p in player_crops_raw:
        j_crop = get_jersey_torso(p, alpha=1.6, beta=20)
        if j_crop is not None:
            brightened_jersey_crops.append(j_crop)

    team_classifier = sports.TeamClassifier(device=DEVICE)
    team_classifier.fit(brightened_jersey_crops)

    print("🚀 Stage 2: Processing Video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker = HybridTracker()
    goal_state = {"frames": 0, "scored": False}
    pbar = tqdm(total=total_frames)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = det_model.predict(frame, threshold=0.5)
        ball_boxes, net_boxes = [], []
        player_reid_batch, player_boxes_batch = [], []

        # 1. Detection Sorting & Re-ID Batch Preparation
        for i, cls in enumerate(results.class_id):
            box = results.xyxy[i].astype(int)
            box[0], box[1] = max(0, box[0]), max(0, box[1])
            box[2], box[3] = min(w, box[2]), min(h, box[3])

            if cls == PLAYER_CLASS_ID:
                # Use standard torso for Re-ID consistency
                reid_crop = get_jersey_torso(frame[box[1]:box[3], box[0]:box[2]], alpha=2, beta=30)
                if reid_crop is not None:
                    img_pil = Image.fromarray(cv2.cvtColor(reid_crop, cv2.COLOR_BGR2RGB))
                    player_reid_batch.append(reid_tf(img_pil))
                    player_boxes_batch.append(box)
            elif cls == BALL_CLASS_ID:
                ball_boxes.append(box)
            elif cls == NET_CLASS_ID:
                net_boxes.append(box)

        # 2. Update Tracker
        if player_reid_batch:
            batch_t = torch.stack(player_reid_batch).to(DEVICE)
            with torch.no_grad():
                feats = reid_backbone(batch_t)
                feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-6)
            tracker.update(player_boxes_batch, feats.cpu().numpy())

        # 3. Goal Detection
        if not goal_state["scored"]:
            for ball in ball_boxes:
                bc = ((ball[0] + ball[2]) // 2, (ball[1] + ball[3]) // 2)
                for net in net_boxes:
                    if net[0] <= bc[0] <= net[2] and net[1] <= bc[1] <= net[3]:
                        goal_state["frames"] += 1
                        if goal_state["frames"] >= GOAL_CONFIRM_FRAMES:
                            goal_state["scored"] = True

        # 4. Team Identification & OCR (Batch logic)
        team_batch_crops, team_batch_tracks = [], []
        for t in tracker.tracks:
            if t.misses > 0: continue
            
            p_crop_raw = frame[t.box[1]:t.box[3], t.box[0]:t.box[2]]
            if p_crop_raw.size == 0: continue

            # Team Logic
            if not t.team_locked and t.age >= TEAM_MIN_AGE:
                j_torso = get_jersey_torso(p_crop_raw, alpha=1.6, beta=20)
                if j_torso is not None:
                    team_batch_crops.append(j_torso)
                    team_batch_tracks.append(t)

            # OCR Logic
            if (frame_idx - t.last_ocr_frame) >= OCR_INTERVAL or t.locked_jersey is None:
                j_res = jersey_area_model.predict(p_crop_raw, conf=0.4, verbose=False)
                if len(j_res[0].boxes) > 0:
                    jb = j_res[0].boxes.xyxy[0].cpu().numpy().astype(int)
                    j_num_crop = p_crop_raw[max(0,jb[1]):jb[3], max(0,jb[0]):jb[2]]
                    
                    if j_num_crop.size > 0:
                        num, conf = ocr_engine.predict(j_num_crop)
                        t.last_ocr_frame = frame_idx
                        if conf > OCR_CONF_TH:
                            t.jersey_votes[num] += conf
                            best_n, best_s = max(t.jersey_votes.items(), key=lambda x: x[1])
                            if best_s >= JERSEY_LOCK_CONF: 
                                t.locked_jersey = best_n

        if team_batch_crops:
            t_preds = team_classifier.predict(team_batch_crops)
            for p, t in zip(t_preds, team_batch_tracks):
                t.team_votes[int(p)] += 1
                if sum(t.team_votes.values()) >= TEAM_LOCK_FRAMES:
                    t.team_id = max(t.team_votes, key=t.team_votes.get)
                    t.team_locked = True

        # 5. Drawing
        for t in tracker.tracks:
            if t.misses > 0: continue
            color = TEAM_COLORS[t.team_id] if t.team_locked else TEAM_COLORS[None]
            x1, y1, x2, y2 = t.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.id} T:{t.team_id if t.team_locked else '?'} #{t.locked_jersey or '?'}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for b in ball_boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            
        for n in net_boxes:
            cv2.rectangle(frame, (n[0], n[1]), (n[2], n[3]), (255, 255, 0), 2)

        if goal_state["scored"]:
            cv2.putText(frame, "GOAL!", (w // 2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"✅ Processing Complete. Saved to: {video_out}")


if __name__ == "__main__":
    ocr_model_path = 'ocr_model_best_v2.pth'
    input_video = 'output-3_9.mp4'
    output_video = 'report_final.mp4'

    ocr_engine = JerseyOCR(ocr_model_path)
    run_pipeline(input_video, output_video, ocr_engine)
