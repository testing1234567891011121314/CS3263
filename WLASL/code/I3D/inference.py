#!/usr/bin/env python3
"""
Simple inference script for WLASL I3D model on user videos.
Usage: python inference.py path/to/your/video.mp4

@author: celeschai
@date: 2025-10-10
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import videotransforms
from pytorch_i3d import InceptionI3d

def load_class_labels(class_file='preprocess/wlasl_class_list.txt'):
    """Load class ID to word mapping"""
    class_labels = {}
    with open(class_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                class_id = int(parts[0])
                word = parts[1]
                class_labels[class_id] = word
    return class_labels

def load_rgb_frames_from_video(video_path, start=0, num=-1):
    """Load frames from video file -> torch.Tensor [T, H, W, C] in [-1, 1]."""
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(num):
        ok, img = vidcap.read()
        if not ok:
            break
        # cv2 returns (H, W, C); resize to 224x224 explicitly
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # BGR->RGB if your model expects RGB (I3D usually does)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize to [-1, 1]
        img = (img.astype(np.float32) / 255.0) * 2.0 - 1.0
        frames.append(img)

    vidcap.release()
    if not frames:
        return torch.empty(0)
    return torch.from_numpy(np.stack(frames, axis=0))  # [T, H, W, C]


def predict_video(video_path, model_weights, num_classes=2000, top_k=5):
    """
    Run inference on a single video and return top-k (word, prob, class_id).
    Handles model outputs of shape [B, C] or [B, C, T].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_labels = load_class_labels()

    print(f"Loading video: {video_path}")
    frames = load_rgb_frames_from_video(video_path)
    if frames.numel() == 0:
        raise ValueError("No frames loaded from video")
    print(f"Loaded {frames.shape[0]} frames")

    # [T, H, W, C] -> [B, C, T, H, W]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device)

    print("Loading model...")
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    state = torch.load(model_weights, map_location=device)
    i3d.load_state_dict(state, strict=False)
    i3d.to(device)
    i3d.eval()

    print("Running inference...")
    with torch.no_grad():
        out = i3d(frames)  # expected [B, C] or [B, C, T]

        # Reduce to [B, C]
        if out.dim() == 3:
            # average over time dimension
            logits = out.mean(dim=2)
        elif out.dim() == 2:
            logits = out
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(out.shape)}")

        probs = torch.softmax(logits, dim=1)  # [B, C]
        top_probs, top_idxs = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)

        # Convert first (and only) batch element to Python lists
        probs_list = top_probs[0].detach().cpu().tolist()   # length K
        idxs_list  = top_idxs[0].detach().cpu().tolist()    # length K

    results = []
    for p, cid in zip(probs_list, idxs_list):
        word = class_labels.get(cid, f"Unknown_{cid}")
        results.append((word, float(p), int(cid)))
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/your/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Model configuration - you can change these
    model_weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    num_classes = 100
    
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found: {model_weights}")
        print("Please download the pretrained weights first.")
        sys.exit(1)
    
    try:
        results = predict_video(video_path, model_weights, num_classes, top_k=5)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        for i, (word, prob, class_id) in enumerate(results, 1):
            print(f"{i}. {word} (Class {class_id}) - {prob:.3f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()