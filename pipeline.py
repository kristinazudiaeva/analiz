import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF
from ultralytics import YOLO

DET_MODEL = YOLO("yolov8n.pt")
SEG_MODEL = YOLO("yolov8n-seg.pt")
CLS_MODEL = YOLO("yolov8n-cls.pt")


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Не удалось прочитать файл: {path}")
    return image


def classify_scene(image: np.ndarray) -> Tuple[str, float]:
    cls_result = CLS_MODEL.predict(image, verbose=False)[0]
    idx = int(cls_result.probs.top1)
    label = cls_result.names[idx]
    score = float(cls_result.probs.top1conf)
    return label, score


def _draw_detections(frame: np.ndarray, det) -> Tuple[np.ndarray, Dict[str, int]]:
    counts = {"chair": 0, "person": 0}
    annotated = frame.copy()
    for box in det.boxes:
        cls_id = int(box.cls[0])
        name = det.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if name == "chair" else (0, 128, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, name, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if name == "chair":
            counts["chair"] += 1
        if name == "person":
            counts["person"] += 1
    return annotated, counts


def _apply_segmentation(frame: np.ndarray, seg) -> np.ndarray:
    annotated = frame.copy()
    if seg.masks is None:
        return annotated

    for mask in seg.masks.data:
        m = mask.cpu().numpy()
        color = np.random.default_rng().integers(0, 255, size=3, dtype=np.uint8)
        h, w = annotated.shape[:2]
        mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(annotated, dtype=np.uint8)
        colored_mask[mask_resized > 0.5] = color
        annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0)
    return annotated


def _compute_occupancy(chairs: int, persons: int, max_chairs: int) -> Tuple[float, str]:
    base = chairs if chairs > 0 else max_chairs
    occupancy = 0.0 if base == 0 else min(persons / base * 100, 100)
    if occupancy < 25:
        level = "Практически пусто"
    elif occupancy < 60:
        level = "Средняя заполненность"
    elif occupancy < 90:
        level = "Высокая заполненность"
    else:
        level = "Полный зал"
    return round(occupancy, 2), level


def process_image(
    input_path: str,
    task: str,
    max_chairs: int,
    result_dir: str,
) -> Tuple[str, Dict]:
    frame = load_image(input_path)
    det_res = DET_MODEL.predict(frame, verbose=False)[0]
    annotated, counts = _draw_detections(frame, det_res)

    if task in {"segmentation", "all"}:
        seg_res = SEG_MODEL.predict(frame, verbose=False)[0]
        annotated = _apply_segmentation(annotated, seg_res)

    label, score = classify_scene(frame)
    occupancy, level = _compute_occupancy(counts["chair"], counts["person"], max_chairs)

    ensure_dirs(result_dir)
    output_path = os.path.join(result_dir, f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(output_path, annotated)

    meta = {
        "chairs": counts["chair"],
        "persons": counts["person"],
        "occupancy_pct": occupancy,
        "level": level,
        "classification": {"label": label, "score": round(score, 4)},
    }
    return output_path, meta


def process_video(
    input_path: str,
    task: str,
    max_chairs: int,
    result_dir: str,
    frame_limit: int = 300,
) -> Tuple[str, Dict]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")

    ensure_dirs(result_dir)
    output_path = os.path.join(result_dir, f"{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    chairs_total = persons_total = frames_processed = 0
    first_frame = None
    while frames_processed < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame is None:
            first_frame = frame.copy()

        det_res = DET_MODEL.predict(frame, verbose=False)[0]
        annotated, counts = _draw_detections(frame, det_res)

        if task in {"segmentation", "all"} and frames_processed % 5 == 0:
            seg_res = SEG_MODEL.predict(frame, verbose=False)[0]
            annotated = _apply_segmentation(annotated, seg_res)

        chairs_total += counts["chair"]
        persons_total += counts["person"]
        frames_processed += 1
        writer.write(annotated)

    cap.release()
    writer.release()

    avg_chairs = round(chairs_total / max(frames_processed, 1), 2)
    avg_persons = round(persons_total / max(frames_processed, 1), 2)
    occupancy, level = _compute_occupancy(avg_chairs, avg_persons, max_chairs)

    label, score = classify_scene(first_frame) if first_frame is not None else ("video_scene", 0.0)

    meta = {
        "chairs": avg_chairs,
        "persons": avg_persons,
        "frames": frames_processed,
        "occupancy_pct": occupancy,
        "level": level,
        "classification": {"label": label, "score": round(score, 4)},
    }
    return output_path, meta


def capture_webcam_frame(target_path: str) -> str:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Камера недоступна")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось получить кадр с камеры")

    ensure_dirs(os.path.dirname(target_path))
    cv2.imwrite(target_path, frame)
    return target_path


def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_history(path: str, data: List[Dict]) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_history(path: str, record: Dict) -> List[Dict]:
    history = load_history(path)
    history.append(record)
    save_history(path, history)
    return history


def compute_summary(history: List[Dict]) -> Dict:
    if not history:
        return {"total": 0, "avg_occupancy": 0, "last_level": "нет данных"}

    occupancies = [item.get("meta", {}).get("occupancy_pct", 0) for item in history]
    last_level = history[-1].get("meta", {}).get("level", "нет данных")
    return {
        "total": len(history),
        "avg_occupancy": round(float(np.mean(occupancies)), 2),
        "last_level": last_level,
    }


def export_history_pdf(history: List[Dict], output_path: str) -> str:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dirs(output_dir)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Hall Occupancy Analysis Report", ln=1)
    pdf.set_font("Arial", "", 10)
    pdf.ln(5)

    if not history:
        pdf.cell(0, 8, "No data available", ln=1)
    else:
        pdf.cell(0, 8, f"Total records: {len(history)}", ln=1)
        pdf.ln(3)
        for idx, item in enumerate(history[-15:], 1):
            meta = item.get("meta", {})
            timestamp = item.get("timestamp", "N/A")
            mode = item.get("mode", "N/A")
            level = meta.get("level", "N/A")
            chairs = meta.get("chairs", 0)
            persons = meta.get("persons", 0)
            occ = meta.get("occupancy_pct", 0)
            
            line = f"{idx}. {timestamp} | Mode: {mode} | Chairs: {chairs} | Persons: {persons} | Occupancy: {occ}%"
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)

    pdf.output(output_path)
    return output_path


def export_history_excel(history: List[Dict], output_path: str) -> str:
    ensure_dirs(os.path.dirname(output_path))
    rows = []
    for item in history:
        meta = item.get("meta", {})
        rows.append(
            {
                "timestamp": item.get("timestamp"),
                "mode": item.get("mode"),
                "task": item.get("task"),
                "chairs": meta.get("chairs"),
                "persons": meta.get("persons"),
                "occupancy_pct": meta.get("occupancy_pct"),
                "level": meta.get("level"),
                "input_path": item.get("input_path"),
                "output_path": item.get("output_path"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return output_path
