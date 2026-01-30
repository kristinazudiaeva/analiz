import os
import uuid
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from pipeline import (
    append_history,
    capture_webcam_frame,
    compute_summary,
    export_history_excel,
    export_history_pdf,
    load_history,
    process_image,
    process_video,
)

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def _make_record(
    mode: str,
    task: str,
    input_path: str,
    output_path: str,
    meta: dict,
) -> dict:
    rel_input = os.path.relpath(input_path, BASE_DIR).replace("\\", "/")
    rel_output = os.path.relpath(output_path, BASE_DIR).replace("\\", "/")
    return {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "task": task,
        "input_path": rel_input,
        "output_path": rel_output,
        "meta": meta,
    }


@app.route("/", methods=["GET"])
def index():
    history = load_history(HISTORY_PATH)
    summary = compute_summary(history)
    return render_template("index.html", history=list(reversed(history[-20:])), summary=summary, result=None)


def _handle_uploaded_file(field_name: str, target_dir: str) -> Optional[str]:
    file = request.files.get(field_name)
    if not file or file.filename == "":
        return None
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    target = os.path.join(target_dir, f"{uuid.uuid4().hex}{ext}")
    file.save(target)
    return target


@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode", "image")
    task = request.form.get("task", "all")

    max_chairs_raw = request.form.get("max_chairs", "").strip()
    if not max_chairs_raw.isdigit() or int(max_chairs_raw) <= 0:
        error = "Укажите положительное число стульев."
        history = load_history(HISTORY_PATH)
        summary = compute_summary(history)
        return render_template("index.html", history=list(reversed(history[-20:])), summary=summary, error=error, result=None)
    max_chairs = int(max_chairs_raw)

    try:
        if mode == "image":
            input_path = _handle_uploaded_file("file", UPLOAD_DIR)
            if not input_path:
                return redirect(url_for("index"))
            output_path, meta = process_image(input_path, task, max_chairs, RESULT_DIR)
        elif mode == "video":
            input_path = _handle_uploaded_file("file", UPLOAD_DIR)
            if not input_path:
                return redirect(url_for("index"))
            output_path, meta = process_video(input_path, task, max_chairs, RESULT_DIR)
        elif mode == "webcam":
            raw_path = os.path.join(RESULT_DIR, f"{uuid.uuid4().hex}_webcam.jpg")
            input_path = capture_webcam_frame(raw_path)
            output_path, meta = process_image(input_path, task, max_chairs, RESULT_DIR)
        else:
            return redirect(url_for("index"))
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", history=load_history(HISTORY_PATH), summary=compute_summary(load_history(HISTORY_PATH)), error=str(exc), result=None)

    record = _make_record(mode, task, input_path, output_path, meta)
    history = append_history(HISTORY_PATH, record)
    summary = compute_summary(history)

    return render_template(
        "index.html",
        history=list(reversed(history[-20:])),
        summary=summary,
        result=record,
    )


@app.route("/history.json")
def history_json():
    return jsonify(load_history(HISTORY_PATH))


@app.route("/report/pdf")
def report_pdf():
    try:
        history = load_history(HISTORY_PATH)
        pdf_path = os.path.join(DATA_DIR, "report.pdf")
        export_history_pdf(history, pdf_path)
        if not os.path.exists(pdf_path):
            return "Error: PDF file was not created", 500
        return send_file(pdf_path, as_attachment=True, mimetype="application/pdf")
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500


@app.route("/report/excel")
def report_excel():
    try:
        history = load_history(HISTORY_PATH)
        excel_path = os.path.join(DATA_DIR, "history.xlsx")
        export_history_excel(history, excel_path)
        if not os.path.exists(excel_path):
            return "Error: Excel file was not created", 500
        return send_file(excel_path, as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        return f"Error generating Excel: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
