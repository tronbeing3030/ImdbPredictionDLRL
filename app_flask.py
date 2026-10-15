"""IMDb Cinema Sentiment Lab — Flask web app.

Section 1 (Studio): abstract input/output form, no internals.
Section 2 (Lab): step-by-step simulation of how the verdict was produced.
Section 3 (Arena + Report): model comparison, sequence study, gradients, write-up.
"""
import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify, redirect, url_for

from src.inference import predict as backend_predict

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

app = Flask(__name__)
app.secret_key = "imdb-cinema-lab"

EXAMPLES = {
    "glowing": "This movie was absolutely brilliant. The acting was superb, the story was moving and the direction was fantastic. I loved every minute of it.",
    "harsh": "Terrible movie. Boring plot, awful acting and a complete waste of time. I hated every dull scene.",
    "mixed": "The cinematography was great but the plot was weak and the pacing dull. Some scenes shone, others dragged.",
}

_last_result = {"result": None, "text": ""}


def load_metrics():
    p = MODEL_DIR / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", examples=EXAMPLES, result=_last_result["result"],
                           text=_last_result["text"], metrics=load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    text = (request.form.get("review") or "").strip()
    example = request.form.get("example") or ""
    if not text and example in EXAMPLES:
        text = EXAMPLES[example]
    if not text:
        _last_result.update(result={"error": "Please enter a review first."}, text="")
        return redirect(url_for("index") + "#studio")
    try:
        r = backend_predict(text)
    except Exception as e:
        _last_result.update(result={"error": f"Prediction failed: {e}"}, text=text)
        return redirect(url_for("index") + "#studio")
    _last_result.update(result=r, text=text)
    return redirect(url_for("index") + "#studio")


@app.route("/lab", methods=["GET"])
def lab():
    r = _last_result["result"]
    if not r or "error" in r:
        return redirect(url_for("index") + "#studio")
    return render_template("lab.html", result=r, text=_last_result["text"])


@app.route("/arena", methods=["GET"])
def arena():
    return render_template("arena.html", metrics=load_metrics())


@app.route("/report", methods=["GET"])
def report():
    rp = BASE / "PROJECT_REPORT.md"
    txt = rp.read_text(encoding="utf-8") if rp.exists() else "(PROJECT_REPORT.md not found)"
    return render_template("report.html", metrics=load_metrics(), report_text=txt)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("review") or "").strip()
    if not text:
        return jsonify({"error": "empty review"}), 400
    try:
        r = backend_predict(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    r = dict(r)
    r.pop("metrics", None)
    r["padded"] = r.get("padded", [])[:60]
    return jsonify(r)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
