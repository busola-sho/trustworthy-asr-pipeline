"""One display model for the terminal and HTML walkthrough; raw outputs stay intact."""

import html
import textwrap

from .config import SOURCE_MODELS
from .files import read_json


METHOD_NAMES = {
    "qwen": "Qwen3-ASR",
    "whisperx": "WhisperX",
    "parakeet": "Parakeet",
    "wav2vec2": "Wav2Vec2",
    "selection": "Selection",
    "anchored": "Anchored fusion",
    "unanchored": "Unanchored fusion",
    "rover": "ROVER",
    "mbr": "MBR",
}


def word_error_rate(reference, hypothesis):
    """Use the pipeline's existing lexical normalization for display-only WER."""
    from jiwer import wer
    from .text_normalization import is_tag_only, normalise

    if not reference or is_tag_only(reference):
        return None
    return wer(normalise(reference), normalise(hypothesis or ""))


def sentence_outputs(path):
    return {
        transcript["dataset_index"]: transcript["segments"]
        for transcript in read_json(path)["transcripts"]
    }


def matching_sentence(outputs, transcript_index, position, text):
    segment = outputs[transcript_index][position]
    # Repeated sentences can occur: join by transcript ID AND position, then check text.
    if segment["segment"] != text:
        raise ValueError(
            "Sentence outputs disagree; refusing to display misaligned scores"
        )
    return segment


def build_results_summary(root, combined_files, evaluated_files, reference):
    asr_rows = []
    for model in SOURCE_MODELS:
        sample = read_json(root / "asr" / f"{model}.json")["samples"][0]
        asr_rows.append(
            {
                "method": model,
                "transcript": sample["hyp"],
                "wer": word_error_rate(reference, sample["hyp"]),
                "severity": None,  # Native ASR hypotheses have not been severity-judged.
            }
        )

    combination_rows = []
    for method, combined_file in combined_files.items():
        data = read_json(evaluated_files.get(method, combined_file))
        sample = data["samples"][0]
        combination_rows.append(
            {
                "method": method,
                "transcript": sample["hyp"],
                "wer": data.get("corpus_wer"),
                "severity": sample.get("severity"),
                "judge_raw_response": sample.get("judge_raw_response"),
            }
        )

    confidence_dir = root / "confidence"
    segmentation = read_json(
        confidence_dir / "segments/sentence_segments_clip_full.json"
    )
    verbalized = sentence_outputs(
        confidence_dir / "verbalized_confidence/verbalized_confscore_clip_full.json"
    )
    agreement = sentence_outputs(
        confidence_dir / "cross_model_agreement/cross_model_agreement_clip_full.json"
    )
    evaluation_path = root / "evaluation/segments/sentence_segments_clip_full.json"
    evaluated_sentences = (
        sentence_outputs(evaluation_path) if evaluation_path.exists() else None
    )
    sentence_rows = []
    for transcript in segmentation["transcripts"]:
        transcript_index = transcript["dataset_index"]
        for position, segment in enumerate(transcript["segments"]):
            text = segment["segment"]
            confidence = matching_sentence(verbalized, transcript_index, position, text)
            agreements = matching_sentence(agreement, transcript_index, position, text)
            evaluation = {}
            if evaluated_sentences is not None:
                evaluation = matching_sentence(
                    evaluated_sentences, transcript_index, position, text
                )
            sentence_rows.append(
                {
                    "dataset_index": transcript_index,
                    "position": position,
                    "sentence": text,
                    "verbalized_score": confidence.get("verbalized_score"),
                    "agreement_mean": agreements.get("crossmodel_mean"),
                    "agreement_min": agreements.get("crossmodel_min"),
                    "wer": word_error_rate(evaluation.get("corresponding_span"), text),
                    "severity": evaluation.get("severity"),
                    "judge_raw_response": evaluation.get("raw_judge_response"),
                }
            )
    return {
        "reference": reference,
        "asr": asr_rows,
        "combination": combination_rows,
        "sentences": sentence_rows,
    }


def format_metric(value, kind):
    if value is None:
        return "—"
    if kind == "wer":
        return f"{value:.1%}"
    if kind == "verbalized_score":
        return f"{value:g}/100"
    if kind.startswith("agreement"):
        return f"{value:.4f}"
    return str(value)


def render_terminal_summary(results, unavailable):
    lines = ["", "ONE-CLIP RESULTS", "=" * 72]
    if results["reference"] is not None:
        lines.extend(["Reference:", textwrap.fill(results["reference"], width=88), ""])
    else:
        lines.extend(
            ["Reference: not supplied; reference-based evaluation skipped.", ""]
        )
    for heading, group in (
        ("ASR TRANSCRIPTS", "asr"),
        ("COMBINATION OUTPUTS", "combination"),
    ):
        lines.append(heading)
        for row in results[group]:
            name = METHOD_NAMES[row["method"]]
            lines.append(
                f"  {name} | WER {format_metric(row['wer'], 'wer')} | severity {format_metric(row['severity'], 'severity')}"
            )
            lines.append(
                textwrap.fill(
                    row["transcript"] or "[empty transcript]",
                    width=88,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        lines.append("")
    lines.append("SENTENCES — UNANCHORED FUSION")
    for number, row in enumerate(results["sentences"], start=1):
        lines.append(textwrap.fill(f"{number}. {row['sentence']}", width=88))
        lines.append(
            f"   Verbalised confidence: {format_metric(row['verbalized_score'], 'verbalized_score')} | "
            f"agreement mean: {format_metric(row['agreement_mean'], 'agreement_mean')} | "
            f"minimum: {format_metric(row['agreement_min'], 'agreement_min')}"
        )
        lines.append(
            f"   WER {format_metric(row['wer'], 'wer')} | severity {format_metric(row['severity'], 'severity')}"
        )
    return "\n".join(lines) + "\n"


def transcript_table(rows):
    table = "<table><thead><tr><th>Method</th><th>Transcript</th><th>WER</th><th>Severity</th></tr></thead><tbody>"
    for row in rows:
        table += (
            f"<tr><th scope='row'>{html.escape(METHOD_NAMES[row['method']])}</th>"
            f"<td>{html.escape(row['transcript'] or '[empty transcript]')}</td>"
            f"<td>{format_metric(row['wer'], 'wer')}</td>"
            f"<td>{format_metric(row['severity'], 'severity')}</td></tr>"
        )
    return table + "</tbody></table>"


def render_html_summary(results, summary, audio_path, root):
    content = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One-clip pipeline results</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;color:#182633;background:#f6f8fa;margin:0}
main{max-width:1120px;margin:28px auto;padding:28px;background:white;border:1px solid #dbe3e9;border-radius:12px}
h1{font-size:28px;margin:0 0 12px}h2{font-size:20px;margin:24px 0 10px}
audio{width:100%;height:40px}table{border-collapse:collapse;width:100%;margin:10px 0}
th,td{border:1px solid #dbe3e9;padding:10px;text-align:left;vertical-align:top}
thead th{background:#edf3f8}tbody th{min-width:140px;font-weight:600}td{overflow-wrap:anywhere}
.reference,.note{background:#edf3f8;padding:12px 16px;border-radius:6px}.note{background:#fff5dc;font-size:14px}
.metrics{display:flex;gap:12px;flex-wrap:wrap;font-size:14px;color:#374b5f}.sentence{border:1px solid #dbe3e9;padding:14px;border-radius:6px}
.sentence p{margin:0 0 8px}details{margin:10px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:12px;font-size:13px}
a{color:#165b99}.table-wrap{overflow:auto}
</style></head><body><main><h1>One-clip pipeline results</h1>"""
    content += f'<audio controls src="input/{html.escape(audio_path.name, quote=True)}"></audio>'
    if results["reference"] is not None:
        content += f'<h2>Reference</h2><p class="reference">{html.escape(results["reference"])}</p>'
    else:
        content += '<p class="reference">Reference not supplied; reference-based evaluation skipped.</p>'
    for heading, group in (
        ("ASR transcripts", "asr"),
        ("Combination outputs", "combination"),
    ):
        content += f'<h2>{heading}</h2><div class="table-wrap">{transcript_table(results[group])}</div>'
    content += "<h2>Sentences — unanchored fusion</h2>"
    for number, row in enumerate(results["sentences"], start=1):
        content += f'<section class="sentence"><p><b>{number}.</b> {html.escape(row["sentence"])}</p><div class="metrics">'
        for label, key in (
            ("Verbalised confidence", "verbalized_score"),
            ("Agreement mean", "agreement_mean"),
            ("Agreement minimum", "agreement_min"),
            ("WER", "wer"),
            ("Severity", "severity"),
        ):
            content += f"<span><b>{label}:</b> {format_metric(row[key], key)}</span>"
        content += "</div></section>"
    content += '<p class="note">— = not available / not evaluated. ASR severity was not judged. Verbalised confidence is not a calibrated probability. Judge scores are unvalidated; raw responses, including contradictory explanations, are preserved.</p>'
    content += "<h2>Unavailable</h2><ul>"
    for method, reason in summary["unavailable"].items():
        content += f"<li><b>{html.escape(method.replace('_', ' '))}:</b> {html.escape(reason)}</li>"
    content += "</ul><h2>Inspect inputs and raw outputs</h2>"
    # Include run.json even though main writes it after this display stage returns.
    files = (
        set(root.rglob("*.json"))
        | set(root.rglob("*.jsonl"))
        | set((root / "logs").glob("*.log"))
    )
    files.update({root / "run.json", root / "summary.txt"})
    for path in sorted(files):
        relative_path = path.relative_to(root).as_posix()
        link = html.escape(relative_path, quote=True)
        content += f'<details><summary>{html.escape(relative_path)}</summary><p><a href="{link}">Open file</a></p>'
        if path.exists():
            content += (
                "<pre>"
                + html.escape(path.read_text(encoding="utf-8", errors="replace"))
                + "</pre>"
            )
        content += "</details>"
    return content + "</main></body></html>\n"
