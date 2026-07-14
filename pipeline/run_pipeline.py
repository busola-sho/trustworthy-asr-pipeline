"""
pipeline/run_pipeline.py

End-to-end deployment pipeline for trustworthy ASR transcription
in high-stakes policing settings.

Pipeline:
  1. Diarisation     — identify speaker turns (pyannote)
  2. Transcription   — run 3 ASR models ONCE on full audio
  3. Speaker split   — divide transcripts into speaker turns using timestamps
  4. Combination     — ROVER + auditor + dialect pass per turn
  5. NER flagging    — flag names, places, dates for reviewer
  6. Output          — structured JSON with confidence scores

Usage:
    python pipeline/run_pipeline.py --audio interview.wav
    python pipeline/run_pipeline.py --audio interview.wav --speakers 2
    python pipeline/run_pipeline.py --audio interview.wav --no-diarisation
"""

import json
import os
import argparse
import time
import numpy as np
from datetime import datetime
from typing import Optional
from ollama import Client

from pipeline.diarise    import diarise_audio, merge_close_segments, check_diarisation_available
from pipeline.transcribe import transcribe_full_audio, split_by_speaker, load_audio
from pipeline.ner_flag   import flag_named_entities
from src.rover           import rover_combine, risky_disagreements
from src.segmenter       import segment_hypotheses
from src.auditor         import audit_sentence
from src.dialect_pass    import run_dialect_pass
from src.judge           import normalise

OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS  = ["qwen", "whisper", "parakeet"]


def run_combination(
    transcriptions: dict,
    client: Client,
    use_auditor: bool = True,
    use_dialect: bool = True,
) -> dict:
    """Run ROVER + auditor + dialect pass on one speaker turn's transcriptions."""
    hyps = {m: t["hyp"] for m, t in transcriptions.items() if t.get("hyp")}

    if not hyps or all(not v for v in hyps.values()):
        return {
            "transcript":          "",
            "sentence_results":    [],
            "utterance_confidence":0.0,
            "flagged_sentences":   [],
            "dialect_log":         {"n_candidates": 0, "n_approved": 0},
        }

    # rank by mean confidence
    scores = {}
    for model, t in transcriptions.items():
        segs = t.get("segments", [])
        if segs:
            confs = [s["confidence"] for s in segs
                     if isinstance(s, dict) and s.get("confidence") is not None]
            scores[model] = float(np.mean(confs)) if confs else 0.5
        else:
            scores[model] = 0.5

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    ordered_models = [m for m, _ in ranked if m in hyps and hyps[m]]

    if not ordered_models:
        return {
            "transcript":          list(hyps.values())[0] if hyps else "",
            "sentence_results":    [],
            "utterance_confidence":0.5,
            "flagged_sentences":   [],
            "dialect_log":         {"n_candidates": 0, "n_approved": 0},
        }

    # segment and ROVER combine
    segments = segment_hypotheses(hyps, anchor_model="qwen" if "qwen" in hyps else ordered_models[0])

    sentence_transcripts = []
    sentence_results     = []

    for sent_idx, seg in enumerate(segments):
        ordered_hyps = [
            (m, seg[m]) for m in ordered_models
            if m in seg and seg[m].strip()
        ]
        if not ordered_hyps:
            continue

        rover_result = rover_combine(ordered_hyps)
        rover_text   = rover_result.transcript
        n_risky      = len(risky_disagreements(rover_result))

        # auditor
        llm_intervened = False
        if use_auditor:
            model_sentences = {m: seg.get(m, "") for m, _ in ordered_hyps}
            resolved_text, res_log = audit_sentence(
                rover_transcript=rover_text,
                model_sentences=model_sentences,
                word_confidences=rover_result.word_confidences,
                client=client,
            )
            llm_intervened = res_log.get("called_llm", False)
        else:
            resolved_text = rover_text

        word_confs  = [c for _, c in rover_result.word_confidences]
        base_conf   = float(np.mean(word_confs)) if word_confs else 0.5
        penalty     = 0.05 * n_risky + (0.05 if llm_intervened else 0.0)
        sent_conf   = max(0.0, base_conf - penalty)

        sentence_transcripts.append(resolved_text)
        sentence_results.append({
            "sentence_idx":        sent_idx,
            "transcript":          resolved_text,
            "sentence_confidence": round(sent_conf, 4),
            "n_disagreements":     len(rover_result.disagreements),
            "n_risky":             n_risky,
            "llm_intervened":      llm_intervened,
            "flagged":             sent_conf < 0.7,
        })

    full_transcript = " ".join(sentence_transcripts).strip()

    # dialect pass
    dialect_log = {"n_candidates": 0, "n_approved": 0}
    if use_dialect and full_transcript:
        full_transcript, dialect_log = run_dialect_pass(full_transcript, client)

    sentence_confs    = [s["sentence_confidence"] for s in sentence_results]
    utterance_conf    = float(np.mean(sentence_confs)) if sentence_confs else 0.0
    flagged_sentences = [s["sentence_idx"] for s in sentence_results if s["flagged"]]

    return {
        "transcript":          full_transcript,
        "sentence_results":    sentence_results,
        "utterance_confidence":round(utterance_conf, 4),
        "flagged_sentences":   flagged_sentences,
        "dialect_log":         dialect_log,
    }


def run_pipeline(
    audio_path:      str,
    output_path:     Optional[str] = None,
    num_speakers:    Optional[int] = None,
    use_diarisation: bool = True,
    use_auditor:     bool = True,
    use_dialect:     bool = True,
    asr_models:      Optional[list] = None,
) -> dict:
    """Full end-to-end pipeline."""
    if asr_models is None:
        asr_models = ASR_MODELS

    client     = Client(host=OLLAMA_HOST)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"PIPELINE: {audio_path}")
    print(f"{'='*60}")

    result = {
        "audio_path":      audio_path,
        "timestamp":       datetime.now().isoformat(),
        "pipeline_config": {
            "use_diarisation": use_diarisation,
            "use_auditor":     use_auditor,
            "use_dialect":     use_dialect,
            "asr_models":      asr_models,
        },
        "speakers":           [],
        "full_transcript":    "",
        "ner_flags":          [],
        "processing_time_s":  None,
    }

    # ── Step 1: Diarisation ───────────────────────────────────────────────────
    if use_diarisation:
        print("\n[1/4] Diarisation...")
        raw_segs   = diarise_audio(audio_path, num_speakers=num_speakers)
        diar_segs  = merge_close_segments(raw_segs)
        n_speakers = len(set(s["speaker"] for s in diar_segs))
        print(f"  {n_speakers} speakers, {len(diar_segs)} turns")
    else:
        audio, sr = load_audio(audio_path)
        diar_segs = [{"speaker": "SPEAKER_00", "start": 0.0, "end": len(audio) / sr}]
        print("\n[1/4] Diarisation skipped — single speaker mode")

    # ── Step 2: Transcribe full audio ONCE per model ──────────────────────────
    print("\n[2/4] Transcription (full audio, once per model)...")
    model_outputs = transcribe_full_audio(audio_path, models=asr_models)

    # ── Step 3: Split into speaker turns ─────────────────────────────────────
    print("\n[3/4] Splitting into speaker turns...")
    turns = split_by_speaker(model_outputs, diar_segs)
    print(f"  {len(turns)} turns to process")

    # ── Step 4: Combine + audit per turn ─────────────────────────────────────
    speaker_results = []
    for i, turn in enumerate(turns):
        speaker = turn["speaker"]
        start   = turn["start"]
        end     = turn["end"]
        print(f"\n  Turn {i+1}/{len(turns)}: {speaker} ({start:.1f}s-{end:.1f}s)")

        combination = run_combination(
            turn["transcriptions"], client,
            use_auditor=use_auditor,
            use_dialect=use_dialect,
        )

        speaker_results.append({
            "speaker":             speaker,
            "start":               start,
            "end":                 end,
            "transcript":          combination["transcript"],
            "utterance_confidence":combination["utterance_confidence"],
            "flagged_sentences":   combination["flagged_sentences"],
            "sentence_results":    combination["sentence_results"],
            "model_transcripts": {
                m: {"hyp": t["hyp"]}
                for m, t in turn["transcriptions"].items()
            },
        })

        print(f"  → {combination['transcript'][:100]}")
        print(f"  → conf={combination['utterance_confidence']:.2f} "
              f"flagged={combination['flagged_sentences']}")

    result["speakers"] = speaker_results

    # ── Step 5: Full transcript + NER ────────────────────────────────────────
    print("\n[4/4] NER flagging...")
    full_lines = [
        f"{t['speaker']}: {t['transcript']}"
        for t in speaker_results if t["transcript"]
    ]
    result["full_transcript"] = "\n".join(full_lines)

    full_text = " ".join(t["transcript"] for t in speaker_results)
    ner_flags = flag_named_entities(full_text)
    result["ner_flags"] = ner_flags
    print(f"  {len(ner_flags)} entities flagged")
    for f in ner_flags[:5]:
        print(f"  [{f['label']}] {f['text']}")

    elapsed                   = time.time() - start_time
    result["processing_time_s"] = round(elapsed, 1)

    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"\nFull transcript:")
    print(result["full_transcript"][:500])
    print(f"{'='*60}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Trustworthy ASR pipeline for high-stakes policing settings"
    )
    parser.add_argument("--audio",          required=True)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--speakers",       type=int, default=None)
    parser.add_argument("--no-diarisation", action="store_true")
    parser.add_argument("--no-auditor",     action="store_true")
    parser.add_argument("--no-dialect",     action="store_true")
    parser.add_argument("--models",         nargs="+", default=ASR_MODELS)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        return

    output_path = args.output
    if output_path is None:
        base        = os.path.splitext(os.path.basename(args.audio))[0]
        output_path = f"pipeline_output/{base}_result.json"

    run_pipeline(
        audio_path=args.audio,
        output_path=output_path,
        num_speakers=args.speakers,
        use_diarisation=not args.no_diarisation,
        use_auditor=not args.no_auditor,
        use_dialect=not args.no_dialect,
        asr_models=args.models,
    )


if __name__ == "__main__":
    main()