"""
sentiment-pro CLI: Analyze text sentiment from command line.

Usage:
    sentiment-pro analyze "text here"
    sentiment-pro analyze "text here" --json
    sentiment-pro analyze "text here" --aspects
    sentiment-pro batch input.txt
    sentiment-pro batch input.txt --json --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import SentimentAnalyzerPro, SentimentResult


def _format_result(result: SentimentResult, fmt: str, show_aspects: bool, show_emotions: bool) -> str:
    if fmt == "json":
        return result.to_json()

    # Human-readable format
    lines = []
    sentiment_icon = {"positive": "✓", "negative": "✗", "neutral": "~"}.get(result.sentiment, "?")
    lines.append(f"Sentiment : {sentiment_icon} {result.sentiment.upper()} (score: {result.score:+.3f}, confidence: {result.confidence:.2f})")
    lines.append(f"Lexicon   : pos={result.lexicon.positive:.2f}, neg={result.lexicon.negative:.2f}, neu={result.lexicon.neutral:.2f}")

    if result.primary_emotion:
        emo_line = f"Emotion   : {result.primary_emotion}"
        if result.secondary_emotion:
            emo_line += f" / {result.secondary_emotion}"
        emo_line += f" (valence: {result.emotion_valence})"
        lines.append(emo_line)
    else:
        lines.append("Emotion   : (none detected)")

    if show_emotions and result.emotion_scores:
        top_emos = sorted(result.emotion_scores.items(), key=lambda x: -x[1])[:5]
        emo_parts = [f"{e}={s:.2f}" for e, s in top_emos if s > 0]
        if emo_parts:
            lines.append(f"  Scores  : {', '.join(emo_parts)}")

    if show_aspects or result.aspects:
        if result.aspects:
            lines.append("Aspects   :")
            for aspect, score in sorted(result.aspects.items(), key=lambda x: -abs(x[1])):
                label = result.aspect_labels.get(aspect, "?")
                icon = "↑" if score >= 0.05 else "↓" if score <= -0.05 else "~"
                lines.append(f"  {icon} {aspect:<22} {score:+.3f}  ({label})")
        else:
            if show_aspects:
                lines.append("Aspects   : (none detected)")

    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace, analyzer: SentimentAnalyzerPro) -> int:
    result = analyzer.analyze(args.text)
    print(_format_result(result, args.format, args.aspects, args.emotions))
    return 0


def cmd_batch(args: argparse.Namespace, analyzer: SentimentAnalyzerPro) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    lines = [l.strip() for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print("Error: input file is empty", file=sys.stderr)
        return 1

    results = analyzer.analyze_batch(lines)

    out_lines = []
    for i, (text, result) in enumerate(zip(lines, results), 1):
        if args.format == "json":
            out_lines.append(json.dumps({"line": i, **result.to_dict()}))
        else:
            out_lines.append(f"[{i}] {text[:80]}{'...' if len(text) > 80 else ''}")
            out_lines.append(_format_result(result, "text", args.aspects, args.emotions))
            out_lines.append("")

    output = "\n".join(out_lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Results written to {args.out} ({len(results)} texts analyzed)")
    else:
        print(output)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentiment-pro",
        description="Multi-level sentiment analysis: lexicon, aspects, and emotions.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Analyze a single text")
    p_analyze.add_argument("text", help="Text to analyze")
    p_analyze.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    p_analyze.add_argument("--aspects", "-a", action="store_true", help="Show aspect breakdown")
    p_analyze.add_argument("--emotions", "-e", action="store_true", help="Show emotion scores")
    p_analyze.add_argument("--json", dest="format", action="store_const", const="json",
                           help="Shorthand for --format json")

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Analyze a file of texts (one per line)")
    p_batch.add_argument("input", help="Input file path (one text per line)")
    p_batch.add_argument("--out", "-o", help="Write output to file instead of stdout")
    p_batch.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    p_batch.add_argument("--aspects", "-a", action="store_true", help="Show aspect breakdown")
    p_batch.add_argument("--emotions", "-e", action="store_true", help="Show emotion scores")
    p_batch.add_argument("--json", dest="format", action="store_const", const="json",
                         help="Shorthand for --format json")

    args = parser.parse_args(argv)
    analyzer = SentimentAnalyzerPro()

    if args.command == "analyze":
        return cmd_analyze(args, analyzer)
    elif args.command == "batch":
        return cmd_batch(args, analyzer)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
