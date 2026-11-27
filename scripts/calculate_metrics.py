"""Calculate the six agreed metrics in their fixed order, comparing models to human.

Run from any directory: python scripts/calculate_metrics.py
Install dependencies in the active environment, or into .runtime/metrics:
    python -m pip install --target .runtime/metrics -r requirements-analysis.txt

The input audit is refreshed first. Original corpus files are never modified.
Sentence boundaries are retained for both analysis levels. BH correction is
applied once to all 30 two-sided Welch tests, after all metrics are calculated.
"""

import argparse
import contextlib
import csv
import html
import io
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPENDENCIES = ROOT / ".runtime/metrics"
if LOCAL_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))

try:
    import numpy as np
    import scipy
    from scipy import stats
except ImportError as error:
    raise SystemExit(
        "Install analysis dependencies: python -m pip install --target "
        ".runtime/metrics -r requirements-analysis.txt"
    ) from error

import audit_analysis_inputs as audit

METRICS = tuple(audit.METRICS)
FAMILIES = ("human", "gemma", "qwen", "llama", "deepseek", "gpt")
MODEL_NAMES = {
    "human": "human", "gemma": "gemma3:27b", "qwen": "qwen3:32b",
    "llama": "llama3:70b", "deepseek": "deepseek-llm:67b", "gpt": "gpt-6-luna",
}
LABELS = dict(zip(METRICS, (
    "M-Index", "I-Index", "Burstiness", "Span Entropy",
    "Mean English Span Length", "Single-word English Span Proportion",
)))
FORMULAS = dict(zip(METRICS, (
    "2CE / (C² + E²)", "S / (N − 1)",
    "(σ − μ) / (σ + μ)，span 长度的 σ 使用 ddof=0",
    "−Σ pₗ log₂(pₗ)，单位 bits",
    "英文词数 / 英文 span 数", "长度为 1 的英文 span 数 / 英文 span 数",
)))
COMPARISON_FIELDS = [
    "metric", "family", "model", "n_valid", "n_na", "sentence_mean",
    "sentence_sd_ddof0", "sentence_mean_difference_model_minus_human",
    "welch_t_model_minus_human", "welch_df", "p_raw_two_sided", "p_bh_30",
    "significant_bh_0_05", "corpus_value", "corpus_difference_model_minus_human",
    "human_n_valid", "human_sentence_mean", "human_sentence_sd_ddof0",
]


def sentence_statistics(sentence):
    """Extract runs within ONE input row; separators have no language label."""
    labels = [kind for kind, _, _, _ in audit.scan(sentence) if kind in {"C", "E"}]
    chinese, english = labels.count("C"), labels.count("E")
    if not chinese or not english:
        raise ValueError("Only valid bilingual rows may enter metric calculations")
    spans = [(language, len(list(tokens))) for language, tokens in groupby(labels)]
    return {
        "chinese_characters": chinese, "english_words": english,
        "effective_tokens": len(labels), "switches": len(spans) - 1,
        "adjacent_positions": len(labels) - 1,
        "span_languages": [language for language, _ in spans],
        "span_lengths": [length for _, length in spans],
        "english_span_lengths": [length for language, length in spans if language == "E"],
        "chinese_span_lengths": [length for language, length in spans if language == "C"],
    }


def pool_statistics(sentences):
    """Pool counts and ALREADY DELIMITED spans, never token sequences."""
    sentences = list(sentences)
    if not sentences:
        raise ValueError("No valid sentences to pool")
    result = {field: sum(sentence[field] for sentence in sentences) for field in (
        "chinese_characters", "english_words", "effective_tokens",
        "switches", "adjacent_positions",
    )}
    for field in ("span_languages", "span_lengths", "english_span_lengths", "chinese_span_lengths"):
        result[field] = [value for sentence in sentences for value in sentence[field]]
    if result["adjacent_positions"] != result["effective_tokens"] - len(sentences):
        raise ValueError("Sentence boundaries were lost from the I-Index denominator")
    if len(result["span_lengths"]) != result["switches"] + len(sentences):
        raise ValueError("Sentence boundaries were lost from the span statistics")
    return result


def metric_values(counts):
    chinese, english = counts["chinese_characters"], counts["english_words"]
    lengths = counts["span_lengths"]
    english_lengths = counts["english_span_lengths"]
    if min(chinese, english, counts["adjacent_positions"], len(lengths), len(english_lengths)) <= 0:
        raise ValueError("Undefined metric on an empty or non-bilingual input")
    mean, sd = statistics.mean(lengths), statistics.pstdev(lengths)
    frequencies = Counter(lengths)
    values = {
        "m_index": 2 * chinese * english / (chinese ** 2 + english ** 2),
        "i_index": counts["switches"] / counts["adjacent_positions"],
        "burstiness": (sd - mean) / (sd + mean),
        "span_entropy": -sum((count / len(lengths)) * math.log2(count / len(lengths))
                             for count in frequencies.values()),
        "mean_english_span_length": english / len(english_lengths),
        "single_word_english_span_proportion": english_lengths.count(1) / len(english_lengths),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite metric value")
    return values


def descriptive_summary(values):
    values = list(values)
    return {"n_valid": len(values), "mean": statistics.mean(values),
            "sd_ddof0": statistics.pstdev(values)}


def welch_comparison(model_values, human_values):
    """SciPy uses sample variances for Welch; descriptive SD stays ddof=0."""
    result = stats.ttest_ind(model_values, human_values, equal_var=False,
                             alternative="two-sided", nan_policy="raise")
    values = {"welch_t_model_minus_human": float(result.statistic),
              "welch_df": float(result.df), "p_raw_two_sided": float(result.pvalue)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Undefined Welch test; inspect the metric distributions")
    return values


def adjust_bh(p_values):
    return stats.false_discovery_control(np.asarray(p_values, dtype=float), method="bh").tolist()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs():
    """Refresh existing exclusions, then check input identity and audit consistency."""
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main()
    report = json.loads((audit.OUT / "summary.json").read_text(encoding="utf-8"))
    rows = audit.read_csv(audit.OUT / "sentence_audit.csv")
    valid = [row for row in rows if row["analysis_status"] == "valid"]
    if valid != audit.read_csv(audit.OUT / "analysis_valid_rows.csv"):
        raise ValueError("Valid-row CSV differs from the row audit")
    summaries = {item["family"]: item for item in report["summary"]}
    for family in FAMILIES:
        group = [row for row in rows if row["family"] == family]
        included = [row for row in group if row["analysis_status"] == "valid"]
        if len(group) != summaries[family]["rows"] or len(included) != summaries[family]["valid_analysis_rows"]:
            raise ValueError(f"Audit row counts differ: {family}")
        for row in group:
            if row["model"] != MODEL_NAMES[family]:
                raise ValueError(f"Unexpected model identifier: {family}")
            if row["analysis_status"] not in {"valid", "NA"}:
                raise ValueError("Unresolved input row")
            if (row["analysis_status"] == "valid") != (row["included_in_analysis"] == "True"):
                raise ValueError("Inconsistent inclusion flag")
            if row["analysis_status"] == "NA" and any(row[metric] != "NA" for metric in METRICS):
                raise ValueError("Excluded row has non-NA metrics")
    input_paths = [path for _, path in audit.SOURCES] + [
        ROOT / "data/research_sentences.csv", audit.LANGUAGE_REVIEW,
        ROOT / "scripts/audit_analysis_inputs.py", Path(__file__).resolve(),
        ROOT / "requirements-analysis.txt", audit.OUT / "sentence_audit.csv",
        audit.OUT / "analysis_valid_rows.csv", audit.OUT / "summary.json",
    ]
    hashes = {path.relative_to(ROOT).as_posix(): audit.digest(path) for path in input_paths}
    return rows, summaries, hashes


def compute_results(rows, audit_summaries):
    sentence_rows, grouped_counts, grouped_values = [], defaultdict(list), defaultdict(list)
    for row in rows:
        record = dict(row)
        if row["analysis_status"] == "valid":
            counts = sentence_statistics(row["sentence"])
            values = metric_values(counts)
            for metric in METRICS:
                if not math.isclose(values[metric], float(row[metric]), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"Sentence metric differs from audit: {row['family']}, {row['data_row']}, {metric}")
                record[metric] = values[metric]
            for field in ("chinese_characters", "english_words", "effective_tokens"):
                if counts[field] != int(row[field]):
                    raise ValueError("Token counts differ from the input audit")
            record.update({field: counts[field] for field in ("switches", "adjacent_positions")})
            for field in ("span_languages", "span_lengths", "english_span_lengths"):
                record[field] = json.dumps(counts[field], ensure_ascii=False)
            grouped_counts[row["family"]].append(counts)
            grouped_values[row["family"]].append(values)
        else:
            record.update({field: "NA" for field in (
                "switches", "adjacent_positions", "span_languages", "span_lengths", "english_span_lengths",
            )})
        sentence_rows.append(record)

    pooled = {family: pool_statistics(grouped_counts[family]) for family in FAMILIES}
    corpus = {family: metric_values(pooled[family]) for family in FAMILIES}
    comparisons, tests = [], []
    for metric in METRICS:
        human_values = [item[metric] for item in grouped_values["human"]]
        human_summary = descriptive_summary(human_values)
        human_mean = human_summary["mean"]
        for family in FAMILIES:
            values = [item[metric] for item in grouped_values[family]]
            summary = descriptive_summary(values)
            record = {
                "metric": metric, "family": family, "model": MODEL_NAMES[family],
                "n_valid": summary["n_valid"], "n_na": audit_summaries[family]["na_rows"],
                "sentence_mean": summary["mean"], "sentence_sd_ddof0": summary["sd_ddof0"],
                "sentence_mean_difference_model_minus_human": summary["mean"] - human_mean,
                "human_n_valid": human_summary["n_valid"],
                "human_sentence_mean": human_mean,
                "human_sentence_sd_ddof0": human_summary["sd_ddof0"],
                "corpus_value": corpus[family][metric],
                "corpus_difference_model_minus_human": corpus[family][metric] - corpus["human"][metric],
            }
            if family != "human":
                record.update(welch_comparison(values, human_values))
                tests.append(record)
            else:
                record.update({field: "NA" for field in (
                    "welch_t_model_minus_human", "welch_df", "p_raw_two_sided", "p_bh_30", "significant_bh_0_05",
                )})
            comparisons.append(record)
    if len(tests) != 30:
        raise ValueError("Expected exactly 30 model-vs-human tests")
    for record, adjusted in zip(tests, adjust_bh([item["p_raw_two_sided"] for item in tests])):
        record["p_bh_30"] = adjusted
        record["significant_bh_0_05"] = adjusted < 0.05
    return sentence_rows, pooled, comparisons, tests


def format_number(value, signed=False):
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def format_p(value):
    return "—" if value == "NA" else (f"{value:.3e}" if value < 0.0001 else f"{value:.6f}")


def render_report(comparisons, summaries):
    parts = ["""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human 与五个模型的 code-switching 指标比较</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;color:#182231;line-height:1.65}h1{font-size:27px}h2{margin-top:44px;border-top:2px solid #dae1ea;padding-top:18px}h3{font-size:17px;margin-bottom:10px}.note{color:#4d5c70}table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 20px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #dae1ea;padding:9px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf2f8}tr.human{background:#f7f9fc;font-weight:600}.table-wrap{overflow-x:auto}code{background:#edf2f8;padding:1px 4px}@media print{body{margin:0;max-width:none}h2{break-before:page}.table-wrap{overflow:visible}table{font-size:11px}th,td{padding:6px}}</style>
<h1>Human 与五个模型的 code-switching 指标比较</h1>
<p>按既定六项指标的顺序呈现。所有差值均为 model − human。</p>
<p class="note">Sentence-level 按句等权，SD 为总体标准差（ddof=0）。仅对 sentence-level 均值做双侧 Welch t-test，30 个 p 值统一进行 Benjamini–Hochberg 校正，以调整后 p &lt; 0.05 为显著标准。Corpus-level 汇总有效行内部的计数与 span；句子不拼接，跨句不产生切换或合并 span，结果作描述性比较。</p>
<p class="note">每句视为独立观测；未对 prompt group 或生成批次的潜在相关性建模。p 值作为探索性证据，组内正相关可能使标准误偏低、p 值偏小。</p>
<h3>有效样本与排除</h3><div class="table-wrap"><table><tr><th>来源</th><th>原始行数</th><th>NA</th><th>有效行数</th></tr>"""]
    for family in FAMILIES:
        summary = summaries[family]
        parts.append(f"<tr><td>{html.escape(MODEL_NAMES[family])}</td><td>{summary['rows']}</td><td>{summary['na_rows']}</td><td>{summary['valid_analysis_rows']}</td></tr>")
    parts.append("</table></div>")
    for index, metric in enumerate(METRICS, 1):
        records = [row for row in comparisons if row["metric"] == metric]
        parts.append(f"<h2>{index}. {html.escape(LABELS[metric])}</h2><p class='note'>{html.escape(FORMULAS[metric])}</p><h3>Sentence-level</h3><div class='table-wrap'><table><tr><th>来源</th><th>n</th><th>Mean</th><th>SD (ddof=0)</th><th>Mean 差值</th><th>BH-adjusted p</th><th>显著</th></tr>")
        for row in records:
            significant = "—" if row["family"] == "human" else ("是" if row["significant_bh_0_05"] else "否")
            css = " class='human'" if row["family"] == "human" else ""
            parts.append(f"<tr{css}><td>{html.escape(row['model'])}</td><td>{row['n_valid']}</td><td>{format_number(row['sentence_mean'])}</td><td>{format_number(row['sentence_sd_ddof0'])}</td><td>{format_number(row['sentence_mean_difference_model_minus_human'], True)}</td><td>{format_p(row['p_bh_30'])}</td><td>{significant}</td></tr>")
        parts.append("</table></div><h3>Corpus-level</h3><div class='table-wrap'><table><tr><th>来源</th><th>有效行数</th><th>整体指标值</th><th>Model − human</th></tr>")
        for row in records:
            css = " class='human'" if row["family"] == "human" else ""
            parts.append(f"<tr{css}><td>{html.escape(row['model'])}</td><td>{row['n_valid']}</td><td>{format_number(row['corpus_value'])}</td><td>{format_number(row['corpus_difference_model_minus_human'], True)}</td></tr>")
        parts.append("</table></div>")
    parts.append("<p class='note'>CSV 保留完整计算精度；本报告的指标展示至小数点后六位。原始 p 值、t、df、逐句指标和 corpus span 分布见同目录 CSV。</p></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/metric_comparisons")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output directory must stay inside the project outputs directory")
    rows, summaries, input_hashes = verify_inputs()
    sentence_rows, pooled, comparisons, tests = compute_results(rows, summaries)
    for path, expected in input_hashes.items():
        if audit.digest(ROOT / path) != expected:
            raise ValueError(f"Input changed during calculation: {path}")

    output.mkdir(parents=True, exist_ok=True)
    extra_fields = ["switches", "adjacent_positions", "span_languages", "span_lengths", "english_span_lengths"]
    write_csv(output / "sentence_metrics.csv", sentence_rows, list(rows[0]) + extra_fields)
    write_csv(output / "sentence_summary.csv", comparisons, COMPARISON_FIELDS[:7])
    write_csv(output / "welch_tests.csv", tests, [
        *COMPARISON_FIELDS[:13], *COMPARISON_FIELDS[15:],
    ])
    corpus_fields = ["metric", "family", "model", "n_valid", "corpus_value", "corpus_difference_model_minus_human"]
    write_csv(output / "corpus_metrics.csv", comparisons, corpus_fields)
    write_csv(output / "sample_counts.csv", [
        {**summaries[family], "na_reasons": json.dumps(summaries[family]["na_reasons"], ensure_ascii=False)}
        for family in FAMILIES
    ], [
        "family", "model", "rows", "na_rows", "valid_analysis_rows", "na_reasons",
    ])
    pooled_rows, distribution = [], []
    for family in FAMILIES:
        counts = pooled[family]
        pooled_rows.append({
            "family": family, "model": MODEL_NAMES[family],
            "n_valid": summaries[family]["valid_analysis_rows"],
            **{key: counts[key] for key in ("chinese_characters", "english_words", "effective_tokens", "switches", "adjacent_positions")},
            "all_spans": len(counts["span_lengths"]),
            "chinese_spans": len(counts["chinese_span_lengths"]),
            "english_spans": len(counts["english_span_lengths"]),
            "single_word_english_spans": counts["english_span_lengths"].count(1),
        })
        chinese, english = Counter(counts["chinese_span_lengths"]), Counter(counts["english_span_lengths"])
        for length in sorted(chinese.keys() | english.keys()):
            distribution.append({"family": family, "model": MODEL_NAMES[family], "span_length": length,
                                 "chinese_span_count": chinese[length], "english_span_count": english[length],
                                 "all_span_count": chinese[length] + english[length]})
    write_csv(output / "corpus_counts.csv", pooled_rows, list(pooled_rows[0]))
    write_csv(output / "corpus_span_distribution.csv", distribution, list(distribution[0]))
    for index, metric in enumerate(METRICS, 1):
        write_csv(output / "by_metric" / f"{index:02d}_{metric}.csv",
                  [row for row in comparisons if row["metric"] == metric], COMPARISON_FIELDS)
    (output / "comparison_report.html").write_text(render_report(comparisons, summaries), encoding="utf-8")
    manifest = {
        "calculated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "system": platform.system()},
        "source_and_audit_sha256": input_hashes,
        "metric_order": list(METRICS), "model_order": [MODEL_NAMES[family] for family in FAMILIES],
        "methods": {
            "sentence_summary_sd_ddof": 0, "burstiness_sd_ddof": 0,
            "sentence_weighting": "Equal weight per included sentence.",
            "corpus_boundaries": "Sum within-sentence counts and pool existing within-sentence spans; never concatenate sentences or merge spans across boundaries.",
            "corpus_i_index": "sum(S_i) / sum(N_i - 1)",
            "test": "Two-sided independent Welch t-test, model minus human, with the test's sample-variance definition.",
            "multiple_testing": "One Benjamini-Hochberg correction across all 30 tests; adjusted p < 0.05.",
            "corpus_inference": "Descriptive only; no corpus-level t-tests.",
            "independence_assumption": "Each sentence treated as independent; prompt/batch dependence not modeled. P-values are exploratory.",
        },
        "sample_counts": [summaries[family] for family in FAMILIES],
        "comparisons": comparisons,
        "output_sha256": {path.relative_to(output).as_posix(): audit.digest(path)
                          for path in sorted(output.rglob("*")) if path.is_file() and path.name != "summary.json"},
    }
    (output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for metric in METRICS:
        significant = sum(row["significant_bh_0_05"] for row in tests if row["metric"] == metric)
        print(f"{LABELS[metric]}: calculated for human and all five models; {significant}/5 BH-adjusted p < 0.05")
    print(f"Saved {len(rows)} sentence rows, {sum(row['analysis_status'] == 'NA' for row in rows)} NA rows, 30 tests: {output}")


if __name__ == "__main__":
    main()
