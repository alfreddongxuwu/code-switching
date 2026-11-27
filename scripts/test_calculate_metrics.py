"""Analytical checks of the study's aggregation and statistical conventions.

Run: python -m unittest discover -s scripts -p test_calculate_metrics.py -v
"""

import math
import unittest

from calculate_metrics import (
    METRICS, FAMILIES, MODEL_NAMES, adjust_bh, compute_results,
    descriptive_summary, metric_values, pool_statistics, sentence_statistics,
    welch_comparison,
)


class MetricCalculationTests(unittest.TestCase):
    def test_balanced_two_spans_have_known_metric_values(self):
        values = metric_values(sentence_statistics("甲乙 one two"))
        expected = (1, 1 / 3, -1, 0, 2, 0)
        for metric, answer in zip(METRICS, expected):
            self.assertAlmostEqual(values[metric], answer)

    def test_corpus_uses_pooled_counts_instead_of_sentence_means(self):
        first = sentence_statistics("甲乙 one two")
        second = sentence_statistics("甲 one 乙 two three")
        values = metric_values(pool_statistics([first, second]))
        # C=4, E=5; switches=1+3; opportunities=3+4.
        # Span lengths: [2,2,1,1,1,2], English: [2,1,2].
        expected = (40 / 41, 4 / 7, -0.5, 1, 5 / 3, 1 / 3)
        for metric, answer in zip(METRICS, expected):
            self.assertAlmostEqual(values[metric], answer)
        sentence_mean = (metric_values(first)["i_index"] + metric_values(second)["i_index"]) / 2
        self.assertNotAlmostEqual(values["i_index"], sentence_mean)

    def test_adjacent_english_spans_across_rows_are_not_merged(self):
        pooled = pool_statistics([
            sentence_statistics("甲 one two"), sentence_statistics("three 乙"),
        ])
        self.assertEqual(pooled["english_span_lengths"], [2, 1])
        self.assertEqual(pooled["span_lengths"], [1, 2, 1, 1])
        self.assertEqual(pooled["switches"], 2)
        self.assertEqual(pooled["adjacent_positions"], 3)
        values = metric_values(pooled)
        self.assertAlmostEqual(values["mean_english_span_length"], 1.5)
        self.assertAlmostEqual(values["single_word_english_span_proportion"], 0.5)
        self.assertAlmostEqual(values["i_index"], 2 / 3)

    def test_confirmed_contact_and_number_rules_preserve_spans(self):
        counts = sentence_statistics("甲 don't part-time GPT4o em@160 pn-19 买了2个 3.5")
        self.assertEqual(counts["chinese_characters"], 4)
        self.assertEqual(counts["english_words"], 5)
        self.assertEqual(counts["span_lengths"], [1, 5, 3])

    def test_descriptive_sd_is_population_but_welch_uses_sample_variance(self):
        self.assertAlmostEqual(descriptive_summary([1, 2, 3])["sd_ddof0"], math.sqrt(2 / 3))
        comparison = welch_comparison([1, 2, 3], [2, 4, 6])
        # Sample variances are 1 and 4; SE²=1/3+4/3; Welch df=50/17.
        self.assertAlmostEqual(comparison["welch_t_model_minus_human"], -2 / math.sqrt(5 / 3))
        self.assertAlmostEqual(comparison["welch_df"], 50 / 17)
        reversed_test = welch_comparison([2, 4, 6], [1, 2, 3])
        self.assertAlmostEqual(comparison["p_raw_two_sided"], reversed_test["p_raw_two_sided"])
        self.assertAlmostEqual(comparison["welch_t_model_minus_human"], -reversed_test["welch_t_model_minus_human"])

    def test_bh_known_vector_keeps_original_order(self):
        for actual, expected in zip(adjust_bh([0.01, 0.04, 0.03, 0.002]), [0.02, 0.04, 0.04, 0.008]):
            self.assertAlmostEqual(actual, expected)

    def test_na_rows_do_not_enter_either_level_and_bh_is_global(self):
        rows, summaries = [], {}
        for index, family in enumerate(FAMILIES):
            texts = ["甲乙 one two", "甲 one 乙 two three", "甲乙丙 one two three four five"]
            for number, text in enumerate(texts, 1):
                # Model groups differ from human without giving any constant metric.
                if family != "human":
                    text += "甲" * index
                counts = sentence_statistics(text)
                rows.append({"family": family, "model": MODEL_NAMES[family], "data_row": number,
                             "sentence": text, "analysis_status": "valid",
                             **{key: counts[key] for key in ("chinese_characters", "english_words", "effective_tokens")},
                             **metric_values(counts)})
            rows.append({"family": family, "sentence": "excluded " * 500, "analysis_status": "NA",
                         **{metric: "NA" for metric in METRICS}})
            summaries[family] = {"na_rows": 1}
        _, pooled, comparisons, tests = compute_results(rows, summaries)
        self.assertEqual(len(tests), 30)
        self.assertEqual(len(comparisons), 36)
        self.assertTrue(all(row["n_valid"] == 3 and row["human_n_valid"] == 3 for row in comparisons))
        self.assertEqual(pooled["human"]["english_words"], 10)
        for row, expected in zip(tests, adjust_bh([item["p_raw_two_sided"] for item in tests])):
            self.assertAlmostEqual(row["p_bh_30"], expected)


if __name__ == "__main__":
    unittest.main()
