"""
Evaluate the translation gloss outputs by the model.
Inputs:
- a csv file containing the source sentences and the model outputs (glosses).
- a csv file (or some structured dataset, like path to a Hugging Face dataset... TODO) containing the reference glosses.
Output:
- Character Error Rate (CER) and Word Error Rate (WER).
"""

import argparse
import re
import os
import pandas as pd
import numpy as np
import openai
import jiwer
# from torchmetrics.text import TranslationEditRate
# ^ I'm ditching torchmetrics because it's super heavy and takes time to import
import sacrebleu
import json
from collections import Counter

from typing import List, Dict, Set, Literal, Tuple, Sequence, Optional
import unittest

# local imports
from structuralize import structuralize, StructuralizedOutput, DEFAULT_SYSTEM_PROMPT


SOURCE_COLUMN = "source_sentence"
MODEL_OUTPUT_COLUMN = "translation_gloss"


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate the translation gloss outputs by the model."
    )
    parser.add_argument(
        "--model_outputs_file",
        type=str,
        default="translation_results.csv",
        help="Path to the CSV file containing source sentences and model outputs.",
    )
    parser.add_argument(
        "--reference_glosses",
        type=str,
        help="Path to the CSV file (or hugging face dataset path) containing reference glosses.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests instead of evaluation.",
    )
    parser.add_argument(
        "--run_example",
        action="store_true",
        help="Run the example evaluation script.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="If set, the evaluation is to be run as part of the morphosyntax pipeline.",
    )
    parser.add_argument(
        "--eval_user_prompt_file",
        type=str,
        default="evaluation/eval_user_prompt_template.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
    )
    parser.add_argument(
        "--no_print",
        action="store_true",
        help="If set, the evaluation output will not be printed."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="The path to the structured output results file."
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        help="The path to the scores output file."
    )
    parser.add_argument(
        "--skip_structuralize",
        action="store_true",
        help="If set, the structuralization step will be skipped."
    )

    return parser.parse_args()


class MorphosyntaxMetrics:
    """Class to calculate morphosyntactic metrics such as Morpheme Error Rate (MER)."""

    def __init__(self):
        """Initialize the MorphosyntaxMetrics class."""
        pass


    def compute_all_metrics(self,
                            hyps: List[str] | str,
                            refs: List[List[str]] | str,
                            print_scores: bool = True) -> Dict[str, float]:
        """Compute all metrics."""
        bleu_score = self.compute_bleu(hyps, refs)
        chrf_score = self.compute_chrf(hyps, refs)
        wer_score = self.compute_wer(hyps, refs)
        cer_score = self.compute_cer(hyps, refs)
        ter_score = self.compute_ter(hyps, refs)
        ser_score = self.compute_ser(hyps, refs)
        mfer_score = self.compute_mfer(hyps, refs)
        mser_score = self.compute_mser(hyps, refs)
        # morph_acc = self.morpheme_accuracy(hyps, refs)
        # word_acc = self.word_accuracy(hyps, refs)
        lemma_prec, lemma_rec, lemma_f1 = self.compute_lemmaf1(hyps, refs)

        if print_scores:
            print(f"BLEU: {bleu_score}")
            print(f"ChrF++: {chrf_score}")
            print(f"WER: {wer_score}")
            print(f"CER: {cer_score}")
            print(f"TER: {ter_score}")
            print(f"SER: {ser_score}")
            print(f"MFER: {mfer_score}")
            print(f"MSER: {mser_score}")
            # print(f"MORPHEME ACCURACY: {morph_acc}")
            # print(f"WORD ACCURACY: {word_acc}")
            print(f"LEMMA PRECISION: {lemma_prec}")
            print(f"LEMMA RECALL: {lemma_rec}")
            print(f"LEMMA F1: {lemma_f1}")

        return {
            "BLEU": bleu_score,
            "ChrF++": chrf_score,
            "WER": wer_score,
            "CER": cer_score,
            "TER": ter_score,
            "SER": ser_score,
            "MFER": mfer_score,
            "MSER": mser_score,
            # "Morpheme accuracy": morph_acc,
            # "Word accuracy": word_acc,
            "Lemma precision": lemma_prec,
            "Lemma recall": lemma_rec,
            "Lemma F1": lemma_f1,
        }


    def morpheme_accuracy(self, hyp: List[str] | str, ref: List[List[str]] | str) -> float:
        """
        Compute morpheme accuracy by comparing the morphemes of the reference and prediction.
        """
        if isinstance(hyp, list):
            hyp_morphemes = [self.strip_punctuation(h).replace("-", " ").split() for h in hyp]
        else:
            hyp_morphemes = self.strip_punctuation(hyp).replace("-", " ").split()

        if isinstance(ref, list):
            ref_morphemes = [self.strip_punctuation(r[0]).replace("-", " ").split() for r in ref]
            # use the first ref sentence for now...
        else: # str
            ref_morphemes = self.strip_punctuation(ref).replace("-", " ").split()

        correct_morphemes = 0

        for ref_morph, pred_morph in zip(ref_morphemes, hyp_morphemes):
            if ref_morph == pred_morph:
                correct_morphemes += 1

        # Morpheme accuracy is the number of correct morphemes divided by the total number of morphemes in the reference
        return correct_morphemes / len(ref_morphemes) if ref_morphemes else 0

    def word_accuracy(self, hyp: List[str] | str, ref: List[List[str]] | str) -> float:
        """
        Compute word accuracy by comparing the entire words (glosses) of the reference and prediction.
        """
        if isinstance(hyp, list):
            hyp_words = [self.strip_punctuation(h).split() for h in hyp]
        else:
            hyp_words = self.strip_punctuation(hyp).split()

        if isinstance(ref, list):
            ref_words = [self.strip_punctuation(r[0]).split() for r in ref]
            # use the first ref sentence for now...
        else: # str
            ref_words = self.strip_punctuation(ref).split()

        correct_words = 0
        for ref in ref_words:
            if ref in hyp_words:
                correct_words += 1

        # Word accuracy is the number of correct words divided by the total number of words in the reference
        return correct_words / len(ref_words) if ref_words else 0

    def compute_ter_dp(self, hyp: str | list, ref: str | list) -> float:
        """Compute Translation Edit Rate (TER) between the reference and hypothesis using dynamic programming.
        Note that this method calculates the score differently from the original TER.
        For example, suppose we have a reference "A B C D" and a hypothesis "C D A B".
        This method will return 1 (4 edits out of 4 ref tokens), while the original TER would return 0.25 (1 edit (shift-right of A B) out of 4 ref tokens).
        See also the original paper: https://aclanthology.org/2006.amta-papers.25.pdf

        Args:
            ref (str): A string representing the reference translation.
            hyp (str): A string representing the hypothesis translation.

        Returns:
            float: The percentage of edits required to transform the reference translation into the hypothesis translation.
        """
        if isinstance(ref, str):
            if ref == hyp:
                return 0.0

            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            return self.minimum_edits(ref_tokens, hyp_tokens) / len(ref_tokens)

        elif isinstance(ref, list) and isinstance(hyp, list):
            if len(ref) != len(hyp):
                raise ValueError(
                    "Reference and hypothesis lists must have the same length."
                )

            return sum(
                self.minimum_edits(r.split(), h.split()) / len(r.split())
                for r, h in zip(ref, hyp)
            ) / len(ref)

        else:
            raise ValueError(
                "Both reference and hypothesis must be either strings or lists of strings."
            )

    def compute_ter(
        self,
        hyp: str | list,
        ref: str | list,
        normalized: bool = False,
        no_punct: bool = True,
        case_sensitive: bool = True,
        # return_sentence_level_score: bool = False,
    ) -> float:
        """Compute Translation Edit Rate (TER) between the reference and hypothesis.
        See the original paper: https://aclanthology.org/2006.amta-papers.25.pdf

        Args:
            ref (str): A string representing the reference translation.
            hyp (str): A string representing the hypothesis translation.

        Returns:
            float: The percentage of edits required to transform the reference translation into the hypothesis translation.
        """
        # ter = TranslationEditRate(
        #     return_sentence_level_score=return_sentence_level_score,
        #     no_punctuation=no_punctuation,
        #     normalize=normalize,
        #     lowercase=lowercase,
        # )
        if isinstance(ref, str):
            ref = [[ref]]
        elif isinstance(ref[0], str):
            ref = [[r] for r in ref]  # make it a list of list
        if isinstance(hyp, str):
            hyp = [hyp]
        return sacrebleu.corpus_ter(hyp,
                                    ref,
                                    normalized=normalized,
                                    no_punct=no_punct,
                                    case_sensitive=case_sensitive,
                                    asian_support=False).score
        # if return_sentence_level_score:
        #     return ter(hyp, ref)[0].item()
        # else:
        #     return ter(hyp, ref).item()

    def compute_ser(
        self,
        hyp: str | list,
        ref: str | list,
        normalized: bool = False,
        no_punct: bool = True,
        case_sensitive: bool = True,
        # return_sentence_level_score: bool = False,
    ) -> float:
        """Stem edit rate.

        Args:
            ref (str): A string representing the reference translation.
            hyp (str): A string representing the hypothesis translation.

        Returns:
            float: The percentage of edits required to transform the stems in
            the reference translation into the stems in the hypothesis translation.
        """
        # ter = TranslationEditRate(return_sentence_level_score=False)
        if isinstance(ref, str):
            ref = [[ref]]
        if isinstance(ref[0], str):
            ref = [[r] for r in ref]  # make it a list of list
        if isinstance(hyp, str):
            hyp = [hyp]

        # for debugging
        # print("--- INPUT ---")
        # print("ref:", ref[0])
        # print("hyp:", hyp[0])

        ref = [
            [label.replace("-", " ").split() for label in labels] for labels in ref
        ]  # replace hyphens with spaces
        hyp = [h.replace("-", " ").split() for h in hyp]

        # print("--- ORIGINAL ---")
        # print("ref:", ref[0])
        # print("hyp:", hyp[0])

        ref_stems = [
            [
                [r for r in label if not r.isupper()] for label in labels
            ]  # upper-case morphemes are features
            for labels in ref
        ]
        hyp_stems = [
            [
                h for h in h_tokens if not h.isupper()
            ]  # upper-case morphemes are features
            for h_tokens in hyp
        ]

        # print("--- STEMS ---")
        # print("ref:", ref_stems[0])
        # print("hyp:", hyp_stems[0])

        ref_stems_str = [[" ".join(label) for label in labels] for labels in ref_stems]
        hyp_stems_str = [" ".join(h) for h in hyp_stems]

        # print("ref:", ref_stems_str[0])
        # print("hyp:", hyp_stems_str[0])

        return sacrebleu.corpus_ter(hyp_stems_str,
                                    ref_stems_str,
                                    normalized=normalized,
                                    no_punct=no_punct,
                                    case_sensitive=case_sensitive,
                                    asian_support=False).score

        # if return_sentence_level_score:
        #     return ter(hyp_stems_str, ref_stems_str)[0].item()
        # else:
        #     return ter(hyp_stems_str, ref_stems_str).item()

    def _mfer_edits_against_one_ref(self, hyp: str, ref: str) -> Tuple[float, int]:
        """
        Return: (raw_edit_mass, ref_token_count) for a single reference string.
        """
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        edits = 0.0

        for ref_t in ref_tokens:
            ref_m = self.get_morphemes(ref_t)
            hits = [h for h in hyp_tokens if self.get_morphemes(h)["stem"] == ref_m["stem"]]

            if hits:
                distances: List[Dict[str, int]] = [
                    self.count_morphological_edits(ref_m, self.get_morphemes(h)) for h in hits
                ]
                best = min(distances, key=lambda x: x["distance"])
                denom = max(1, best["num_base_feats"])
                penalty = best["distance"] / denom
            else:
                penalty = 1.0

            edits += penalty

        return edits, len(ref_tokens)

    def _compute_mfer_sentence(
        self,
        hyp: str,
        ref: str,
        multi_ref_denominator: str = "avg",  # {"avg","best","shortest","longest"}
    ) -> Tuple[float, int, float]:
        """
        Compute sentence-level MFER with possibly multiple references.

        Returns:
          mfer_sent: normalized [0,1]
          denom_tokens_used: denominator token count used for this segment
          edits_chosen: raw edit mass of the chosen/best reference
        """
        # Normalize refs into a list of candidate references
        candidates: List[str] = [ref] if isinstance(ref, str) else list(ref)
        if len(candidates) == 0:
            # no references: define as 0/0 -> 0.0 by convention
            return 0.0, 0, 0.0

        # Compute edits vs each reference
        per_ref: List[Tuple[float, int]] = [
            self._mfer_edits_against_one_ref(r, hyp) for r in candidates
        ]
        edits_list = [e for (e, _) in per_ref]
        len_list = [n for (_, n) in per_ref]

        # Pick the best (fewest edits) reference for numerator
        best_idx = min(range(len(per_ref)), key=lambda i: edits_list[i])
        edits_best = edits_list[best_idx]

        # Choose denominator policy
        if multi_ref_denominator == "avg":
            denom_tokens = max(
                1, int(round(sum(len_list) / len(len_list)))
            )  # TER-style average
            # If you prefer floating denominator, remove int(round(...)) and keep as float
            denom_float = sum(len_list) / len(len_list)
            mfer_sent = (edits_best / denom_float) if denom_float > 0 else 0.0
            # Return integer denom for micro tallying? Use float-aware tallying below instead.
            return mfer_sent, 0, edits_best  # we'll tally floats for micro
        elif multi_ref_denominator == "best":
            denom_tokens = max(1, len_list[best_idx])
        elif multi_ref_denominator == "shortest":
            denom_tokens = max(1, min(len_list))
        elif multi_ref_denominator == "longest":
            denom_tokens = max(1, max(len_list))
        else:
            raise ValueError(
                "multi_ref_denominator must be one of {'avg','best','shortest','longest'}"
            )

        mfer_sent = edits_best / denom_tokens
        return mfer_sent, denom_tokens, edits_best

    def compute_mfer(
        self,
        hyp: str | List[str],
        ref: str | List[str],
        average: Literal["micro", "macro"] = "micro",
        multi_ref_denominator: str = "best",
    ) -> float:
        """
        Morphological Feature Error Rate (MFER) with multi-reference support.

        - If ref/hyp are str: returns sentence-level MFER.
        - If ref is a sequence and hyp is a sequence of equal length:
            * Each ref[i] can be a str (single reference) OR a sequence[str] (multiple references).
            * Returns corpus-level MFER (micro or macro).
        """
        # Sentence mode
        if isinstance(ref, str) and isinstance(hyp, str):
            mfer_sent, _, _ = self._compute_mfer_sentence(hyp=hyp,
                                                          ref=ref,
                                                          multi_ref_denominator=multi_ref_denominator)
            return mfer_sent

        # Corpus mode
        if not (isinstance(ref, Sequence) and isinstance(hyp, Sequence)):
            raise TypeError("In corpus mode, ref and hyp must both be Sequences of equal length.")

        if len(ref) != len(hyp):
            raise ValueError(f"ref and hyp must have the same length (got {len(ref)} vs {len(hyp)}).")

        if not ref:
            return 0.0

        if average not in {"micro", "macro"}:
            raise ValueError("average must be 'micro' or 'macro'.")

        total_edits = 0.0
        total_denom_tokens = 0.0  # allow float when using 'avg' denominator
        sent_scores: List[float] = []

        for refs_i, hyp_i in zip(ref, hyp):
            # refs_i can be a str (single) or a Sequence[str] (multiple)
            mfer_i, denom_i, edits_i = self._compute_mfer_sentence(
                hyp=hyp_i, ref=refs_i, multi_ref_denominator=multi_ref_denominator
            )
            sent_scores.append(mfer_i)
            total_edits += edits_i

            if multi_ref_denominator == "avg":
                # Recompute float denominator for micro tally in 'avg' mode
                cand = [refs_i] if isinstance(refs_i, str) else list(refs_i)
                if len(cand) == 0:
                    denom_float = 0.0
                else:
                    denom_float = sum(len(r.split()) for r in cand) / len(cand)
                total_denom_tokens += denom_float
            else:
                total_denom_tokens += denom_i

        if average == "micro":
            return (total_edits / total_denom_tokens) * 100 if total_denom_tokens > 0 else 0.0
        else:  # macro
            return sum(sent_scores) / len(sent_scores) * 100

    def compute_mser(self, hyp: str, ref: str, alpha: float = 0.5) -> float:
        """Compute Morphosyntactic Edit Rate.
        A combined metric of stem edit rate and morpheme feature edit rate.

        Args:
            ref (str): A string representing the reference translation.
            hyp (str): A string representing the hypothesis translation.
            alpha (float, optional): weight. smfer = alpha * ser + (1 - alpha) * mfer. Defaults to 0.5.

        Returns:
            float: The percentage of edits required to transform the reference translation into the hypothesis translation.

        Note that SER returns a percentage (0-100), while MFER returns a ratio (0-1).
        Therefore, we multiply MFER by 100 to align the scales before combining them.
        """
        return alpha * self.compute_ser(hyp=hyp, ref=ref) + (1 - alpha) * self.compute_mfer(
            hyp=hyp, ref=ref
        )

    def count_morphological_edits(
        self, hyp: Dict[str, Set[str]], ref: Dict[str, Set[str]]
    ) -> Dict[str, int]:
        """Count the number of morphological edits and the number of base features."""
        suffix_diff = ref["suffixes"].symmetric_difference(hyp["suffixes"])
        prefix_diff = ref["prefixes"].symmetric_difference(hyp["prefixes"])
        ref_num_feats = len(ref["prefixes"]) + len(ref["suffixes"]) + 1
        hyp_num_feats = len(hyp["prefixes"]) + len(hyp["suffixes"]) + 1
        num_base_feats = max(ref_num_feats, hyp_num_feats)
        return {
            "distance": len(suffix_diff) + len(prefix_diff),
            "num_base_feats": num_base_feats,
        }

    def minimum_edits(self, hyp_tokens: List[str], ref_tokens: List[str]) -> int:
        """
        Calculate Translation Edit Rate (TER) between the reference and hypothesis.

        Args:
            ref_tokens (List[str]): A list of tokens in the reference translation.
            hyp_tokens (List[str]): A list of tokens in the hypothesis translation.

        Returns:
            int: The minimum number of edits required to transform the reference translation into the hypothesis translation.
        """
        # Create a DP matrix where dp[i][j] represents the minimum edit distance
        # between the first i tokens of reference and first j tokens of hypothesis
        m = len(ref_tokens)
        n = len(hyp_tokens)
        dp = np.zeros((m + 1, n + 1))

        for i in range(m + 1):
            dp[i][0] = i
            # Cost of deleting all reference tokens to match empty hypothesis

        for j in range(n + 1):
            dp[0][j] = j
            # Cost of inserting all hypothesis tokens to match empty reference

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    # same token; no edit needed
                    dp[i][j] = dp[i - 1][j - 1]

                else:
                    dp[i][j] = (
                        min(
                            dp[i - 1][j - 1],  # substitution
                            dp[i - 1][j],  # deletion
                            dp[i][j - 1],  # insertion
                        )
                        + 1
                    )

        return int(dp[m][n])

    def get_morphemes(self, word: str) -> Dict[str, Set[str] | None]:
        """Extract prefix features and suffix features.

        Args:
            word (str): A string representing the word.

        Returns:
            Dict[str, Set[str] | None]: A dictionary containing sets of prefixes and suffixes,
            and the stem of the word.
        """
        morphemes = word.replace("-", " ").split()
        pre_feats = set()
        suf_feats = set()
        stem = ""  # There can be a token without a stem, e.g., "NEG" as a particle
        is_prefix = True
        for m in morphemes:
            if m.islower():  # the stem
                stem = m
                is_prefix = False
            else:
                if is_prefix:
                    pre_feats.add(m.lower())
                else:  # suffix
                    suf_feats.add(m.lower())
        return {"prefixes": pre_feats, "stem": stem, "suffixes": suf_feats}

    def strip_punctuation(self, gloss: str) -> str:
        """
        Strips punctuation from a gloss, except for within glosses (e.g., parentheses).
        """
        # Replace all punctuation except for those inside parentheses or within glosses
        return re.sub(r"[^\w\s\(\)-]", "", gloss)

    # BLEU, ChrF++, WER, CER
    def compute_bleu(self,
                     hyp: str | List[str],
                     ref: str | List[List[str]]
                     ) -> float:
        """Compute BLEU score using sacrebleu."""
        if isinstance(ref, str):
            ref = [[ref]]
        if isinstance(hyp, str):
            hyp = [hyp]
        return sacrebleu.corpus_bleu(hyp, ref).score

    def compute_chrf(self,
                     hyp: str | List[str],
                     ref: str | List[List[str]]
                     ) -> float:
        """Compute ChrF score using sacrebleu."""
        if isinstance(ref, str):
            ref = [[ref]]
        if isinstance(hyp, str):
            hyp = [hyp]
        return sacrebleu.corpus_chrf(hyp, ref).score

    def compute_wer(self,
                    hyp: str | List[str],
                    ref: str | List[List[str]],
                    ) -> float:
        """Compute Word Error Rate (WER) using jiwer.
        WER does not support multiple references, so we assume ref is a single string or a list of strings."""
        if isinstance(ref, str):
            ref = [ref]
        else:
            if isinstance(ref[0], list):
                ref = [r[0] for r in ref]

        if isinstance(hyp, str):
            hyp = [hyp]

        return jiwer.wer(ref, hyp) * 100

    def compute_cer(self,
                    hyp: str | List[str],
                    ref: str | List[str]
                    ) -> float:
        """Compute Character Error Rate (CER) using jiwer.
        CER does not support multiple references, so we assume ref is a single string or a list of strings."""
        if isinstance(ref, str):
            ref = [ref]
        else:
            if isinstance(ref[0], list):
                ref = [r[0] for r in ref]

        if isinstance(hyp, str):
            hyp = [hyp]

        return jiwer.cer(ref, hyp) * 100

    def compute_lemmaf1(self,
                        ref: str | List[str],
                        hyp: str | List[str],
                        average: Literal["micro", "macro", "weighted"] = "macro"
                        ) -> Tuple[float, float, float]:
        """Compute Lemma F1 score.

        Args:
            ref (str | List[str]): Reference gloss or list of glosses.
            hyp (str | List[str]): Hypothesis gloss or list of glosses.

        Returns:
            tuple: (precision, recall, f1)
        """
        def _lemmatize(gloss: str) -> List[str]:
            """A simple lemmatizer that removes morphological features from a gloss."""
            return [m for m in gloss.replace("-", " ").split() if m.islower()]

        def _get_lemma_counts(lemmas_list: List[List[str]]) -> Dict[str, int]:
            """Get lemma counts from a list of lemma lists."""
            all_lemmas = []
            for lemmas in lemmas_list:
                all_lemmas.extend(lemmas)
            return Counter(all_lemmas)

        if isinstance(ref, str):
            ref = [ref]
        if isinstance(hyp, str):
            hyp = [hyp]

        # ref: List[List[str]] -> List[str]  # list of reference glosses
        # hyp: List[str]  # list of hypothesis glosses
        # For now, we only use the first reference gloss if multiple are provided
        ref = [r[0] if isinstance(r, list) else r for r in ref]

        if average == "micro":
            # Micro-averaging: aggregate counts across all sentences
            ref_counter = _get_lemma_counts([_lemmatize(r) for r in ref])
            hyp_counter = _get_lemma_counts([_lemmatize(h) for h in hyp])

            # Calculate true positives, false positives, false negatives
            true_positives = 0
            for lemma, count in hyp_counter.items():
                true_positives += min(count, ref_counter.get(lemma, 0))

            false_positives = sum(hyp_counter.values()) - true_positives
            false_negatives = sum(ref_counter.values()) - true_positives

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return precision * 100, recall * 100, f1 * 100

        elif average == "macro":
            # Macro-averaging: calculate F1 for each sentence and average
            precisions = []
            recalls = []
            f1s = []

            for r, h in zip(ref, hyp):
                r_counter = _get_lemma_counts([_lemmatize(r)])
                h_counter = _get_lemma_counts([_lemmatize(h)])

                tp = 0
                for lemma, count in h_counter.items():
                    tp += min(count, r_counter.get(lemma, 0))

                fp = sum(h_counter.values()) - tp
                fn = sum(r_counter.values()) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            precision = sum(precisions) / len(precisions) if precisions else 0.0
            recall = sum(recalls) / len(recalls) if recalls else 0.0
            f1 = sum(f1s) / len(f1s) if f1s else 0.0

            return precision * 100, recall * 100, f1 * 100

        elif average == "weighted":
            # Weighted averaging: macro-average weighted by support (number of true lemmas)
            sentence_weights = [len(r) for r in ref]
            total_weight = sum(sentence_weights)

            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0

            for r, h, weight in zip(ref, hyp, sentence_weights):
                r_counter = _get_lemma_counts([_lemmatize(r)])
                h_counter = _get_lemma_counts([_lemmatize(h)])

                tp = 0
                for lemma, count in h_counter.items():
                    tp += min(count, r_counter.get(lemma, 0))

                fp = sum(h_counter.values()) - tp
                fn = sum(r_counter.values()) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                weighted_precision += precision * weight
                weighted_recall += recall * weight
                weighted_f1 += f1 * weight

            precision = weighted_precision / total_weight if total_weight > 0 else 0.0
            recall = weighted_recall / total_weight if total_weight > 0 else 0.0
            f1 = weighted_f1 / total_weight if total_weight > 0 else 0.0

            return precision * 100, recall * 100, f1 * 100

        else:
            raise ValueError("average must be 'micro', 'macro', or 'weighted'.")


class TestMTER(unittest.TestCase):
    """Unit tests for MorphosyntaxMetrics class."""

    def setUp(self):
        """Set up the MorphosyntaxMetrics instance for testing."""
        self.morph_metrics = MorphosyntaxMetrics()

    def test_ter(self):
        # Test case 1: Identical reference and hypothesis
        reference = "The cat sat on the mat"
        hypothesis = "The cat sat on the mat"
        self.assertEqual(self.morph_metrics.compute_ter(reference, hypothesis), 0.0)

        # Test case 2: Some substitution
        reference = "The cat sat on the mat"
        hypothesis = "The dog sat on the mat"
        self.assertAlmostEqual(
            self.morph_metrics.compute_ter(reference, hypothesis), 1 / 6
        )  # 1 substitution

        # Test case 3: Insertion
        reference = "The cat sat on the mat"
        hypothesis = "The cat sat on the mat at home"
        self.assertAlmostEqual(
            self.morph_metrics.compute_ter(reference, hypothesis), 2 / 6
        )  # 2 insertions

        # Test case 4: Deletion
        reference = "The cat sat on the mat"
        hypothesis = "The cat on mat"
        self.assertAlmostEqual(
            self.morph_metrics.compute_ter(reference, hypothesis), 2 / 6
        )  # 2 deletions

        # Test case 5: Multiple edits (substitution + insertion)
        reference = "The cat sat on the mat"
        hypothesis = "A dog is sitting in a home"
        self.assertAlmostEqual(
            self.morph_metrics.compute_ter(reference, hypothesis), 7 / 6
        )  # Multiple edits

    def test_ser(self):
        # Test case 1: Identical reference and hypothesis
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat sit-PAST on DEF-mat"
        self.assertEqual(
            self.morph_metrics.compute_ser(reference, hypothesis), 0.0
        )  # complete stems

        # Test case 2: Identical stems
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "cat sit on mat"
        self.assertEqual(
            self.morph_metrics.compute_ser(reference, hypothesis), 0.0
        )  # complete stems

        # Test case 3: Stem position permutation
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat on DEF-mat sit-PAST"
        self.assertEqual(
            self.morph_metrics.compute_ser(reference, hypothesis), 1 / 4
        )  # complete stems

    def test_mfer(self):
        # Test case 1: Identical reference and hypothesis
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat sit-PAST on DEF-mat"
        self.assertEqual(self.morph_metrics.compute_mfer(reference, hypothesis), 0.0)

        # Test case 2: Missing features
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "cat sit on mat"
        self.assertEqual(self.morph_metrics.compute_mfer(reference, hypothesis), 3 / 8)

        # Test case 3: Too much features
        reference = "cat sit on mat"
        hypothesis = "DEF-cat sit-PAST on DEF-mat"
        self.assertEqual(self.morph_metrics.compute_mfer(reference, hypothesis), 3 / 8)

        # Test case 4
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat-SING sit-PAST-3SGNOM on DEF-mat-SING"
        self.assertEqual(
            self.morph_metrics.compute_mfer(reference, hypothesis),
            (1 / 3 + 1 / 3 + 0 + 1 / 3) / 4,
        )

        # Test case 5
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF cat sit PAST on DEF mat"
        self.assertEqual(
            self.morph_metrics.compute_mfer(reference, hypothesis),
            (1 / 2 + 1 / 2 + 0 + 1 / 2) / 4,
        )

        # Test case 6: different order
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat on DEF-mat sit-PAST"
        self.assertEqual(self.morph_metrics.compute_mfer(reference, hypothesis), 0.0)

    def test_mser(self):
        # Test case 1:
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat on DEF-mat sit-PAST"
        self.assertEqual(
            self.morph_metrics.compute_mser(reference, hypothesis), (1 / 4 + 0.0) / 2
        )

        # Test case 2:
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat-SING sit-PAST-3SGNOM on DEF-mat-SING"
        self.assertEqual(
            self.morph_metrics.compute_mser(reference, hypothesis),
            (0.0 + (1 / 3 + 1 / 3 + 0 + 1 / 3) / 4) / 2,
        )

    # def test_morpheme_accuracy(self):
    #     # Test case 1: Identical morphemes
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "cat-NOM sit-PAST"
    #     self.assertEqual(
    #         self.morph_metrics.morpheme_accuracy(reference, hypothesis), 1.0
    #     )

    #     # Test case 2: The first morpheme is wrong
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "dog-NOM sit-PAST"
    #     self.assertEqual(
    #         self.morph_metrics.morpheme_accuracy(reference, hypothesis), 3 / 4
    #     )

    #     # Test case 3: Some morphemes match
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "cat-NOM stand-PAST"
    #     self.assertEqual(
    #         self.morph_metrics.morpheme_accuracy(reference, hypothesis), 3 / 4
    #     )

    #     # Test case 4: Insertion
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "cat-NOM at home sit-PAST"
    #     self.assertEqual(
    #         self.morph_metrics.morpheme_accuracy(reference, hypothesis), 2 / 4
    #     )

    # def test_word_accuracy(self):
    #     # Test case 1: Identical words
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "cat-NOM sit-PAST"
    #     self.assertEqual(self.morph_metrics.word_accuracy(reference, hypothesis), 1.0)

    #     # Test case 2: The first morpheme is wrong
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "dog-NOM sit-PAST"
    #     self.assertEqual(self.morph_metrics.word_accuracy(reference, hypothesis), 0.5)

    #     # Test case 3: Some morphemes match
    #     reference = "cat-NOM sit-PAST"
    #     hypothesis = "cat-NOM stand-PAST"
    #     self.assertEqual(self.morph_metrics.word_accuracy(reference, hypothesis), 0.5)

    def test_lemmaf1(self):
        # Test case 1: Identical glosses
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "DEF-cat sit-PAST on DEF-mat"
        precision, recall, f1 = self.morph_metrics.compute_lemmaf1(reference, hypothesis)
        self.assertEqual((precision, recall, f1), (100.0, 100.0, 100.0))

        # Test case 2: Some matching lemmas
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "cat sit on mat"
        precision, recall, f1 = self.morph_metrics.compute_lemmaf1(reference, hypothesis)
        self.assertEqual((precision, recall, f1), (100.0, 100.0, 100.0))

        # Test case 3: No matching lemmas
        reference = "DEF-cat sit-PAST on DEF-mat"
        hypothesis = "dog run FUT in house"
        precision, recall, f1 = self.morph_metrics.compute_lemmaf1(reference, hypothesis)
        self.assertEqual((precision, recall, f1), (0.0, 0.0, 0.0))


def eval_output(model_output_file: str,
                label_file: str,
                results_file: Optional[str] = None,
                source_sentences_file: str = "sentence_design_output/grammatical_test_sentences.txt",
                eval_user_prompt_file: str = "evaluation/eval_user_prompt_template.txt",
                skip_structuralize: bool = False) -> None:
    """Evaluate the model outputs."""
    if results_file is None:
        model_output_dir = model_output_file.split("/")[-2]
        idx = os.path.splitext(model_output_dir)[0].split("_")[-1]
        results_file = f"evaluation/translation_results_0_{idx}.csv"

    # if args.results_file is None and results_file is None:
    #     model_output_dir = model_output_file.split("/")[-2]
    #     idx = os.path.splitext(model_output_dir)[0].split("_")[-1]
    #     results_file = f"evaluation/translation_results_0_{idx}.csv"
    # else:
    #     results_file = args.results_file

    if not skip_structuralize:
        print("Structuralizing the model outputs...")
        client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

        # Read the source sentences
        with open(source_sentences_file, "r") as f:
            source_sentences = f.read().strip()

        # Read the structuralization user prompt
        with open(eval_user_prompt_file, "r") as f:
            eval_user_prompt = f.read().strip()

        with open(model_output_file, "r") as f:
            model_output = f.read().strip()

        if label_file.endswith(".csv"):
            labels_df = pd.read_csv(label_file)
            labels = labels_df["label"].tolist()
        elif label_file.endswith(".txt"):
            with open(label_file, "r") as f:
                labels = f.read().strip()
        else:
            raise ValueError("Label file must be a .csv or .txt file")

        # format the user prompt
        input_text = f"### Source sentences\n{source_sentences}\n\n### Translation glosses\n{model_output}"

        if not eval_user_prompt.endswith("\n"):
            eval_user_prompt += "\n"

        user_prompt = eval_user_prompt + input_text

        # structuralize the output
        print(f"Structuralizing the output with {args.model}...")
        sentence_pair_list = structuralize(
            model=args.model,
            client=client,
            user_prompt=user_prompt,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            text_format=StructuralizedOutput,
            temperature=0.0
        )
        print(f"Structuralization completed. Found {len(sentence_pair_list)} sentence pairs.")

        # check if the number of sentence pairs matches the number of labels
        assert len(sentence_pair_list) == len(labels), (
            f"Number of structuralized sentence pairs ({len(sentence_pair_list)}) does not match "
            f"the number of labels ({len(labels)})."
        )

        with open(results_file, "w") as f:
            f.write("source_sentence,translation_gloss,label\n")
            for i, pair in enumerate(sentence_pair_list):
                if not args.no_print:
                    print(f"Source Sentence: {pair.source_sentence}")
                    print(f"Translation Gloss: {pair.translation_gloss}")
                    print(f"Reference Gloss: {labels[i]}")
                    print("-" * 40)

                f.write(f'"{pair.source_sentence}","{pair.translation_gloss}","{labels[i]}"\n')

        print(f"Structuralized output saved to {results_file}")

    assert os.path.exists(results_file), f"Results file {results_file} does not exist."

    if label_file.endswith(".csv"):
        labels_df = pd.read_csv(label_file)
        labels = labels_df["label"].tolist()
    elif label_file.endswith(".txt"):
        with open(label_file, "r") as f:
            labels = f.read().strip()
    else:
        raise ValueError("Label file must be a .csv or .txt file")

    # Now, run the evaluation.
    model_outputs_df = pd.read_csv(results_file)
    # source_sentences = model_outputs_df.source_sentence.tolist()
    predictions = model_outputs_df.translation_gloss.tolist()
    # convert nan to empty string (just in case)
    predictions = [pred if isinstance(pred, str) else "" for pred in predictions]
    references = labels
    print("predictions:", predictions)
    print("references:", references)

    metrics = MorphosyntaxMetrics()
    scores = metrics.compute_all_metrics(predictions, references)
    return scores


def main(args: argparse.Namespace) -> None:
    """Main function to evaluate the model outputs."""
    # some tests
    assert args.model_outputs_file.endswith(".csv"), (
        "Model outputs file must be a .csv file"
    )
    if not args.reference_glosses.endswith(".csv"):
        raise NotImplementedError(
            "Currently, only CSV files are supported for reference glosses."
        )
        # TODO: support Hugging Face datasets

    # Load model outputs
    model_outputs_df = pd.read_csv(args.model_outputs_file)

    # Load reference glosses
    reference_glosses_df = pd.read_csv(args.reference_glosses)

    # Assuming both dataframes have a 'gloss' column for the glosses
    model_glosses = model_outputs_df[MODEL_OUTPUT_COLUMN].tolist()
    reference_glosses = reference_glosses_df[MODEL_OUTPUT_COLUMN].tolist()

    # Calculate WER and CER
    wer_score = jiwer.wer(reference_glosses, model_glosses)
    cer_score = jiwer.cer(reference_glosses, model_glosses)

    # Morphological metrics
    morph_metrics = MorphosyntaxMetrics()
    ter_scores = [
        morph_metrics.compute_ter(ref, hyp)
        for ref, hyp in zip(reference_glosses, model_glosses)
    ]

    print(f"Word Error Rate (WER): {wer_score:.4f}")
    print(f"Character Error Rate (CER): {cer_score:.4f}")


def main_example():
    """Main function for running an example."""
    df = pd.read_csv("french_like.csv")
    refs = df["label"].tolist()
    refs = [[ref] for ref in refs]
    hyp = df["prediction_example"].tolist()

    metric = MorphosyntaxMetrics()

    # TER
    ter_score = metric.compute_ter(hyp, refs, normalized=False, no_punct=True, case_sensitive=False)
    print(f"TER: {ter_score.score:.4f}")

    # SER
    ser_score = metric.compute_ser(hyp, refs, normalized=False, no_punct=True, case_sensitive=False)
    print(f"SER: {ser_score.score:.4f}")

    # MFER
    mfer_score = metric.compute_mfer(hyp, refs, average="macro", multi_ref_denominator="best")
    print(f"MFER: {mfer_score:.4f}")

    # MTER
    mter_score = metric.compute_mser(hyp, refs, alpha=0.5)
    print(f"MTER: {mter_score:.4f}")

    # BLEU
    bleu_score = metric.compute_bleu(hyp, refs)
    print(f"BLEU: {bleu_score:.4f}")

    # ChrF++
    chrf_score = metric.compute_chrf(hyp, refs)
    print(f"ChrF++: {chrf_score:.4f}")

    # WER and CER
    refs = [ref[0] for ref in refs]  # flatten the list of lists
    wer_score = metric.compute_wer(hyp, refs)
    print(f"WER: {wer_score:.4f}")

    # CER
    cer_score = metric.compute_cer(hyp, refs)
    print(f"CER: {cer_score:.4f}")


if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.pipeline:
        scores = eval_output(
            model_output_file=args.model_outputs_file,
            label_file=args.reference_glosses,
            # source_sentences_file=args.source_sentences_file,
            eval_user_prompt_file=args.eval_user_prompt_file,
            skip_structuralize=args.skip_structuralize
        ) # eval_output prints the scores by default
        with open(args.scores_file, "w") as f:
            json.dump(scores, f, indent=4)
        print("Results saved to ", args.scores_file)

    elif args.test:
        print("Running unit tests...")
        unittest.main(argv=[""])
    elif args.run_example:
        main_example()
    else:
        main(args)
