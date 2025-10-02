"""Wrapper for Phoible"""

import copy
import csv
import os
import random

from collections import defaultdict
from typing import Callable, Dict, Sequence, Union, Tuple

PHOIBLE_PATH = "data/phoible/dev/data/phoible.csv"

Features = Dict[str, str]
SFeatures = Sequence[str]


def _unicode_substitutions(phoneme):
  subst = {
    # One annoying feature of Phoible that I have run up again and again is that
    # it uses LATIN SMALL LETTER SCRIPT G (U+0261), which is IPA standard, but
    # usually identical in form to "g", the latter being much more convenient to
    # use.
    #
    # See, e.g.
    #
    # https://www.reddit.com/r/asklinguistics/comments/d89w8g/why_is_ipa_%C9%A1_a_different_character_than_latin/
    #
    # To that end, we prefer to use regular "g".
    chr(0x0261): "g",
  }
  if phoneme in subst:
    return subst[phoneme]
  return phoneme


def _to_strings(features: Features) -> SFeatures:
  return [f"{v}{f}" for f, v in features.items()]


def _to_features(features: SFeatures) -> Features:
  return {s[1:] : s[0] for s in features}


class Phoible:
  """Container class for Phoible data."""
  def __init__(self):
    self._ph_counts = defaultdict(int)
    self._ph_features = defaultdict(dict[str, str])
    self._ph_class = defaultdict(str)
    with open(PHOIBLE_PATH, "r") as stream:
      reader = csv.reader(stream, delimiter=",", quotechar='"')
      rows = [r for r in reader]
      hdr = rows[0]
      rest = rows[1:]
      name_to_idx_table = {}
      for idx, name in enumerate(hdr):
        name_to_idx_table[name] = idx
      self._feature_names = hdr[11:]
      for row in rest:
        phoneme = row[name_to_idx_table["Phoneme"]]
        features = {}
        for f in self._feature_names:
          setting = row[name_to_idx_table[f]]
          # Silly things like "+,-" and their ilk:
          if setting not in ["+", "-", "0"]:
            setting = setting[0]
          features[f] = setting
        phoneme = _unicode_substitutions(phoneme)
        self._ph_counts[phoneme] += 1
        self._ph_features[phoneme] = features
        self._ph_class[phoneme] = row[name_to_idx_table["SegmentClass"]]
    self._hist = [(self._ph_counts[ph], ph) for ph in self._ph_counts]
    self._hist.sort(reverse=True)

  def ph_features(self, phoneme: str) -> SFeatures:
    return _to_strings(self._ph_features[phoneme])

  def ph_class(self, phoneme: str) -> str:
    return self._ph_class[phoneme]

  def ph_count(self, phoneme: str) -> int:
    return self._ph_counts[phoneme]

  @property
  def feature_names(self) -> Sequence[str]:
    return self._feature_names

  def closest_phoneme(
      self,
      probe: SFeatures,
      subset: Sequence[str] = [],
  ) -> str:
    probe = _to_features(probe)

    def dist(probe, features):
      dist = 0
      for fname, fvalue in probe.items():
        assert fname in features, f"`{fname}` not in feature set"
        if features[fname] == fvalue:
          pass
        elif fvalue == "0" or features[fname] == "0":
          dist += 0.5
        else:
          dist += 1
      return dist

    minimum = 10_000
    phoneme = ""
    for ph, features in self._ph_features.items():
      if not features:
        continue
      if subset and ph not in subset:
        continue
      distance = dist(probe, features)
      if distance < minimum:
        minimum = distance
        phoneme = ph
    return phoneme, minimum

  def change_features(
      self,
      ph: str,
      feats_and_values: SFeatures,
      subset: Sequence[str] = [],
  ) -> str:
    features = self._ph_features[ph]
    if not features:
      return ""
    features = copy.deepcopy(features)
    for f, v in _to_features(feats_and_values).items():
      features[f] = v
    features = _to_strings(features)
    return self.closest_phoneme(features, subset)[0]

  def phonemes_with_features(
      self,
      feats_and_values: SFeatures,
      subset: Sequence[str] = [],
  ) -> Sequence[str]:
    feats_and_values = _to_features(feats_and_values)
    result = []
    for ph, features in self._ph_features.items():
      if subset and ph not in subset:
        continue
      keep = True
      for f, v in feats_and_values.items():
        if v == features[f]:
          pass
        else:
          keep = False
          break
      if keep:
        result.append(ph)
    return result

  def feature_diffs(self, ph1: str, ph2: str) -> Sequence[Tuple[str]]:
    f1 = self.ph_features(ph1)
    f2 = self.ph_features(ph2)
    result = []
    for f, v in f1.items():
      if f2[f] != v:
        result.append((f, v, f2[f]))
    return result

  @property
  def all_phonemes(self) -> Sequence[str]:
    return [p[1] for p in self._hist]

  def generate_phoneme_set(
      self,
      num_segments: int,
  ) -> Tuple[Sequence[int], Sequence[str]]:
    return (
      [p[0] for p in self._hist[:num_segments]],
      [p[1] for p in self._hist[:num_segments]],
    )


PHOIBLE = Phoible()
