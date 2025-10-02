"""Lexicon management tools."""

import csv
import random
import subprocess
import re

from absl import logging
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import Dict, Sequence

AFFIX = re.compile(r"^[0-9A-Z]+$")
NO_TRANSLATE = re.compile(r"^[A-Z][a-z]+")
WORD = re.compile(r"[A-Za-z]")
PUNCT = re.compile(r"([\.,\?\"!:;])")
# Detect sentence boundary within a line:
SENTENCE_MARKER = re.compile(r"([\.\?!])( +)([^ ])")

Text = Sequence[Dict[str, Sequence[str]]]


## TODO(rws): This is duplicated in ../utils/common_utils.py
def load_python_source_module(path: str) -> ModuleType:
    """Loads Python source from path as a module.

    **NB**: This is potentially dangerous. Also load_module() is deprecated so we
    need to investigate how to use exec_module() and create_module(), which of
    course have a different interface.

    Args:
      path: path to the predefined Python code.
    Returns:
      A module.
    """
    return SourceFileLoader("python_source", path).load_module()


def _isaffix(morph):
    if morph.startswith("-"):
        return AFFIX.match(morph[1:])
    elif morph.endswith("-"):
        return AFFIX.match(morph[:-1])
    else:
        return AFFIX.match(morph)


def _notranslate(morph):
    return NO_TRANSLATE.match(morph) or PUNCT.match(morph)


class Lexicon:
    """Container for lexicon.

    Argument `phonotactics` is the python code to generate the phonotactics.
    """

    def __init__(
        self,
        phonotactics: str,
        existing_lexicon: str = "",
        min_phonetic_length: int = 5,
        orthography: str = "",
    ):
        self.entries = defaultdict(str)
        self.longest_word = 0
        self.affix_nominal_length = 3
        self.min_phonetic_length = min_phonetic_length
        self.phonotactics = phonotactics
        self.num_morphemes = 1_000_000
        self.phonetic_forms = defaultdict(list)
        self.longest_phonetic_form = 0
        self.orthography = orthography
        self.orthography_converter = None
        if existing_lexicon:
            self.read(existing_lexicon)
        self._run_phonotactics()
        self._setup_orthography()

    def _setup_orthography(self):
        if not self.orthography:
            return
        __orthography__ = load_python_source_module(self.orthography)
        self.orthography_converter = __orthography__.ipa_to_orthography

    def read(self, path):
        with open(path) as s:
            reader = csv.reader(s, delimiter="\t", quotechar='"')
            for row in reader:
                self.entries[row[0].strip()] = row[1].strip()

    def write(self, path):
        with open(path, "w") as s:
            writer = csv.writer(s, delimiter="\t", quotechar='"')
            for gloss in self.entries:
                writer.writerow([gloss, self.entries[gloss]])

    def write_orthography(self, path):
        if not self.orthography_converter:
            logging.warning("No orthography converter")
            return
        with open(path, "w") as s:
            writer = csv.writer(s, delimiter="\t", quotechar='"')
            for gloss in self.entries:
                form = self.entries[gloss]
                prefix = ""
                suffix = ""
                if form.startswith("-"):
                    prefix = "-"
                    form = form[1:].strip()
                if form.endswith("-"):
                    suffix = "-"
                    form = form[:-1].strip()
                try:
                    form = self.orthography_converter(form)
                except KeyError:
                    pass
                writer.writerow([gloss, f"{prefix}{form}{suffix}"])

    def _run_phonotactics(self):
        cmd = [
            "python3",
            self.phonotactics,
            f"--num_morphemes={self.num_morphemes}",
        ]
        process_output = subprocess.run(
            cmd,
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if "Error" in process_output.stderr:
            raise Exception(process_output.stderr)
        phonetic_forms = process_output.stdout
        phonetic_forms = [p.strip() for p in phonetic_forms.split("\n")]
        for p in phonetic_forms:
            p = p.split()
            L = len(p)
            self.phonetic_forms[L].append(" ".join(p))
            if L > self.longest_phonetic_form:
                self.longest_phonetic_form = len(p)

    def construct_lexicon(self, texts: Sequence[Text]):
        for text in texts:
            for elt in text:
                for word in elt["morphosyntax"]:
                    for morph in word:
                        if _isaffix(morph) or _notranslate(morph):
                            continue
                        if len(morph) > self.longest_word:
                            self.longest_word = len(morph)
        for text in texts:
            for elt in text:
                for word in elt["morphosyntax"]:
                    for morph in word:
                        if morph in self.entries or _notranslate(morph):
                            continue
                        elif WORD.search(morph):
                            self.create_morph(morph)

    def create_morph(self, morph: str):
        """Creates a phonetic form proportional in length to the English morph."""
        if _isaffix(morph):
            plen = random.randint(1, self.affix_nominal_length)
        else:
            length = len(morph)
            proportion = length / self.longest_word
            plen = int(proportion * self.longest_phonetic_form)
            if plen < self.min_phonetic_length:
                plen = self.min_phonetic_length

        phon = ""
        max_plen = max(self.phonetic_forms.keys(), default=0)
        attempts = 0
        while not phon and attempts < 100:
            if plen in self.phonetic_forms:
                phon = random.choice(self.phonetic_forms[plen])
            else:
                plen += 1
                if plen > max_plen:
                    plen = self.min_phonetic_length
            attempts += 1
        if not phon:
            logging.warning(
                "No phonetic form found for morph %s with length %d", morph, plen
            )
            phon = morph
        if _isaffix(morph):
            if morph.startswith("-"):
                phon = "- " + phon
            elif morph.endswith("-"):
                phon = phon + " -"
        self.entries[morph] = phon

    def populate_text(self, text: Text) -> Text:
        for elt in text:
            phonetic_forms = []
            for word in elt["morphosyntax"]:
                phonetics = []
                for morph in word:
                    ph = self.entries[morph]
                    if ph:
                        phonetics.append(ph)
                    else:
                        phonetics.append(morph)
                phonetics = " ".join(phonetics)
                if not phonetics:
                    phonetics = " ".join(word)
                phonetic_forms.append(phonetics)
            elt["deep_phonology"] = phonetic_forms
            assert len(elt["deep_phonology"]) == len(elt["morphosyntax"])
        return text

    def printed_text_form(self, orthographic_forms: Sequence[str]) -> str:
        printed_form = []
        for form in orthographic_forms:
            if PUNCT.match(form) and not form.endswith("_"):
                printed_form.append(f"_{form}")
            else:
                printed_form.append(form)
        printed_form = " ".join(printed_form)
        printed_form = printed_form.replace(" _", "").replace("_ ", "")
        printed_form = printed_form.strip()
        printed_form = SENTENCE_MARKER.sub(
            lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
            printed_form,
        )
        # Make sure the initial letter is capitalized.
        if printed_form:
            printed_form = printed_form[0].upper() + printed_form[1:]
        return printed_form

    def spell_text(self, text: Text) -> Text:
        if not self.orthography_converter:
            return text
        # Set up orthography

        def remove_syllable_markers(phonetics, marker="."):
            if phonetics == ".":
                return phonetics
            # TODO(rws): Sometimes the system decides to add syllable boundary markers
            # to the phonology generator, even if it wasn't asked to do that. So
            # far the symbol used has always been ".".  This may break the orthography
            # converter though. As a temporary expedient, remove these markers, though
            # the right solution ultimately is to have the LLM learn to deal with this
            # gracefully when it creates the orthographic converter.
            return " ".join([c.strip() for c in phonetics.split(marker)])

        for elt in text:
            orthographic_forms = []
            # TODO: Need to determine whether to use deep or surface.
            # Also we'll have to deal with things like English names somehow in the
            # phonological rules ...
            for phonetics in elt["deep_phonology"]:
                phonetics = remove_syllable_markers(phonetics)
                phonetics = [m.strip() for m in phonetics.split("-")]
                # First try spelling the whole word:
                try:
                    orthography = self.orthography_converter(" ".join(phonetics))
                except KeyError:
                    orthography = []
                    for subword in phonetics:
                        try:
                            ortho = self.orthography_converter(subword)
                        except KeyError:
                            ortho = subword
                        orthography.append(ortho)
                    orthography = "".join(orthography)
                orthographic_forms.append(orthography)
            elt["orthography"] = orthographic_forms
            assert len(elt["orthography"]) == len(elt["morphosyntax"])
            printed_form = self.printed_text_form(orthographic_forms)
            elt["printed_form"] = printed_form
        return text
