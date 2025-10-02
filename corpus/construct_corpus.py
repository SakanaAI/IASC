"""Corpus tools."""
import sys
import os

sys.path.append(os.path.abspath("."))

import corpus_management as lib
import lexicon as lex

from absl import app
from absl import flags
from absl import logging
from typing import Dict


SOURCES = flags.DEFINE_list("sources", [], "Source texts.")

GLOSSES = flags.DEFINE_list(
    "glosses",
    [],
    "Texts with morphosyntactic glosses.",
)

PHONOTACTICS = flags.DEFINE_string(
    "phonotactics",
    None,
    "Path to phonotactic model.",
)
ORTHOGRAPHY = flags.DEFINE_string(
    "orthography",
    None,
    "Path to orthographic model.",
)

OUTDIR = flags.DEFINE_string("outdir", None, "Output directory for JSONL.")
LEXICON = flags.DEFINE_string("lexicon", "", "Path to existing lexicon.")
SCRIPT_OUTDIR = flags.DEFINE_string(
    "script_outdir",
    None,
    "Optional output directory for orthographic lexicon.",
)


def main(unused_argv):
    L = len(SOURCES.value)
    assert len(GLOSSES.value) == L, "# of source files and gloss files must match"
    try:
        os.makedirs(OUTDIR.value)
    except FileExistsError:
        pass
    corpus = []
    for i in range(L):
        corpus.append(lib.interleave(SOURCES.value[i], GLOSSES.value[i]))
    lexicon = lex.Lexicon(
        PHONOTACTICS.value,
        LEXICON.value,
        orthography=ORTHOGRAPHY.value,
    )
    lexicon.construct_lexicon(corpus)
    lexicon.write(os.path.join(OUTDIR.value, "lexicon.tsv"))
    if SCRIPT_OUTDIR.value:
        lexicon.write_orthography(
            os.path.join(
                SCRIPT_OUTDIR.value,
                "lexicon_orth.tsv",
            ),
        )
    else:
        lexicon.write_orthography(os.path.join(OUTDIR.value, "lexicon_orth.tsv"))
    new_corpus = []
    for text in corpus:
        text = lexicon.populate_text(text)
        text = lexicon.spell_text(text)
        new_corpus.append(text)
    if SCRIPT_OUTDIR.value:
        lib.write_corpus(new_corpus, SCRIPT_OUTDIR.value)
    else:
        lib.write_corpus(new_corpus, OUTDIR.value)


if __name__ == "__main__":
    flags.mark_flag_as_required("outdir")
    flags.mark_flag_as_required("phonotactics")
    app.run(main)
