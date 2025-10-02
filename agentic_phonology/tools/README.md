# Ngram scoring tools for phonotactic models.

These tools require the use of the FST and NGram libraries, available from
[https://www.openfst.org/](https://www.openfst.org/) and
[https://www.opengrm.org/](https://www.opengrm.org/).

CSV files in `../data/ipas` are derived from Wiktionary resources for 221
languages. The final results of `create_lms.py` are the ngram language models in
`../data/ipas/*.mod`.

`score_generator.py` can be used to provide perplexity scores against various
languages for a phonotactic model produced by the LLM. Pass the python code for
the model as the `--phonotactics` argument.
