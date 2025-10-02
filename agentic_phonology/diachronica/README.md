From the licensing info at:

`https://chridd.nfshost.com/diachronica/index-diachronica.pdf`

```
This document is licensed under the Creative Commons Attribution-NonCommercial
ShareAlike (CC BY-NC-SA) 3.0 license. Visit http://creativecommons.org/licenses/1by-nc-sa/3.0/ for further details.
```

From this we extract the data as JSONL using `make_jsonl.py`.

Then the phoneme inventories for the daughter languages are estimated by using
the LLM to apply the rules to the parent language, and then iterating down the
tree. This results in `diachronica_expanded.jsonl`.

We then iterate the process, asking the LLM to catch any mistakes it may have
made the last time around. The output of this stage is
`diachronica_expanded_2.jsonl`.

See the script `run.sh`.
