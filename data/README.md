# Data

| Path | What | Count |
|---|---|---|
| `function_words.txt` | The 70 function words (Mosteller & Wallace list), one per line, in feature order | 70 |
| `function_words/human/<topic>/` | 71-dim count vectors for the human essays (full length) | 2,173 files |
| `function_words/gpt/<topic>/` | 71-dim count vectors for the GPT-4o-mini essays (full length) | 2,173 files |
| `gpt_essays/<topic>/` | Full text of the GPT-4o-mini essays we generated (one per human title) | 2,173 files |
| `titles.csv` | `topic, file, title` for every essay pair, so the pairing can be rebuilt | 2,173 rows |

Each feature file holds one comma-separated row: counts of the 70 function words in file order,
followed by the count of all other words. Topics are the 110 Aeon subject areas (Appendix A of the paper).

## Human essays (not redistributed)

The human essays come from the Aeon Essays dataset (Acharya, 2024) and are copyrighted by Aeon Media,
so their text is **not** in this repository. To rebuild them: obtain the dataset, select the essays whose
titles appear in `titles.csv`, and save each as `<topic>/<topic> - <n>.txt` mirroring `gpt_essays/`.
The derived feature vectors in `function_words/human/` are sufficient to run every study in the paper.

## Regenerating features

```sh
# full length
python3 code/python/extract_function_words.py --input <essays_dir> --output data/function_words/<human|gpt>
# first 200 words only (Study 2); written to a git-ignored folder
python3 code/python/extract_function_words.py --input <essays_dir> --output data/function_words_200/<human|gpt> --max-words 200
```

## How the GPT essays were generated

Each human essay's title was sent to `gpt-4o-mini` with the prompt in Section 3.1 of the paper
("…write a 1000-word essay on this subject…"), one request per title, no post-editing.
