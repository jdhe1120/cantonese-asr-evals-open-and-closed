# cantonese-asr-evals-open-and-closed

Which speech-to-text model is best for Cantonese? We ran open and paid models on the same 2,283 clips and scored them by
character error rate.

**Built on the work of others:**

- **[hon9kon9ize's Cantonese ASR evals](https://github.com/hon9kon9ize/cantonese_asr_eval)**, continuing
  [AlienKevin's](https://github.com/AlienKevin/cantonese_asr_eval), inspired this round. We use their scoring method and
  all our test sets except CantoMap and LIDA.
- **[CantoCaptions](https://cantocaptions.com)** and **[rookes](https://github.com/rookes)** made two of the models tested
  here, the graph-folding rules behind our scoring, and the Hong Kong converter table.

[Other credits](#acknowledgements) are at the bottom.

## Best model by task

- **Best overall: [CantoCaptions v3](https://huggingface.co/rookes/cantocaptions-cantonese-asr)**, from the
  [CantoCaptions](https://cantocaptions.com) project ([code](https://github.com/rookes/cantocaptions-ai)), with
  Qwen3-ASR-1.7B close behind (the gap is within noise). CantoCaptions writes traditional out of the box; Qwen often writes
  simplified and needs a Hong Kong converter (`cantoeval/convert.py`).
- **Best paid API:** cantonese.ai, then Azure Speech standard (`zh-HK`), though the top open models beat both.

<!-- results:begin -->
| Task | Best | Within a point |
|---|---|---|
| Many Speakers | **Qwen3-ASR-1.7B prompted for traditional** 4.1% | Qwen3-ASR-1.7B 4.4%, CantoCaptions v2 4.5% |
| Storytelling | **Qwen3-ASR-1.7B** 8.2% | CantoCaptions v3 8.5%, Qwen3-ASR-1.7B prompted for traditional 8.6% |
| Conversation | **Qwen3-ASR-1.7B** 17.1% | Qwen3-ASR-1.7B prompted for traditional 17.3%, cantonese.ai 17.9% |
| Canto + Eng Mixed | **CantoCaptions v2** 8.0% | CantoCaptions v3 8.3% |
| Synthetic Voice | **Qwen3-ASR-1.7B** 6.4% | CantoCaptions v3 6.6%, CantoCaptions v2 7.3% |
| GZ Daily Use | **CantoCaptions v2** 2.6% | Qwen3-ASR-1.7B 2.7%, Qwen3-ASR-1.7B prompted for traditional 2.8% |
| GZ Car Commands | **cantonese.ai** 4.1% | Qwen3-ASR-1.7B prompted for traditional 4.5%, Qwen3-ASR-1.7B 4.7% |
| Stories by Level | **CantoCaptions v3** 3.2% | – |

Scored in simplified. The winners are the same models in traditional, where Qwen needs a converter; the GZ tasks have no traditional references. A model flagged on a task isn't listed for it.

## Overall

![Models ranked by character error rate](results/ranking.svg)

### Simplified

| Model | Type | Average | Many Speakers (300) † | Storytelling (300) † | Conversation (300) | Canto + Eng Mixed (300) | Synthetic Voice (300) | GZ Daily Use (300) | GZ Car Commands (300) | Stories by Level (183) |
|---|---|---|---|---|---|---|---|---|---|---|
| CantoCaptions v3 | open | **7.5%** | 4.5% | 8.5% | 18.9% | 8.3% | 6.6% | 2.9% | 5.3% | 3.2% |
| Qwen3-ASR-1.7B prompted for traditional | open | **7.8%** | 4.1% | 8.6% | 17.3% | 9.4% | 7.3% | 2.8% | 4.5% | 5.5% |
| Qwen3-ASR-1.7B | open | **7.8%** | 4.4% | 8.2% | 17.1% | 10.8% | 6.4% | 2.7% | 4.7% | 5.2% |
| CantoCaptions v2 | open | **8.0%** | 4.5% | 8.7% | 20.5% | 8.0% | 7.3% | 2.6% | 5.7% | 4.3% |
| cantonese.ai | closed | **8.5%** | 1.5% ⚠️ | 4.3% ⚠️ | 17.9% | 10.9% | 10.5% | 3.4% | 4.1% | 4.3% |
| SenseVoice-Small | open | **10.5%** | 6.5% | 11.2% | 20.3% | 17.5% | 8.7% | 3.5% | 6.0% | 7.1% |
| Azure Speech standard | closed | **11.2%** | 7.4% | 16.0% | 19.5% | 16.5% | 10.5% | 4.5% | 10.5% | 5.4% |
| GLM-ASR-Nano-2512 + Cantonese prompt | open | **16.2%** | 12.0% | 19.4% | 25.3% | 17.8% | 21.8% | 7.1% | 13.3% | 11.9% |
| GLM-ASR-Nano-2512 | open | **17.9%** | 13.8% | 19.8% | 29.5% | 17.0% | 25.4% | 7.8% | 14.9% | 12.9% |
| Gemini 3.5 Transcribe | closed | **18.9%** | 15.3% | 24.6% | 30.1% | 15.2% | 19.0% | 15.9% | 20.0% | 13.5% |
| MAI-Transcribe-2 | closed | **21.0%** | 12.3% | 28.8% | 22.1% | 14.5% | 27.0% | 17.9% | 17.9% | 26.5% |
| JackyHoCL Whisper fine-tune | open | unranked | 0.9% ⚠️ | 19.0% | 26.1% | 2.2% ⚠️ | 13.6% | 5.5% | 10.6% | 9.1% |

### Traditional

| Model | Type | Average | Many Speakers (300) † | Storytelling (299) † | Conversation (300) | Canto + Eng Mixed (243) | Synthetic Voice (300) | Stories by Level (182) |
|---|---|---|---|---|---|---|---|---|
| CantoCaptions v3 | open | **9.3%** | 4.7% | 8.7% | 19.1% | 8.0% | 6.8% | 3.2% |
| Qwen3-ASR-1.7B prompted for traditional + HK converter | open | **9.9%** | 4.5% | 9.0% | 17.5% | 9.1% | 7.5% | 5.7% |
| Qwen3-ASR-1.7B + HK converter | open | **10.0%** | 4.8% | 8.6% | 17.3% | 10.7% | 6.5% | 5.5% |
| CantoCaptions v2 | open | **10.0%** | 4.7% | 8.9% | 20.6% | 7.9% | 7.3% | 4.4% |
| cantonese.ai | closed | **10.8%** | 1.6% ⚠️ | 4.6% ⚠️ | 18.1% | 10.1% | 10.7% | 4.4% |
| SenseVoice-Small + HK converter | open | **13.4%** | 6.9% | 11.5% | 20.5% | 17.1% | 8.8% | 7.4% |
| Azure Speech standard | closed | **13.8%** | 7.7% | 16.6% | 22.8% | 16.1% | 10.6% | 5.8% |
| Gemini 3.5 Transcribe | closed | **19.3%** | 15.6% | 24.9% | 30.4% | 14.1% | 19.1% | 13.6% |
| GLM-ASR-Nano-2512 + Cantonese prompt | open | **20.3%** | 13.4% | 21.0% | 28.5% | 17.6% | 22.0% | 13.1% |
| MAI-Transcribe-2 | closed | **22.7%** | 12.9% | 29.7% | 22.2% | 14.3% | 27.6% | 26.7% |
| GLM-ASR-Nano-2512 | open | **22.9%** | 15.1% | 22.1% | 33.2% | 16.6% | 27.0% | 14.8% |
| JackyHoCL Whisper fine-tune | open | unranked | 0.9% ⚠️ | 19.8% | 26.3% | 2.1% ⚠️ | 14.7% | 9.4% |

Percentages are character error rates, graphs folded, digits → 中文. ⚠️ very likely trained on this test data. † left out of every Average. All eight scoring combinations and every transcript are in [`results/`](results/summary.md) (run `20261031T052510Z`).
<!-- results:end -->

**How to read this**

- A gap of a point or two is noise.
- **Average** is the plain mean of the groups no ranked model appears trained on, so ranked models share the same data.
  Groups marked † are left out, for all of them.
- **Simplified comes first, as in the community evals:** everything is converted to simplified, so a model's script doesn't
  count. Traditional scores what Hong Kong readers see, but has no Guangzhou groups, where Qwen is strong. The two aren't
  comparable, so only compare within a table.
- **Graphs folded** counts interchangeable spellings as equal (啦/喇/嘞). **Digits → 中文** counts 2016 and 二零一六 as equal.
- **Scoring changes the details, not the picture.** Across all eight combinations, CantoCaptions v3 and Qwen stay in the
  top four, cantonese.ai is the best paid API, and Azure is next.
- **⚠️ marks scores that can't be compared,** because the model very likely trained on that data, which is common with
  public datasets. cantonese.ai is ranked on the rest. The JackyHoCL Whisper fine-tune is unranked: its model card lists
  Common Voice and the Canto + Eng Mixed data as training data, and we have no stock Whisper beside it. Unlike a ranked model's
  flags, its flags don't remove a group from the average, which also keeps CantoCaptions' strong Canto + Eng Mixed result in.
- Conversation's references are rough, so judge that column by big gaps only.

More detail is in [`docs/details.md`](docs/details.md).

## Run it

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```
uv sync
uv run python -m cantoeval ui                        # browse the results and diffs; needs no audio (first load ~30 s)
uv run python -m cantoeval check                     # verify the clips and scoring
uv run python -m cantoeval run --models sensevoice   # transcribe and score
```

Transcribing needs the clips (below). Hosted models need keys: `cp .env.example .env`, then fill in what you run.

| Model | `.env` entries |
|---|---|
| `gemini` | `GEMINI_API` (paid) or `GEMINI_API_FREE_KEY` (free tier) |
| `azure-standard`, `mai` | `MAI_ENDPOINT` (`https://<resource>.cognitiveservices.azure.com`), `MAI_API_KEY` |
| `cantonese-ai` | `CANTONESE_AI_API_KEY` |

## Test data

| Task | Source | Clips | License | What it tests |
|---|---|---|---|---|
| Many Speakers (their Mixed) | [Mozilla Common Voice](https://commonvoice.mozilla.org) 17.0 `yue` test split, via [this mirror](https://huggingface.co/datasets/fsicoli/common_voice_17_0) | 300 | CC0 | Many speakers reading short prompts |
| Storytelling | 張悦楷 reading 《三國演義》, [hon9kon9ize's set](https://huggingface.co/datasets/hon9kon9ize/zoengjyutgaai_saamgwokjinji) | 300 | CC0 | One storyteller, literary text |
| Conversation | CantoMap, [Winterstein, Tang & Lai (2020)](https://github.com/gwinterstein/CantoMap), test split, via [safecantonese](https://huggingface.co/datasets/safecantonese/cantomap) | 300 | GPL-3.0 | Two people talking: spontaneous speech, fillers, some English |
| Canto + Eng Mixed (their Yue & Eng) | [Whisper_MCE](https://github.com/Shelton1013/Whisper_MCE), via [AlienKevin](https://huggingface.co/datasets/AlienKevin/mixed_cantonese_and_english_speech) | 300 | None stated | English words inside Cantonese sentences |
| Synthetic Voice | [words.hk](https://words.hk) example sentences, AWS Polly's Hiujin voice, via [AlienKevin](https://huggingface.co/datasets/AlienKevin/wordshk_cantonese_speech) | 300 | CC BY 4.0 | Hong Kong written Cantonese, wide vocabulary; synthetic speech |
| GZ Daily Use | MagicHub ASR-SCCantDuSC, via [AlienKevin](https://huggingface.co/datasets/AlienKevin/guangzhou-daily-use-speech) | 300 | CC BY-NC-ND 4.0 | Guangzhou accent, mainland spellings (咁/俾/啦) |
| GZ Car Commands | MagicHub ASR-SCCantCabSC, via [AlienKevin](https://huggingface.co/datasets/AlienKevin/guangzhou-cabin-speech) | 300 | CC BY-NC-ND 4.0 | Guangzhou accent in a car: digits, commands, queries |
| Stories by Level | [LIDA Stories Hong Kong](https://lidastories.net) (© LIDA故事香港), read by Stephanie Hung; [our cut](https://huggingface.co/datasets/jdhe1120/cantonese-lida-stories-speech) into pages, text normalised | 183 | CC BY 4.0 | Graded learner material, levels 1-5, one narrator |

Only Common Voice and CantoMap are held-out test splits. The other public sets have no test split, so models may have
trained on them. "Their" names are the community evals' labels. We use the same dataset under a clearer name.

### Getting the clips

```
uv run python scripts/prepare_public.py -n 300 --sets commonvoice narration cantomap mixed wordshk gzdaily gzcabin
uv run python scripts/fetch_lida.py
```

Sampling is seeded, so the same command gives the same clips. `scripts/prepare_stories.py` and
`scripts/prepare_norway.py` are how the LIDA clips were cut from the original mp3 and PDF files (they also need
`pdftotext`, from poppler); `fetch_lida.py` downloads the result.

Why each group behaves as it does, and which clips were excluded, is in [`docs/details.md`](docs/details.md#test-data).

## License

The code is MIT (`LICENSE`). Third-party files in `rules/` keep their own licenses, with each license file beside them.
The reference sentences in `results/` come from the datasets below and stay under those datasets' terms; that includes
CantoMap (GPL-3.0) and the Guangzhou sets (CC BY-NC-ND 4.0).

## Acknowledgements

- **Scoring.** The protocol (OpenCC t2s, strip punctuation and spaces, mean of per-utterance CER) follows the evals above.
- **CantoCaptions.** The models are [rookes/cantocaptions-cantonese-asr](https://huggingface.co/rookes/cantocaptions-cantonese-asr),
  trained on the community [CantoCaptions](https://github.com/notHulK11/CantoCaptions) subtitles.
- **Graph folding.** `rules/bench_conventions.toml` is by Jon Miller ([rookes](https://github.com/rookes)), MIT, used with
  permission. The lines in `cantoeval/normalize.py` that add simplified forms are adapted from the same project.
- **HK converter.** `rules/opencc_hk_cantocaptions.txt` is from [cantocaptions-ai](https://github.com/rookes/cantocaptions-ai)
  (BSD 2-Clause), built on [OpenCC](https://github.com/BYVoid/OpenCC)'s table (Apache-2.0).
- **Data.** Every dataset is credited in [Test data](#test-data).

