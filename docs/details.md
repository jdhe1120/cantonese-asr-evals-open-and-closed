# Details

## Reading the results

- **Writing convention.** Qwen3-ASR (stock) writes mainland-style Cantonese spellings, the same convention as the
  Guangzhou references, so it is often perfect there even in the strict view; with interchangeable spellings folded it
  ties with CantoCaptions. That's a spelling match, not memorisation (its verbatim-match rate is in line with Azure's).
- **Script.** Stock Qwen3-ASR writes traditional on Common Voice but mostly simplified elsewhere, so unconverted it
  falls from about 8% to 34% on narration in traditional. Prompting it for traditional barely helps (the model doesn't
  follow instructions). Running its output through a Hong Kong converter afterwards does: it lands within about a point
  of its simplified score.

## Running

- Models: `cantocaptions-v3`, `cantocaptions`, `qwen3-asr-stock`, `qwen3-asr-trad`, `sensevoice`, `whisper-yue`, `glm-asr`,
  `glm-asr-yue` (same weights, Cantonese prompt), `azure-standard`, `gemini`, `mai`, `cantonese-ai`. Hugging Face models are pinned
  to a revision, so a new upload never mixes into old results.
- Every transcription is cached in `cache/<model>/`, keyed on the audio and the reference, so a model is only called
  for clips it hasn't done; re-running is cheap and paid APIs are never billed twice. `--force` ignores the cache.
- The results page lists every model: the ones in `--models` run live, the rest come from their cache (never called).
  `--only` lists just `--models`.
- `--model-clips MODEL=clip1,clip2` limits a model to some clips. For a rate-limited model it also sets the order, so
  one that runs out of quota partway has done the start of its list (so a list that alternates groups covers each).
- Rate-limited APIs (like `gemini`) pace themselves and stop for the day when their quota is used up;
  re-run later to continue. Gemini's limits reset at midnight Pacific.
- On a machine with little memory, run local models one at a time: each run keeps its models loaded.
- Results: `runs/latest.json` (live, what the page reads) and `runs/<id>.json|csv` per run.
- On the page, click a score for its diff; shift-click a second score in the same row, or use "compare with", to put
  two models side by side.

### Adding a model

Add `cantoeval/models/<name>.py` exposing `transcribe(wav_path) -> str` (optionally `pace()`, called before each
request, for rate-limited APIs), then register it in `cantoeval/models/__init__.py` (`REGISTRY`, `INFO`, `COLUMNS`).

### Bringing your own clips

Put pairs of `<id>.wav` (16 kHz mono recommended) and `<id>.txt` (the reference, UTF-8) in `clips/`. Clip ids look like
`<group>-p<N>`; `<group>-full` clips (a whole recording as one clip) are only listed with `--with-full`, and never averaged.
`scripts/audit_clips.py <group>` ranks a group's clips by how likely their reference or audio is wrong, from cached
results, so you know which to listen to.

## Scoring

`cantoeval/normalize.py`, `cantoeval/metrics.py`: NFKC, then OpenCC traditional-to-simplified, then punctuation,
symbols and spaces stripped; CER by Levenshtein distance. The headline is the mean of per-clip CER, with corpus CER
(all errors / all reference characters) beside it. Groups are scored separately; the README's Average is only a rough guide across them. Four views:

- **strict**
- **graphs folded**: interchangeable spellings count as equal (啦/喇/嘞, 俾/畀...; `rules/bench_conventions.toml`)
- **digits → 中文**: numbers written as digits become Chinese numerals before scoring (2016 → 二零一六, 40 → 四十)
- **both**

Each view also has a traditional version: the same steps, except that instead of converting to simplified, Hong Kong
and Taiwan variant spellings are folded together (OpenCC's HK and TW variant tables), so a simplified character is an
error. Only clips whose reference is traditional get these, so those columns can be smaller: no Guangzhou clips, 243 of 300
mixed clips (the rest are written in simplified), and 299 narration and 182 LIDA clips (each loses one clip whose
reference has a simplified character). Rows ending in
"+ HK converter" are a model's cached transcripts put through `cantoeval/convert.py` (OpenCC s2t, then a Cantonese
variant table) rather than a new run; they appear only in the traditional view.

A gap of a point or two between models is within noise at these clip counts; read the diffs.

## Known issues and gotchas

- **Look-alike radicals.** Text extracted from PDFs can use CJK Radicals Supplement characters (⻝ for 食, ⻑ for 長).
  NFKC doesn't fold these; `_RADICALS` in `normalize.py` maps the ones seen so far, and `check` fails on any other.
- **MAI-Transcribe-2** writes standard written Chinese (的 在 來), not spoken Cantonese, and no setting changes that.
  Azure's standard recognizer (same endpoint, no `enhancedMode`, locale `zh-HK`) writes spoken Cantonese.
- **Whisper on Common Voice.** The Cantonese Whisper fine-tune gets most Common Voice clips exactly right, almost
  certainly because it was trained on them. Don't compare models on that group with it.
- **Narration references** contain some errors.
- **Narration may be training data.** The same recordings, with the same transcripts (281 of our 300 references
  match word for word), are published as [CanCLID/zoengjyutgaai](https://huggingface.co/datasets/CanCLID/zoengjyutgaai),
  a CC0 corpus offered for ASR training. Any model trained on public Cantonese speech may have heard these clips, so
  read narration scores as an upper bound on how well a model handles this speaker.
- **cantonese.ai API**: Cloudflare rejects the default Python `urllib` User-Agent (403, error 1010); the live response
  is `{"data": {"transcription": ...}}`, not the documented `{"text": ...}`; the free tier allows about 2 requests a
  minute and 10 a day.

## Test data

Mixed, words.hk and the two Guangzhou sets are sampled from the Hugging Face copies used by the community Cantonese ASR benchmark
(AlienKevin, continued by hon9kon9ize): 50 of each dataset's 100-row blocks are picked at random (seeded) and 6 clips
taken from each, so only those blocks are downloaded. Some models may have been trained on these sets, as the Cantonese
Whisper fine-tune evidently was on Common Voice. The mixed Cantonese–English set states no license and the two
Guangzhou sets are CC BY-NC-ND 4.0 (no derivatives), so we never publish the cut clips. `results/` includes the Guangzhou
references and transcripts, as the community benchmark does; for the mixed set it has scores only.

**Mixed Cantonese–English: six clips excluded.** `scripts/audit_clips.py mixed` flagged six clips that every model
got 85-100% wrong; listening confirmed their audio doesn't say the reference. They were swapped for other rows from
the same blocks (`rules/mixed_review.toml`), so the group is still 300 clips. Five of the six sit in two 100-row
blocks of the upload (`train-00001` block 9: rows 54, 80, 87; `train-00000` block 6: rows 5, 8) and the sixth is
`train-00003` block 19 row 85; the other clips sampled from those blocks are fine. If you use that dataset, check
those rows.

### LIDA stories: graded by learner level, one voice

Twenty illustrated storybooks from [LIDA Stories Hong Kong](https://lidastories.net) (LIDA故事香港), written for adult
learners and graded from level 1 to 5 (級). Each story's narration is cut into one clip per page, with the page text as
the reference; `<story>-full` is the whole story as one clip, left out of results (`--with-full` to run it). The level of each story is in
`rules/story_levels.toml`, and the results page breaks the group down by level.

- **Purpose: evaluation by level, in one fixed voice.** All twenty are read by the same person (Stephanie Hung), so
  differences between levels come from the material, not the speaker. Use this group to compare models on learner
  Cantonese level by level; use Common Voice (many speakers) and CantoMap (conversation) to judge how a model copes
  with different voices. Published as
  [jdhe1120/cantonese-lida-stories-speech](https://huggingface.co/datasets/jdhe1120/cantonese-lida-stories-speech).
- **Higher level isn't harder for the models.** Levels 4-5 score better than 2-3 for most models. Low-level pages are
  very short (a few words), and one wrong character on a short clip is a large share of it; level 3 is only 18 clips.
  Read the levels as "how does a model do on this material", not "advanced Cantonese is harder to recognise".
- **Cutting.** Pages are split at the narrator's ~1.7 s pauses between pages; a story is skipped rather than cut in
  the wrong place when its pauses are ambiguous. Coming to Norway (`norway`) was cut at fixed times
  (`scripts/prepare_norway.py`).
- **Look-alike characters.** The PDFs' text layer uses Kangxi and CJK Radicals Supplement code points (⼆ ⽉, ⻝ ⻑)
  that look like ordinary characters; they are normalised before scoring (see [Known issues](#known-issues-and-gotchas)).

### CantoMap: noisy references, three recordings excluded

CantoMap is the only conversational group, and its references are rough. Compare models on it by ranking and large
gaps, not by exact CER. We sample it from [safecantonese/cantomap](https://huggingface.co/datasets/safecantonese/cantomap)
(`transcript/yue/test.tsv` plus the test audio tar). Known issues in that upload:

- **Two speakers, two channels.** Most segments are stereo with one microphone per speaker, and each reference covers
  only the speaker whose turn it is. Mixing to mono lets the other speaker in, and models then transcribe the wrong
  person. We keep each clip's louder channel (the speaker's own mic), chosen from the audio alone, and skip segments
  whose channels are within 3 dB of each other.
- **Words cut off at clip edges.** Segments are cut exactly at the annotation times, with no padding. In about 12% of
  clips every model misses the reference's opening words (e.g. 之後), and in about 7% the closing words, because they
  are not in the audio.
- **Characters picked from a dictionary.** The corpus is transcribed in Jyutping; the uploader converted it to
  characters with a dictionary, choosing by hand where a sound had several matches. Names and rare words can come out
  as characters nobody would write.
- **Made-up place names.** Map Task maps use invented landmark names (擁腰沙漠, 勁盛泳池...), which no model can be
  expected to spell the reference's way.
- **Three recordings don't match their transcripts, so we exclude them entirely:**

  | Recording | Evidence |
  |---|---|
  | `160725_012_3_4_B1` | One clip's audio matches a different line of the same conversation, about 4.5 minutes from its own timestamp |
  | `160815_002_17_18_B` | Every sampled clip wrong for every model (median CER 88-100%); audio matches no line of its transcript |
  | `160815_002_17_18_D` | Every sampled clip wrong for every model (median CER 94%); audio matches no line of its transcript |

  Clips from every other recording were mostly fine. If you use this upload, skip these recordings or check them
  first. The list is `CMAP_BROKEN` in `scripts/prepare_public.py`.
- **Single segments checked by listening.** `scripts/audit_clips.py cantomap` ranks clips by how likely their reference
  or audio is wrong (the best model still far off, every model missing the same edge words, the audio matching another
  line), from cached results. Suspects are then checked by ear, and the verdicts go in `rules/cantomap_review.toml`:
  `broken` segments (audio doesn't say the reference) are skipped; `cut-start`/`cut-end` and `ok` ones are kept as they
  are. Clips are never dropped just because models do badly on them.
