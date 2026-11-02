"""Samples clips from public Cantonese test sets into clips/ as <set>-p1..pN (16 kHz mono wav + txt).

  commonvoice : Common Voice 17.0 `yue` test split (fsicoli/common_voice_17_0 mirror, CC0). Volunteers reading a
                prompted sentence: many speakers, short clips. The reference is the prompt text.
  narration   : 張悦楷 reading 《三國演義》 (hon9kon9ize/zoengjyutgaai_saamgwokjinji, CC0). One professional
                storyteller, literary text, 306 clips.
  cantomap    : CantoMap (Winterstein, Tang & Lai, LREC 2020; GPL-3.0), Hong Kong Cantonese Map Task dialogues:
                two people giving map directions, so spontaneous speech with fillers and some English. Segmented
                and uploaded by safecantonese/cantomap; its references were converted from the corpus's Jyutping
                to characters with a dictionary. Sampled from that repo's test split (test.tsv + one 443 MB tar).
                Most segments are stereo with one microphone per speaker, and a reference covers only the speaker
                whose turn it is, so mixing to mono lets the other speaker in (the model then transcribes the wrong
                person). Each clip keeps only its louder channel (the speaker's own mic), chosen from the audio alone;
                segments whose channels are within CMAP_MIN_DB of each other are skipped as ambiguous.
                Three recordings whose audio doesn't match their transcripts are skipped entirely (CMAP_BROKEN), and
                so are single segments a person marked "broken" after listening (rules/cantomap_review.toml).
  mixed       : AlienKevin/mixed_cantonese_and_english_speech (Whisper_MCE): volunteers reading daily-life sentences
                that mix Cantonese and English, at their own pace. No license stated: results only, don't share clips.
  wordshk     : AlienKevin/wordshk_cantonese_speech (CC BY 4.0): words.hk dictionary example sentences, read aloud.
  gzdaily     : AlienKevin/guangzhou-daily-use-speech (MagicHub ASR-SCCantDuSC, CC BY-NC-ND 4.0): 10 Guangzhou speakers,
                scripted daily-use sentences, mainland spellings. No derivatives: results only, never the cut clips.
  gzcabin     : AlienKevin/guangzhou-cabin-speech (MagicHub ASR-SCCantCabSC, CC BY-NC-ND 4.0): 10 Guangzhou speakers in a
                car: digits, commands, queries. Same terms as gzdaily.
                These four are Parquet files with audio inside, read in 100-row groups: PQ_GROUPS groups are picked
                at random (seeded) and rows sampled within them, so only those groups are downloaded.

Sampling is seeded (default 42), so the same N gives the same clips. Provenance goes to clips/public_sources.json.

  python scripts/prepare_public.py --check          # print what would be downloaded; downloads nothing large
  python scripts/prepare_public.py -n 50            # download and write clips
  python scripts/prepare_public.py --counts commonvoice=300 narration=300
  python scripts/prepare_public.py --sets cantomap --counts cantomap=300
  python scripts/prepare_public.py --sets mixed wordshk gzdaily gzcabin -n 300

Common Voice's audio is one 73 MB tar of all 2,626 test clips (no way to fetch part of it), read once and not
extracted to disk; only the sampled clips are decoded. The narration set's wavs are fetched one file at a time,
so only the sampled ones are downloaded.
"""
import argparse
import csv
import json
import random
import subprocess
import tarfile
import tomllib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "clips"
CV = "fsicoli/common_voice_17_0"
NARR = "hon9kon9ize/zoengjyutgaai_saamgwokjinji"
CMAP = "safecantonese/cantomap"
CMAP_MIN_DB = 3.0  # louder channel must beat the other by this much, or the segment is skipped
# Recordings whose audio doesn't line up with their transcripts in safecantonese/cantomap (found 2026-10-20). Every
# sampled clip from each was wrong for every model (median CER 88-150%, mono mix and own-channel alike), while clips
# from other recordings were mostly fine. One clip's audio matched a different line of the same conversation, 4.5 min
# away from its own timestamp; the rest matched no line in their transcript. Whole recordings are skipped.
CMAP_BROKEN = {"160725_012_3_4_B1", "160815_002_17_18_B", "160815_002_17_18_D"}
CMAP_REVIEW = ROOT / "rules" / "cantomap_review.toml"  # single segments checked by listening; "broken" ones are skipped


def to_wav(src: bytes | Path, dest: Path, channel: int | None = None) -> float:
    """Any audio -> 16 kHz mono wav; returns seconds. channel keeps one input channel instead of mixing them all."""
    if isinstance(src, Path):
        cmd, data = ["ffmpeg", "-v", "error", "-y", "-i", str(src)], None
    else:
        cmd, data = ["ffmpeg", "-v", "error", "-y", "-i", "pipe:0"], src
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    subprocess.run(cmd + ["-ar", "16000", "-ac", "1", str(dest)], input=data, check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dest)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def narration_rows() -> list[tuple[str, str]]:
    path = hf_hub_download(NARR, "train.csv", repo_type="dataset")
    with open(path, encoding="utf-8", newline="") as f:  # no header row: wav, speaker, language, text
        return [(r[0].removeprefix("./"), r[3]) for r in csv.reader(f) if len(r) >= 4]


def check(counts: dict[str, int]) -> None:
    n = counts.get('narration', 50)
    api = HfApi()
    cv = {e.path: e.size for e in api.list_repo_tree(CV, "audio/yue/test", repo_type="dataset")}
    tsv = {e.path: e.size for e in api.list_repo_tree(CV, "transcript/yue", repo_type="dataset")}
    print(f"Common Voice yue test: audio tar {cv['audio/yue/test/yue_test_0.tar'] / 1e6:.1f} MB "
          f"+ test.tsv {tsv['transcript/yue/test.tsv'] / 1e6:.1f} MB  (whole tar is needed; it holds all 2,626 clips)")
    sizes = {e.path: e.size for e in api.list_repo_tree(NARR, "wavs", repo_type="dataset")}
    rows = narration_rows()
    picked = random.Random(42).sample(rows, min(n, len(rows)))
    mb = sum(sizes.get(w, 0) for w, _ in picked) / 1e6
    print(f"Narration: {len(rows)} clips in the set, {sum(sizes.values()) / 1e6:.1f} MB for all of them; "
          f"the {len(picked)} sampled ones are {mb:.1f} MB (downloaded one by one)")


def commonvoice(n: int, seed: int) -> list[dict]:
    tsv = hf_hub_download(CV, "transcript/yue/test.tsv", repo_type="dataset")
    tar_path = hf_hub_download(CV, "audio/yue/test/yue_test_0.tar", repo_type="dataset")
    with open(tsv, encoding="utf-8", newline="") as f:
        meta = {r["path"]: r["sentence"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    with tarfile.open(tar_path) as tf:  # the tar is untrusted: members are read into memory, never extracted
        members = {Path(m.name).name: m for m in tf if m.isfile() and Path(m.name).name in meta}
        names = sorted(members)
        out = []
        for i, name in enumerate(random.Random(seed).sample(names, min(n, len(names))), 1):
            cid = f"commonvoice-p{i}"
            secs = to_wav(tf.extractfile(members[name]).read(), OUT / f"{cid}.wav")
            (OUT / f"{cid}.txt").write_text(meta[name].strip() + "\n", encoding="utf-8")
            out.append({"id": cid, "source": f"{CV}:{name}", "seconds": round(secs, 1)})
    return out


def narration(n: int, seed: int) -> list[dict]:
    out = []
    for i, (wav, text) in enumerate(random.Random(seed).sample(narration_rows(), n), 1):
        cid = f"narration-p{i}"
        src = Path(hf_hub_download(NARR, wav, repo_type="dataset"))
        secs = to_wav(src, OUT / f"{cid}.wav")
        (OUT / f"{cid}.txt").write_text(text.strip() + "\n", encoding="utf-8")
        out.append({"id": cid, "source": f"{NARR}:{wav}", "seconds": round(secs, 1)})
    return out


def cantomap(n: int, seed: int) -> list[dict]:
    # test.tsv: the repo's 516 test segments (segments under 5 characters already dropped). It matches combined.tsv
    # except for two small cleanups. The repo's loading script isn't used: it adds nothing (no audio processing).
    tsv = hf_hub_download(CMAP, "transcript/yue/test.tsv", repo_type="dataset")
    tar_path = hf_hub_download(CMAP, "audio/yue/test/yue_test_0.tar", repo_type="dataset")
    with open(tsv, encoding="utf-8", newline="") as f:
        meta = {r["path"]: r["sentence"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    import io

    import numpy as np
    import soundfile as sf

    with tarfile.open(tar_path) as tf:  # untrusted: members are read into memory, never extracted
        members = {Path(m.name).name: m for m in tf if m.isfile() and Path(m.name).name in meta}
        names = sorted(members)
        random.Random(seed).shuffle(names)  # take the first n usable segments in this seeded order
        out, skipped, broken = [], 0, 0
        review = tomllib.loads(CMAP_REVIEW.read_text(encoding="utf-8"))["segments"] if CMAP_REVIEW.exists() else {}
        for name in names:
            if len(out) == n:
                break
            if name.rsplit("_ts", 1)[0] in CMAP_BROKEN or review.get(name) == "broken":
                broken += 1
                continue
            data = tf.extractfile(members[name]).read()
            x, _ = sf.read(io.BytesIO(data), always_2d=True)
            channel, margin = None, None
            if x.shape[1] > 1:
                rms = np.sqrt((x ** 2).mean(axis=0)) + 1e-12
                channel = int(rms.argmax())
                margin = float(20 * np.log10(rms.max() / np.sort(rms)[-2]))
                if margin < CMAP_MIN_DB:
                    skipped += 1
                    continue
            cid = f"cantomap-p{len(out) + 1}"
            secs = to_wav(data, OUT / f"{cid}.wav", channel)
            (OUT / f"{cid}.txt").write_text(meta[name].strip() + "\n", encoding="utf-8")
            out.append({"id": cid, "source": f"{CMAP}:{name}", "seconds": round(secs, 1), "channel": channel,
                        "marginDb": None if margin is None else round(margin, 1)})
    print(f"cantomap: test split has {len(names)} segments with a transcript; skipped {broken} from broken recordings or reviewed as broken "
          f"and {skipped} with channels within {CMAP_MIN_DB} dB")
    return out


PQ_GROUPS = 50  # row groups read per Parquet set; rows are spread evenly across them
PARQUET = {  # set -> (Hugging Face dataset, transcript column)
    "mixed": ("AlienKevin/mixed_cantonese_and_english_speech", "sentence"),
    "wordshk": ("AlienKevin/wordshk_cantonese_speech", "text"),
    "gzdaily": ("AlienKevin/guangzhou-daily-use-speech", "sentence"),
    "gzcabin": ("AlienKevin/guangzhou-cabin-speech", "sentence"),
}


def parquet_set(name: str, n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    repo, col = PARQUET[name]
    fs = HfFileSystem()
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    groups = []  # (file, row group index)
    for f in files:
        with fs.open(f"datasets/{repo}/{f}") as fh:
            groups += [(f, g) for g in range(pq.ParquetFile(fh).metadata.num_row_groups)]
    rng = random.Random(seed)
    picked = sorted(rng.sample(groups, min(PQ_GROUPS, len(groups))))
    per = -(-n // len(picked))  # rows per group, rounded up
    review_path = ROOT / "rules" / f"{name}_review.toml"  # rows a person marked "broken" after listening
    broken = {k for k, v in (tomllib.loads(review_path.read_text(encoding="utf-8"))["segments"].items()
                             if review_path.exists() else []) if v == "broken"}
    rows, spare = [], []  # (source, text, audio bytes); spare = unpicked rows of the same groups, for replacements
    for f, g in picked:
        with fs.open(f"datasets/{repo}/{f}") as fh:
            t = pq.ParquetFile(fh).read_row_group(g, columns=["audio", col]).to_pylist()
        keep = [i for i, r in enumerate(t) if (r[col] or "").strip().strip('"“”').strip()]
        chosen = rng.sample(keep, min(per, len(keep)))
        row = lambda i: (f"{repo}:{f}#group{g}/row{i}", t[i][col].strip().strip('"“”').strip(), t[i]["audio"]["bytes"])
        rows += [row(i) for i in chosen]
        spare += [row(i) for i in keep if i not in chosen]
    rng.shuffle(rows)
    rows = rows[:n]
    # A broken row is replaced in place, so every other clip keeps its id (and its cached results). Replacements come
    # from a separate seeded stream, so the main sample above is unchanged by the review.
    key = lambda src: src.split(":", 1)[1]
    spare = [r for r in spare if key(r[0]) not in broken]
    random.Random(seed + 1).shuffle(spare)
    rows = [r if key(r[0]) not in broken else spare.pop() for r in rows]
    out = []
    for i, (src, text, audio) in enumerate(rows, 1):
        cid = f"{name}-p{i}"
        secs = to_wav(audio, OUT / f"{cid}.wav")
        (OUT / f"{cid}.txt").write_text(text + "\n", encoding="utf-8")
        out.append({"id": cid, "source": src, "seconds": round(secs, 1)})
    print(f"{name}: {len(groups)} row groups in {len(files)} files; read {len(picked)}")
    return out


SETS = {"commonvoice": commonvoice, "narration": narration, "cantomap": cantomap}
SETS |= {name: (lambda name: lambda n, seed: parquet_set(name, n, seed))(name) for name in PARQUET}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50, help="clips per set (default for both)")
    ap.add_argument("--counts", nargs="+", default=[], metavar="SET=N", help="per-set override, e.g. commonvoice=300 narration=300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sets", nargs="+", default=["commonvoice", "narration"], choices=list(SETS))
    a = ap.parse_args()
    counts = {s: a.n for s in a.sets} | {k: int(v) for k, v in (x.split("=") for x in a.counts)}
    if a.check:
        return check(counts)
    OUT.mkdir(exist_ok=True)
    sources_path = OUT / "public_sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}  # keep other sets
    for s in a.sets:
        sources[s] = SETS[s](counts[s], a.seed)
        secs = sum(c["seconds"] for c in sources[s])
        print(f"{s}: {len(sources[s])} clips, {secs / 60:.1f} min, longest {max(c['seconds'] for c in sources[s])} s")
    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
