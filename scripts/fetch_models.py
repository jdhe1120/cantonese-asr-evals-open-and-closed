"""Download the WenetSpeech-Yue checkpoints (ASLP-lab/WSYue-ASR) into models/ (gitignored) and apply the fixes WeNet's
load_model needs. Finished models are skipped; checksums are verified.

    uv run python scripts/fetch_models.py [whisper-m-yue u2pp-yue sensevoice-yue]

  whisper-m-yue   whisper_medium_yue/ (3.5 GB): .pt -> final.pt, an empty units.txt (the Whisper tokenizer ignores it).
                  train.yaml already has input_dim / output_dim.
  u2pp-yue        u2pp_conformer_yue/ (0.55 GB): .pt -> final.pt, lang_char.txt also copied to units.txt (load_model
                  wants it; the BPE tokenizer finds bpe.model and lang_char.txt by name), fbank dither 0.1 -> 0.0
                  (training noise left on at inference makes the output random).
  sensevoice-yue  sensevoice_small_yue/ (2.8 GB), a FunASR model dir, unmodified.

"""
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPO, REV = "ASLP-lab/WSYue-ASR", "4a74a91430485d7456cb786f1d0ff0aba1185ded"
SHA = {"whisper-m-yue": "32b35310a5ab618b5f82c2fec4b7a212371e0f8b91dea66e5c6addfbd389af6e",
       "u2pp-yue": "eebf054fcc32ae5a8bdefaff78a62e244aae4592e09363e25564cac02435d982",
       "sensevoice-yue": "38940ad5ae8a580efef0069f6a9c38ac67656dd8f046af9bd8d162cea3448706"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(name: str, weights: Path) -> None:
    got = sha256(weights)
    if got != SHA[name]:
        raise SystemExit(f"{name}: {weights} has the wrong sha256; delete it and run again")
    print(f"{name}: ok ({weights.parent}, sha256 {got[:12]})")


def files(subdir: str, names: list[str]) -> Path:
    from huggingface_hub import hf_hub_download
    tmp = MODELS / ".download" / "WSYue-ASR"
    for f in names:
        if not (tmp / subdir / f).exists():
            hf_hub_download(REPO, f"{subdir}/{f}", revision=REV, local_dir=tmp)
    return tmp / subdir


def whisper() -> None:
    out = MODELS / "whisper-m-yue"
    if not (out / "final.pt").exists():
        src = files("whisper_medium_yue", ["whisper_medium_yue.pt", "train.yaml"])
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "train.yaml", out / "train.yaml")
        (out / "units.txt").write_text("")
        shutil.move(src / "whisper_medium_yue.pt", out / "final.pt")
    check("whisper-m-yue", out / "final.pt")


def u2pp() -> None:
    out = MODELS / "u2pp-yue"
    if not (out / "final.pt").exists():
        src = files("u2pp_conformer_yue", ["u2pp_conformer_yue.pt", "train.yaml", "bpe.model", "lang_char.txt"])
        out.mkdir(parents=True, exist_ok=True)
        for f in ("bpe.model", "lang_char.txt"):
            shutil.copy(src / f, out / f)
        shutil.copy(src / "lang_char.txt", out / "units.txt")
        yaml, n = re.subn(r"(\n\s+dither:) 0\.1", r"\1 0.0  # training-time noise off for inference",
                          (src / "train.yaml").read_text())
        assert n == 1, "u2pp train.yaml: expected one `dither: 0.1`"
        (out / "train.yaml").write_text(yaml)
        shutil.move(src / "u2pp_conformer_yue.pt", out / "final.pt")
    check("u2pp-yue", out / "final.pt")


def sensevoice() -> None:
    out = MODELS / "sensevoice-yue"
    if not (out / "model.pt").exists():
        names = ["model.pt", "config.yaml", "configuration.json", "am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"]
        src = files("sensevoice_small_yue", names)
        out.mkdir(parents=True, exist_ok=True)
        for f in names:
            shutil.move(src / f, out / f)
    check("sensevoice-yue", out / "model.pt")


if __name__ == "__main__":
    jobs = {"whisper-m-yue": whisper, "u2pp-yue": u2pp, "sensevoice-yue": sensevoice}
    for name in sys.argv[1:] or jobs:
        jobs[name]()
    shutil.rmtree(MODELS / ".download", ignore_errors=True)
