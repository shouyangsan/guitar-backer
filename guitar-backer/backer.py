""" backer.py — 自动伴奏器 v0.4 (htdemucs_6s 六轨版 | 勾选式选择轨道)
=========================
六轨：vocals, drums, bass, guitar, piano, other
两种模式二选一：
  --keep：勾选【要保留】的轨道
  --remove：勾选【要去掉】的轨道
二选一，不能同时用；不填默认全部保留
示例：
python backer.py song.mp3 --keep drums bass piano other
python backer.py song.mp3 --remove vocals guitar
依赖：demucs, librosa, soundfile, numpy, torch, ncmdump
"""
from __future__ import annotations
ALL_STEMS = ["drums", "bass", "guitar", "piano", "other", "vocals"]

# HuggingFace国内镜像，解决Windows 10060连接超时
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import librosa
from demucs.api import Separator

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

def _to_2d(a: np.ndarray) -> np.ndarray:
    while a.ndim > 2:
        a = a[0]
    return a

def _peak_normalize(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (ceiling / peak)
    return audio

def prepare_input(input_path: Path) -> Path:
    if input_path.suffix.lower() != ".ncm":
        return input_path
    try:
        import ncmdump
    except ImportError:
        raise SystemExit(
            "检测到 .ncm 文件，但没装 ncmdump。\n请运行：pip install ncmdump"
        )
    out_dir = input_path.parent / f"decoded_{input_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _out(_src, meta):
        return str(out_dir / f"{input_path.stem}.{meta['format']}")

    decoded = ncmdump.dump(str(input_path), output_path=_out, skip=False)
    print(f"      .ncm 已解密 → {Path(decoded).name}")
    return Path(decoded)

def separate_tracks(input_path: Path, out_dir: Path) -> tuple[dict, int]:
    print("[1/3] 分离音轨中（首次会下载 htdemucs_6s 模型，约170MB）…")
    separator = Separator(model="htdemucs_6s")
    _, stems = separator.separate_audio_file(str(input_path))
    sr = separator.samplerate

    stems_np: dict = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in stems.items():
        audio = _to_2d(tensor.cpu().numpy())
        stems_np[name] = audio
        sf.write(out_dir / f"{name}.wav", audio.T, sr)
    print(f"      六轨已保存到 {out_dir}/")
    return stems_np, sr

def mix_single_backing(stems: dict, sr: int, out_dir: Path, keep_list: List[str]) -> None:
    print("[2/3] 生成伴奏中…")
    print(f"      ✅ 最终保留轨道：{keep_list}")
    if not keep_list:
        raise SystemExit("错误：至少需要保留一条音轨！")

    def sum_stems(names: list[str]) -> np.ndarray:
        acc = None
        for n in names:
            a = stems[n]
            acc = a.copy() if acc is None else acc + a
        return acc

    def save(name: str, audio: np.ndarray) -> None:
        sf.write(out_dir / name, _peak_normalize(audio).T, sr)

    suffix = "_".join(sorted(keep_list))
    out_name = f"backing_keep_{suffix}.wav"
    save(out_name, sum_stems(keep_list))
    print(f"      已生成唯一伴奏文件：{out_name}")

# ---------- BPM & Key ----------
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _detect_key(chroma_avg: np.ndarray) -> str:
    best = (-1.0, "?", "major")
    for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
        for i in range(12):
            rotated = np.roll(profile, i)
            score = float(np.corrcoef(chroma_avg, rotated)[0, 1])
            if not np.isfinite(score):
                score = -1.0
            if score > best[0]:
                best = (score, _PITCHES[i], mode)
    return f"{best[1]} {best[2]}"

def analyze_song(input_path: Path) -> tuple[float, str]:
    print("[3/3] 分析 BPM / 调性…")
    y, sr = librosa.load(str(input_path), sr=22050, mono=True)
    tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = chroma.mean(axis=1)
    key = _detect_key(chroma_avg)
    return tempo, key

def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动伴奏器 v0.4：六轨音源分离 | 二选一：--keep勾选保留 / --remove勾选移除，只输出一个伴奏"
    )
    parser.add_argument("input", type=Path, help="输入音频文件路径（mp3/wav/flac/ncm）")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="输出目录，默认在输入同目录下建 out_<文件名>/"
    )
    parser.add_argument(
        "--keep", nargs="+", default=None, choices=ALL_STEMS,
        help="勾选【保留】的轨道，可选：vocals drums bass guitar piano other"
    )
    parser.add_argument(
        "--remove", nargs="+", default=None, choices=ALL_STEMS,
        help="勾选【移除】的轨道，和--keep二选一"
    )

    args = parser.parse_args()

    # 互斥判断
    if args.keep is not None and args.remove is not None:
        parser.error("不能同时使用 --keep 和 --remove！二者选其一")
    elif args.keep is not None:
        final_keep = args.keep
    elif args.remove is not None:
        final_keep = [s for s in ALL_STEMS if s not in args.remove]
    else:
        # 两个参数都不给，默认全部保留
        final_keep = ALL_STEMS.copy()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"找不到输入文件：{input_path}")

    out_dir = (args.out or input_path.parent / f"out_{input_path.stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = out_dir / "stems"

    audio_path = prepare_input(input_path)
    stems, sr = separate_tracks(audio_path, stems_dir)
    mix_single_backing(stems, sr, out_dir, final_keep)
    tempo, key = analyze_song(audio_path)

    print("\n" + "=" * 44)
    print(f"  BPM  : {tempo:.1f}")
    print(f"  调性 : {key}")
    print(f"  输出 : {out_dir}")
    print("=" * 44)

if __name__ == "__main__":
    main()
