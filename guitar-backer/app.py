import gradio as gr
import os
import numpy as np
import soundfile as sf
import shutil
from pathlib import Path
import tempfile

# 导入 v0.4 的函数
from backer import prepare_input, separate_tracks, analyze_song, ALL_STEMS

TMP_ROOT = Path(tempfile.gettempdir()) / "guitar_backer_tmp"
TMP_ROOT.mkdir(exist_ok=True)


def mix_single_backing(stems: dict, sr: int, out_dir: Path, keep_list):
    def sum_stems(names):
        acc = None
        for n in names:
            a = stems[n]
            acc = a.copy() if acc is None else acc + a
        return acc

    def _peak_normalize(audio, ceiling=0.99):
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio * (ceiling / peak)
        return audio

    suffix = "_".join(sorted(keep_list))
    out_name = f"backing_keep_{suffix}.wav"
    audio_out = _peak_normalize(sum_stems(keep_list))
    sf.write(out_dir / out_name, audio_out.T, sr)
    return out_name


def process_audio(input_audio, keep_stems):
    if input_audio is None:
        return None, None, "请先上传音频文件！"
    if not keep_stems:
        return None, None, "❌ 至少勾选一条音轨！"

    input_path = Path(input_audio)
    task_dir = TMP_ROOT / f"task_{os.urandom(6).hex()}"
    task_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = task_dir / "stems"

    try:
        audio_path = prepare_input(input_path)
        stems, sr = separate_tracks(audio_path, stems_dir)
        out_filename = mix_single_backing(stems, sr, task_dir, keep_stems)
        result_wav = task_dir / out_filename
        bpm, key = analyze_song(audio_path)

        guitar_track = stems_dir / "guitar.wav"
        info_text = (
            f"✅处理完成\n"
            f"BPM：{bpm:.1f}\n"
            f"调性：{key}\n"
            f"✅本次保留轨道：{keep_stems}\n"
            f"⚠️模型限制：htdemucs_6s 的吉他轨道包含全部吉他，无法区分主音/节奏吉他"
        )
        return str(result_wav), str(guitar_track), info_text

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return None, None, f"❌处理失败：{str(e)}\n{err_msg}"


with gr.Blocks(title="Guitar‑Backer 吉他伴奏生成器（自定义勾选版）") as demo:
    gr.Markdown("""
# 🎸 Guitar‑Backer 吉他练习伴奏生成工具
基于 htdemucs‑6s 六轨模型，**勾选想要保留的轨道**，自由组合伴奏
> 轨道列表：vocals人声｜drums鼓｜bass贝斯｜guitar吉他｜piano钢琴｜other其他合成器
> ⚠️重要限制：模型只能整体提取全部吉他，**节奏吉他会和主音Solo在同一个轨道，无法拆分**
""")

    audio_input = gr.Audio(
        label="上传歌曲音频",
        type="filepath",
        format="mp3"
    )

    keep_check = gr.CheckboxGroup(
        choices=ALL_STEMS,
        value=["drums", "bass", "piano", "other"],
        label="✅勾选【需要保留在伴奏里】的音轨",
        info="示例：去掉吉他+人声练习solo → 默认选中项"
    )

    run_btn = gr.Button("🚀开始处理", variant="primary")
    out_audio = gr.Audio(label="生成的伴奏（可直接下载）", type="filepath")
    out_guitar_stem = gr.Audio(label="提取出来的纯吉他轨道（用于扒谱）", type="filepath")
    info_box = gr.Textbox(label="运行信息 / BPM / 调性", interactive=False)

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, keep_check],
        outputs=[out_audio, out_guitar_stem, info_box]
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
