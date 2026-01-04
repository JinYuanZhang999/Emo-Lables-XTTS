# 用Resemblyzer计算说话人相似度(多个音频)（适配emospeech生成的音频）
"""
批量计算说话人相似度脚本（最终版）

功能：
1) 读取输入 CSV（metadata_test.csv，列：audio_file|text|speaker_name）。
2) audio_file 就是目标音频（target/GT）。
3) 用 convert_gt_path_to_gen_path(target_path) 得到对应生成音频路径（generated）。
4) Resemblyzer 提取两段音频 embedding，点积（=余弦相似度）。
5) 写输出 CSV + summary txt。
"""

# from resemblyzer import VoiceEncoder, preprocess_wav
# from pathlib import Path
# import numpy as np
# import csv
# import os
# import re


# # ==========================
# # 可根据需要修改的参数（在代码中写死）
# # ==========================
# # 输入的 CSV 文件路径（metadata_test.csv）
# INPUT_CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# CSV_DELIM = "|"

# # 当 CSV 中 audio_file 是相对路径时，用它作为前缀（若全是绝对路径可忽略）
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# # ✅ 生成音频固定目录（与你情感脚本一致）
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1803/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1774/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1591/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1453/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1446/generated/"
# GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_pretrain_4.1504/generated/"

# # 输出的 CSV 文件路径（会自动创建/覆盖）
# OUTPUT_CSV_PATH = os.path.join(GEN_DIR_FIXED, "spk_sim.csv")

# # 保存最终计算的平均相似度
# SUMMARY_TEXT_FILE = os.path.join(GEN_DIR_FIXED, "spk_sim_sum.txt")


# # ==========================
# # ✅ 与情感脚本一致的：GT/target -> generated 文件名转换规则
# # ==========================

# EMO2ID = {
#     "neutral": 0,
#     "angry": 1,
#     "happy": 2,
#     "sad": 3,
#     "surprise": 4,
# }

# DATASET_OFFSETS = {
#     "ESD": 0,
#     "generate_dataset_200": 350,
#     "generate_dataset_600": 550,
# }


# def convert_gt_path_to_gen_path(gt_path: str, gen_dir: str = GEN_DIR_FIXED) -> str:
#     """
#     按你描述的规则，将目标(=GT)音频路径转换为生成音频路径。

#     示例：
#       /mnt/hd/zjy/wavs/ESD/0020/Angry/0020_000299.wav
#     ->/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1803/generated/10_299_1.wav
#     """
#     gt_path = os.path.normpath(gt_path)

#     # 1) 判断数据集与 offset
#     dataset_name = None
#     for ds in DATASET_OFFSETS.keys():
#         if f"{os.sep}{ds}{os.sep}" in gt_path:
#             dataset_name = ds
#             break
#     if dataset_name is None:
#         raise ValueError(
#             f"Cannot determine dataset from gt_path: {gt_path}. "
#             f"Expected one of: {list(DATASET_OFFSETS.keys())}"
#         )
#     offset = DATASET_OFFSETS[dataset_name]

#     # 2) 判断情感目录并映射到 emo_id（大小写不敏感）
#     gt_lower = gt_path.lower()
#     emo_dir = None
#     for emo in ["neutral", "angry", "happy", "sad", "surprise"]:
#         if f"{os.sep}{emo}{os.sep}" in gt_lower:
#             emo_dir = emo
#             break
#     if emo_dir is None:
#         raise ValueError(
#             f"Cannot determine emotion category from gt_path: {gt_path}. "
#             "Expected one of Neutral/Angry/Happy/Sad/Surprise as a directory name."
#         )
#     emo_id = EMO2ID[emo_dir]

#     # 3) 解析文件名 speaker 与 idx：0020_000299.wav
#     base = os.path.basename(gt_path)
#     m = re.match(r"^(\d+)_([0-9]+)\.wav$", base, flags=re.IGNORECASE)
#     if not m:
#         raise ValueError(
#             f"Unexpected filename format: {base}. Expected like 0020_000299.wav"
#         )
#     spk_str, idx_str = m.group(1), m.group(2)

#     spk = int(spk_str)   # 去前导 0
#     idx = int(idx_str)   # 去前导 0

#     spk_new = spk - 10
#     if spk_new < 0:
#         raise ValueError(f"speaker_id - 10 becomes negative for {base}: {spk} -> {spk_new}")

#     idx_new = idx + offset

#     # 4) 输出文件名：{spk_new}_{idx_new}_{emo_id}.wav
#     gen_filename = f"{spk_new}_{idx_new}_{emo_id}.wav"
#     gen_path = os.path.normpath(os.path.join(gen_dir, gen_filename))
#     return gen_path


# def main():
#     # ==========================
#     # 1. 初始化声纹编码器
#     # ==========================
#     print("初始化 VoiceEncoder（第一次运行会自动下载模型，可能稍慢）...")
#     encoder = VoiceEncoder()
#     print("VoiceEncoder 初始化完成。")

#     similarities = []
#     output_rows = []

#     # ==========================
#     # 2. 读取输入 CSV（metadata_test.csv）
#     # ==========================
#     if not os.path.exists(INPUT_CSV_PATH):
#         raise FileNotFoundError(f"找不到输入 CSV 文件：{INPUT_CSV_PATH}")

#     with open(INPUT_CSV_PATH, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter=CSV_DELIM)

#         required_cols = {"audio_file", "text", "speaker_name"}
#         missing = required_cols - set(reader.fieldnames or [])
#         if missing:
#             raise ValueError(f"CSV 缺少必要列: {missing}，当前列名: {reader.fieldnames}")

#         # ==========================
#         # 3. 逐行处理
#         # ==========================
#         for idx, row in enumerate(reader, start=1):
#             # target/GT 音频：来自 metadata_test.csv 的 audio_file
#             tgt_path_raw = (row["audio_file"] or "").strip()
#             speaker_name = (row["speaker_name"] or "").strip()

#             # target 路径绝对化
#             if os.path.isabs(tgt_path_raw):
#                 tgt_path = os.path.normpath(tgt_path_raw)
#             else:
#                 tgt_path = os.path.normpath(os.path.join(BASE_GT_PREFIX, tgt_path_raw))

#             # ✅ 用转换规则从 target 得到生成音频路径
#             try:
#                 gen_path = convert_gt_path_to_gen_path(tgt_path, gen_dir=GEN_DIR_FIXED)
#             except Exception as e:
#                 print(f"\n[{idx}] [ERROR] target->generated 路径转换失败，跳过。")
#                 print(f"  target: {tgt_path}")
#                 print(f"  err   : {e}")
#                 continue

#             print(f"\n[{idx}] 处理样本：")
#             print(f"  目标音频(target): {tgt_path}")
#             print(f"  生成音频(gen)  : {gen_path}")
#             print(f"  说话人         : {speaker_name}")

#             # 文件存在性检查
#             if not os.path.exists(tgt_path):
#                 print(f"  [WARN] 找不到目标音频文件，跳过该样本。")
#                 continue
#             if not os.path.exists(gen_path):
#                 print(f"  [WARN] 找不到生成音频文件，跳过该样本。")
#                 continue

#             try:
#                 wav_target = preprocess_wav(Path(tgt_path))
#                 wav_generated = preprocess_wav(Path(gen_path))

#                 emb_target = encoder.embed_utterance(wav_target)
#                 emb_generated = encoder.embed_utterance(wav_generated)

#                 similarity = float(np.dot(emb_target, emb_generated))

#                 print(f"  相似度 (cosine similarity) = {similarity:.6f}")

#                 similarities.append(similarity)
#                 output_rows.append(
#                     {
#                         "generated_audio_file": gen_path,
#                         "target_audio_file": tgt_path,
#                         "speaker_name": speaker_name,
#                         "similarity": similarity,
#                     }
#                 )

#             except Exception as e:
#                 print(f"  [ERROR] 处理该样本时出错，已跳过。错误信息: {e}")
#                 continue

#     # ==========================
#     # 4. 计算整体平均相似度
#     # ==========================
#     if len(similarities) == 0:
#         summary_text = (
#             "===========================================\n"
#             "未成功计算任何一对音频的说话人相似度，请检查输入数据。\n"
#             "===========================================\n"
#         )
#         print(summary_text)
#     else:
#         avg_similarity = float(np.mean(similarities))
#         summary_text = (
#             "===========================================\n"
#             f"共成功计算 {len(similarities)} 对音频的说话人相似度。\n"
#             f"平均说话人相似度: {avg_similarity:.6f}\n"
#             "===========================================\n"
#         )
#         print(summary_text)

#     # 写 summary txt
#     out_sum_dir = os.path.dirname(SUMMARY_TEXT_FILE)
#     if out_sum_dir and not os.path.exists(out_sum_dir):
#         os.makedirs(out_sum_dir, exist_ok=True)

#     with open(SUMMARY_TEXT_FILE, "w", encoding="utf-8") as f_sum:
#         f_sum.write(summary_text)

#     print(f"已将总体结果写入：{SUMMARY_TEXT_FILE}\n")

#     # ==========================
#     # 5. 写输出 CSV
#     # ==========================
#     fieldnames = ["generated_audio_file", "target_audio_file", "speaker_name", "similarity"]

#     out_dir = os.path.dirname(OUTPUT_CSV_PATH)
#     if out_dir and not os.path.exists(out_dir):
#         os.makedirs(out_dir, exist_ok=True)

#     with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
#         writer = csv.DictWriter(f_out, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in output_rows:
#             writer.writerow(r)

#     print(f"已将每对音频的相似度写入：{OUTPUT_CSV_PATH}")
#     print("脚本运行结束。")


# if __name__ == "__main__":
#     main()









# # 用Resemblyzer计算说话人相似度(多个音频)（适用etd-tts生成音频）
# 用 Resemblyzer 计算说话人相似度（多个音频）
# ✅ 新版：GEN 音频路径不再通过“替换 GT 路径片段”得到，而是
#     1) 从 metadata.csv 读取 GT 路径（audio_file）
#     2) 从 GT 路径解析 subset + emotion
#     3) 从 GT 文件名按 offset 规则得到 new_idx
#     4) 生成 GEN 文件名：{spk}_{Emotion}_{spk}_{new_idx:06d}_generated_e2e.wav
#     5) GEN 路径 = GEN_ROOT / GEN 文件名
#
# 例：
#   GT:  /mnt/hd/zjy/wavs/ESD/0020/Angry/0020_000299.wav
#   -> subset=ESD, emotion=Angry, old_idx=299, offset=1150, new_idx=1449
#   -> gen_filename = 0020_Angry_0020_001449_generated_e2e.wav
#   -> gen_path = /mnt/hd/zjy/test/etd-tts_no-ft-hifigan/0020_Angry_0020_001449_generated_e2e.wav

# from resemblyzer import VoiceEncoder, preprocess_wav
# from pathlib import Path
# import numpy as np
# import csv
# import os
# import re
# from typing import List, Tuple, Dict, Any


# # ==========================
# # 参数区（按你实际情况修改）
# # ==========================

# # metadata_test.csv（至少包含 audio_file 列；speaker_name/text 可选）
# CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# CSV_DELIM = "|"   # metadata 用 "|" 分隔

# # 当 CSV 中 audio_file 是相对路径时，用它作为前缀
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# # ✅ 新规则：GEN 音频固定根目录（与“最新版情感相似度脚本”一致）
# # GEN_ROOT = "/mnt/hd/zjy/test/etd-tts_no-ft-hifigan/"
# GEN_ROOT = "/mnt/hd/zjy/test/etd-tts_ft-hifigan/"

# # 输出目录（建议也放 GEN_ROOT 或单独目录都行）
# OUTPUT_DIR = GEN_ROOT
# OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "spk_sim.csv")
# SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "spk_sim_sum.txt")


# # ============================================================
# # 1) GT -> GEN 文件名映射（offset + 前后缀规则）
# # ============================================================

# # 目录中可能出现的情感（Title Case）
# EMOTIONS_TITLE = {"Neutral", "Angry", "Happy", "Sad", "Surprise"}

# # 你给定的 offset 规则（最终版本）
# OFFSETS = {
#     "ESD": {
#         "Neutral": 0,     # 1-350 不处理
#         "Angry": 1150,    # 1151-1500
#         "Happy": 2300,    # 2301-2650
#         "Sad": 3450,      # 3451-3800
#         "Surprise": 4600  # 4601-4950
#     },
#     "generate_dataset_200": {
#         "Neutral": 350,   # 351-550
#         "Angry": 1500,    # 1501-1700
#         "Happy": 2650,    # 2651-2850
#         "Sad": 3800,      # 3801-4000
#         "Surprise": 4950  # 4951-5150
#     },
#     "generate_dataset_600": {
#         "Neutral": 550,   # 551-1150
#         "Angry": 1700,    # 1701-2300
#         "Happy": 2850,    # 2851-3450
#         "Sad": 4000,      # 4001-4600
#         "Surprise": 5150  # 5151-5750
#     },
# }

# # GT 文件名格式：0020_000299.wav
# FNAME_RE = re.compile(r"^(?P<spk>\d{4})_(?P<idx>\d{6})\.wav$", re.IGNORECASE)


# def detect_subset_from_gt_path(gt_path: str) -> str:
#     """
#     从 GT 路径推断 subset。

#     假设你的 GT 路径结构类似：
#         .../wavs/<subset>/<speaker>/<emotion>/<file.wav>
#     例如：
#         /mnt/hd/zjy/wavs/ESD/0020/Angry/0020_000299.wav
#         /mnt/hd/zjy/wavs/generate_dataset_600/0020/Angry/0020_000120.wav

#     实现方式：
#         - 找到路径中名为 "wavs" 的那一段
#         - subset = "wavs" 后面的下一段
#     """
#     parts = gt_path.split("/")
#     try:
#         i = parts.index("wavs")
#         subset = parts[i + 1]
#     except Exception:
#         raise ValueError(f"Cannot detect subset from GT path (expect .../wavs/<subset>/...): {gt_path}")

#     if subset not in OFFSETS:
#         raise ValueError(f"Unknown subset '{subset}' from GT path: {gt_path}")
#     return subset


# def detect_emotion_from_gt_path(gt_path: str) -> str:
#     """
#     从 GT 路径目录名中找 emotion（Neutral/Angry/Happy/Sad/Surprise）。

#     返回 Title Case（例如 'Angry'），以便用于：
#         - 查 OFFSETS[subset][emotion]
#         - 以及拼接到最终 GEN 文件名中
#     """
#     for p in gt_path.split("/"):
#         if p in EMOTIONS_TITLE:
#             return p
#     raise ValueError(f"Cannot detect emotion from GT path (expect one of {sorted(EMOTIONS_TITLE)} in path): {gt_path}")


# def build_gen_filename_from_gt_filename(gt_filename: str, subset: str, emotion: str) -> str:
#     """
#     将 GT 文件名映射为 GEN 文件名（你的最新命名规则）：

#     输入:
#         gt_filename: 0020_000299.wav
#         subset:      ESD / generate_dataset_200 / generate_dataset_600
#         emotion:     Neutral/Angry/Happy/Sad/Surprise（Title Case）

#     输出:
#         0020_Angry_0020_001449_generated_e2e.wav
#     """
#     m = FNAME_RE.match(gt_filename)
#     if not m:
#         raise ValueError(f"Bad GT filename format (expect ####_######.wav): {gt_filename}")

#     spk = m.group("spk")
#     old_idx = int(m.group("idx"))

#     # 按规则获取 offset
#     offset = OFFSETS[subset][emotion]
#     new_idx = old_idx + offset

#     # “原规则”新核心名：0020_001449（不带 .wav）
#     core = f"{spk}_{new_idx:06d}"

#     # 你指定的最终文件名
#     final_name = f"{spk}_{emotion}_{core}_generated_e2e.wav"
#     return final_name


# # ============================================================
# # 2) 读取 CSV -> 构造 GT / GEN 路径（✅新版规则）
# # ============================================================

# def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str, gen_root: str):
#     """
#     从 metadata.csv 构造 gt_paths / gen_paths / rows

#     GT 构造规则：
#         - audio_file 是绝对路径：直接用
#         - audio_file 是相对路径：base_gt_prefix + audio_file

#     GEN 构造规则（新版）：
#         - 从 GT 路径解析 subset + emotion
#         - 从 GT 文件名按 offset 计算新 idx，并拼接新文件名：
#               0020_000299.wav -> 0020_Angry_0020_001449_generated_e2e.wav
#         - GEN 路径 = gen_root / gen_filename
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"找不到 CSV：{csv_path}")

#     gt_paths: List[str] = []
#     gen_paths: List[str] = []
#     rows: List[Dict[str, str]] = []

#     gen_root = os.path.normpath(gen_root)

#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter=csv_delim)

#         if not reader.fieldnames or "audio_file" not in reader.fieldnames:
#             raise ValueError(f"CSV 必须包含 audio_file 列，当前列名: {reader.fieldnames}")

#         for row in reader:
#             rows.append(row)
#             af = (row.get("audio_file") or "").strip()
#             if not af:
#                 raise ValueError("发现空的 audio_file，请检查 CSV")

#             # 1) 构造 GT 路径
#             if os.path.isabs(af):
#                 gt = af
#             else:
#                 gt = os.path.join(base_gt_prefix, af)
#             gt = os.path.normpath(gt)
#             gt_paths.append(gt)

#             # 2) 构造 GEN 路径（新版：固定 gen_root + 文件名映射）
#             subset = detect_subset_from_gt_path(gt)
#             emotion = detect_emotion_from_gt_path(gt)
#             gt_filename = os.path.basename(gt)

#             gen_filename = build_gen_filename_from_gt_filename(gt_filename, subset, emotion)
#             gen = os.path.join(gen_root, gen_filename)
#             gen = os.path.normpath(gen)
#             gen_paths.append(gen)

#     return gt_paths, gen_paths, rows


# # ============================================================
# # 3) 主流程：Resemblyzer 计算 embedding + cosine similarity
# # ============================================================

# def main():
#     # 1) 初始化声纹编码器
#     print("初始化 VoiceEncoder（第一次运行会自动下载模型，可能稍慢）...")
#     encoder = VoiceEncoder()
#     print("VoiceEncoder 初始化完成。")

#     # 2) 从 metadata_test.csv 构造 gt/gen 路径（✅新版规则）
#     print(f"读取 CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#     gt_paths, gen_paths, rows = build_paths_from_csv(
#         CSV_PATH, CSV_DELIM, BASE_GT_PREFIX, GEN_ROOT
#     )

#     # 打印前 5 条映射确认
#     print("\nExample mapping (first 5):")
#     for i in range(min(5, len(gt_paths))):
#         print(f"  GT : {gt_paths[i]}")
#         print(f"  GEN: {gen_paths[i]}")

#     similarities: List[float] = []
#     output_rows: List[Dict[str, Any]] = []

#     # 3) 逐对计算相似度
#     for idx, (gt_path, gen_path, row) in enumerate(zip(gt_paths, gen_paths, rows), start=1):
#         speaker_name = (row.get("speaker_name") or "").strip()  # 可选列

#         print(f"\n[{idx}] 处理样本：")
#         print(f"  生成音频: {gen_path}")
#         print(f"  目标音频: {gt_path}")
#         if speaker_name:
#             print(f"  说话人  : {speaker_name}")

#         # --- 文件存在性检查（缺失则跳过，不中断整批） ---
#         if not os.path.exists(gen_path):
#             print("  [WARN] 找不到生成音频文件，跳过该样本。")
#             continue
#         if not os.path.exists(gt_path):
#             print("  [WARN] 找不到目标音频文件，跳过该样本。")
#             continue

#         try:
#             # preprocess_wav：会做采样率统一、响度归一等预处理
#             wav_gt = preprocess_wav(Path(gt_path))
#             wav_gen = preprocess_wav(Path(gen_path))

#             # embed_utterance：提取 256-D speaker embedding（已归一化，点积=cosine）
#             emb_gt = encoder.embed_utterance(wav_gt)
#             emb_gen = encoder.embed_utterance(wav_gen)

#             # resemblyzer 的 embedding 已归一化，可直接 dot 得到 cosine similarity
#             similarity = float(np.dot(emb_gt, emb_gen))
#             print(f"  相似度 (cosine similarity) = {similarity:.6f}")

#             similarities.append(similarity)
#             output_rows.append({
#                 "gen_path": gen_path,
#                 "gt_path": gt_path,
#                 "speaker_name": speaker_name,
#                 "similarity": similarity,
#             })

#         except Exception as e:
#             print(f"  [ERROR] 处理该样本时出错，已跳过。错误信息: {e}")
#             continue

#     # 4) 汇总平均
#     os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

#     if len(similarities) == 0:
#         avg_similarity = float("nan")
#         summary_text = (
#             "===========================================\n"
#             "未成功计算任何一对音频的说话人相似度，请检查输入数据。\n"
#             "===========================================\n"
#         )
#     else:
#         avg_similarity = float(np.mean(similarities))
#         summary_text = (
#             "===========================================\n"
#             f"共成功计算 {len(similarities)} 对音频的说话人相似度。\n"
#             f"平均说话人相似度: {avg_similarity:.6f}\n"
#             "===========================================\n"
#         )

#     print("\n" + summary_text)
#     with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f_sum:
#         f_sum.write(summary_text)
#     print(f"已将总体结果写入：{SUMMARY_TXT_PATH}\n")

#     # 5) 写出 CSV
#     fieldnames = ["gen_path", "gt_path", "speaker_name", "similarity"]
#     with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
#         writer = csv.DictWriter(f_out, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in output_rows:
#             # similarity 也可以格式化保留 6 位
#             writer.writerow({
#                 "gen_path": r["gen_path"],
#                 "gt_path": r["gt_path"],
#                 "speaker_name": r.get("speaker_name", ""),
#                 "similarity": f"{float(r['similarity']):.6f}",
#             })

#     print(f"已将每对音频的相似度写入：{OUTPUT_CSV_PATH}")
#     print("脚本运行结束。")


# if __name__ == "__main__":
#     main()















# 用Resemblyzer计算说话人相似度(多个音频)（通过目标音频路径来替换得到生成音频路径）
"""
批量计算说话人相似度脚本（对齐代码段一的路径构造方式）

功能：
1) 读取 metadata_test.csv（至少包含 audio_file 列；speaker_name/text 可选）
2) 构造 GT 路径：
   - audio_file 为绝对路径：直接用
   - audio_file 为相对路径：BASE_GT_PREFIX + audio_file
3) 构造 GEN 路径：
   在 GT 路径中将 "/{ORIG_SUBDIR}/" 替换为 "/{GEN_SUBDIR}/"（只替换第一次）
4) 使用 Resemblyzer 提取 embedding 并计算余弦相似度（点积）
5) 写出 spk_sim.csv，并写出总体平均相似度 spk_sim_sum.txt
"""

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import csv
import os


# ==========================
# 参数区（按你实际情况修改）
# ==========================
# 读取的 metadata_test.csv（与代码段一一致）
CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
CSV_DELIM = "|"   # metadata 用 "|" 分隔

# 当 CSV 中 audio_file 是相对路径时，用它作为前缀
BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# 路径替换规则（与代码段一一致）
ORIG_SUBDIR = "wavs"

# 这里改成你要评估的生成目录
# GEN_SUBDIR = "test/generate_from_pretrain_embedding"  
# GEN_SUBDIR = "test/generate_from_test_embedding"  
# GEN_SUBDIR = "test/generate_from_train_embedding"  
# GEN_SUBDIR = "test/no_ecl_emo_labels_gpt-last-layer"  
GEN_SUBDIR = "test/no_pre-ft"  

# 输出路径（建议放到你实验目录下）
# OUTPUT_DIR = "/mnt/hd/zjy/test/generate_from_pretrain_embedding"   
# OUTPUT_DIR = "/mnt/hd/zjy/test/generate_from_test_embedding"  
# OUTPUT_DIR = "/mnt/hd/zjy/test/generate_from_train_embedding"   
OUTPUT_DIR = "/mnt/hd/zjy/test/no_pre-ft"   

OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "spk_sim.csv")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "spk_sim_sum.txt")


def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str):
    """
    复刻代码段一的构造方式：
    - GT: 来自 audio_file（绝对路径直接用；相对路径 base_gt_prefix 拼接）
    - GEN: 将 GT 中 "/ORIG_SUBDIR/" 替换为 "/GEN_SUBDIR/"（只替换第一次）
    返回：gt_paths, gen_paths, rows(原始行dict列表，用于取 speaker_name 等可选列)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    gt_paths, gen_paths, rows = [], [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=csv_delim)

        if not reader.fieldnames or "audio_file" not in reader.fieldnames:
            raise ValueError(f"CSV 必须包含 audio_file 列，当前列名: {reader.fieldnames}")

        for row in reader:
            rows.append(row)
            af = (row.get("audio_file") or "").strip()
            if not af:
                raise ValueError("发现空的 audio_file，请检查 CSV")

            # 1) 构造 GT 路径
            if os.path.isabs(af):
                gt = af
            else:
                gt = os.path.join(base_gt_prefix, af)
            gt = os.path.normpath(gt)
            gt_paths.append(gt)

            # 2) 由 GT 构造 GEN 路径（按段一逻辑）
            pattern = f"/{ORIG_SUBDIR}/"
            replacement = f"/{GEN_SUBDIR}/"
            if pattern not in gt:
                raise ValueError(
                    f"GT path 不包含 '{pattern}': {gt}，无法按规则替换生成音频路径。"
                    f"请检查 ORIG_SUBDIR 或 CSV 内容。"
                )
            gen = gt.replace(pattern, replacement, 1)
            gen = os.path.normpath(gen)
            gen_paths.append(gen)

    return gt_paths, gen_paths, rows


def main():
    # 1) 初始化声纹编码器
    print("初始化 VoiceEncoder（第一次运行会自动下载模型，可能稍慢）...")
    encoder = VoiceEncoder()
    print("VoiceEncoder 初始化完成。")

    # 2) 从 metadata_test.csv 构造 gt/gen 路径
    print(f"读取 CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
    gt_paths, gen_paths, rows = build_paths_from_csv(CSV_PATH, CSV_DELIM, BASE_GT_PREFIX)

    # 打印前 5 条映射确认
    print("Example mapping (first 5):")
    for i in range(min(5, len(gt_paths))):
        print(f"  GT : {gt_paths[i]}")
        print(f"  GEN: {gen_paths[i]}")

    similarities = []
    output_rows = []

    # 3) 逐对计算相似度
    for idx, (gt_path, gen_path, row) in enumerate(zip(gt_paths, gen_paths, rows), start=1):
        speaker_name = (row.get("speaker_name") or "").strip()  # 可选列，没有就为空

        print(f"\n[{idx}] 处理样本：")
        print(f"  生成音频: {gen_path}")
        print(f"  目标音频: {gt_path}")
        if speaker_name:
            print(f"  说话人  : {speaker_name}")

        if not os.path.exists(gen_path):
            print("  [WARN] 找不到生成音频文件，跳过该样本。")
            continue
        if not os.path.exists(gt_path):
            print("  [WARN] 找不到目标音频文件，跳过该样本。")
            continue

        try:
            wav_gt = preprocess_wav(Path(gt_path))
            wav_gen = preprocess_wav(Path(gen_path))

            emb_gt = encoder.embed_utterance(wav_gt)
            emb_gen = encoder.embed_utterance(wav_gen)

            similarity = float(np.dot(emb_gt, emb_gen))
            print(f"  相似度 (cosine similarity) = {similarity:.6f}")

            similarities.append(similarity)
            output_rows.append({
                "gen_path": gen_path,
                "gt_path": gt_path,
                "speaker_name": speaker_name,
                "similarity": similarity,
            })

        except Exception as e:
            print(f"  [ERROR] 处理该样本时出错，已跳过。错误信息: {e}")
            continue

    # 4) 汇总平均
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    if len(similarities) == 0:
        avg_similarity = float("nan")
        summary_text = (
            "===========================================\n"
            "未成功计算任何一对音频的说话人相似度，请检查输入数据。\n"
            "===========================================\n"
        )
    else:
        avg_similarity = float(np.mean(similarities))
        summary_text = (
            "===========================================\n"
            f"共成功计算 {len(similarities)} 对音频的说话人相似度。\n"
            f"平均说话人相似度: {avg_similarity:.6f}\n"
            "===========================================\n"
        )

    print(summary_text)
    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f_sum:
        f_sum.write(summary_text)
    print(f"已将总体结果写入：{SUMMARY_TXT_PATH}\n")

    # 5) 写出 CSV
    fieldnames = ["gen_path", "gt_path", "speaker_name", "similarity"]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)

    print(f"已将每对音频的相似度写入：{OUTPUT_CSV_PATH}")
    print("脚本运行结束。")


if __name__ == "__main__":
    main()








# 用Resemblyzer计算说话人相似度(多个音频)（通过生成音频路径来替换得到目标音频路径）
"""
批量计算说话人相似度脚本

功能：
1. 读取输入 CSV（列：audio_file|text|speaker_name）。
2. 对每一行，根据 audio_file 构造对应的 target_audios 路径。
3. 使用 Resemblyzer 提取两段音频的说话人 embedding，计算余弦相似度。
4. 将每对音频的相似度写入输出 CSV：spk_emo.csv。
5. 在控制台打印每一对的相似度，以及所有有效样本的平均相似度。
"""

# from resemblyzer import VoiceEncoder, preprocess_wav
# from pathlib import Path
# import numpy as np
# import csv
# import os


# # ==========================
# # 可根据需要修改的参数（在代码中写死）
# # ==========================
# # 输入的 CSV 文件路径（请修改为你自己的）
# INPUT_CSV_PATH = "/mnt/hd/zjy/test/no_emo_labels_en/generate.csv"

# # 输出的 CSV 文件路径（会自动创建/覆盖）
# OUTPUT_CSV_PATH = "/mnt/hd/zjy/test/no_emo_labels_en/spk_sim.csv"

# # 保存最终计算的平均相似度
# summary_text_file = "/mnt/hd/zjy/test/no_emo_labels_en/spk_sim_sum.txt"

# # 实验生成音频所在目录名（出现在 audio_file 路径中）
# EXPERIMENT_DIR_NAME = "no_emo_labels_en"

# # 目标音频所在目录名（用来替换 EXPERIMENT_DIR_NAME）
# TARGET_DIR_NAME = "target_audios"



# def build_target_path(generated_audio_path: str) -> str:
#     """
#     根据生成音频的路径构造目标音频路径。

#     例如：
#     /mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Angry/0020_000299.wav
#     -> /mnt/hd/zjy/test/target_audios/ESD/0020/Angry/0020_000299.wav

#     这里简单地把路径字符串中第一次出现的 EXPERIMENT_DIR_NAME
#     替换为 TARGET_DIR_NAME。
#     """
#     return generated_audio_path.replace(EXPERIMENT_DIR_NAME, TARGET_DIR_NAME, 1)


# def main():
#     # ==========================
#     # 1. 初始化声纹编码器
#     # ==========================
#     print("初始化 VoiceEncoder（第一次运行会自动下载模型，可能稍慢）...")
#     encoder = VoiceEncoder()
#     print("VoiceEncoder 初始化完成。")

#     # 用来保存成功计算相似度的样本
#     similarities = []       # 只存数值，用来最后取平均
#     output_rows = []        # 存每一行要写入 spk_emo.csv 的数据

#     # ==========================
#     # 2. 读取输入 CSV
#     # ==========================
#     if not os.path.exists(INPUT_CSV_PATH):
#         raise FileNotFoundError(f"找不到输入 CSV 文件：{INPUT_CSV_PATH}")

#     with open(INPUT_CSV_PATH, "r", encoding="utf-8") as f:
#         # 注意：你的 CSV 用的是 '|' 作为分隔符
#         reader = csv.DictReader(f, delimiter="|")

#         # 简单检查列名是否完整
#         required_cols = {"audio_file", "text", "speaker_name"}
#         missing = required_cols - set(reader.fieldnames or [])
#         if missing:
#             raise ValueError(f"CSV 缺少必要列: {missing}，当前列名: {reader.fieldnames}")

#         # ==========================
#         # 3. 逐行处理
#         # ==========================
#         for idx, row in enumerate(reader, start=1):
#             gen_path = row["audio_file"]           # 生成音频
#             speaker_name = row["speaker_name"]     # 说话人 id / 名字
#             tgt_path = build_target_path(gen_path) # 目标音频

#             # 打印当前处理到哪一行，方便监控进度
#             print(f"\n[{idx}] 处理样本：")
#             print(f"  生成音频: {gen_path}")
#             print(f"  目标音频: {tgt_path}")
#             print(f"  说话人  : {speaker_name}")

#             # 先检查这两个路径是否都存在，不存在就跳过
#             if not os.path.exists(gen_path):
#                 print(f"  [WARN] 找不到生成音频文件，跳过该样本。")
#                 continue
#             if not os.path.exists(tgt_path):
#                 print(f"  [WARN] 找不到目标音频文件，跳过该样本。")
#                 continue

#             try:
#                 # ==========================
#                 # 4. 预处理音频并提取 embedding
#                 # ==========================
#                 # preprocess_wav 会完成读文件、重采样、归一化等操作
#                 wav_target = preprocess_wav(Path(tgt_path))
#                 wav_generated = preprocess_wav(Path(gen_path))

#                 # 对每段语音提取 256 维说话人 embedding
#                 # Resemblyzer 内部会对 embedding 做 L2 归一化，
#                 # 因此向量点积等价于余弦相似度。
#                 emb_target = encoder.embed_utterance(wav_target)
#                 emb_generated = encoder.embed_utterance(wav_generated)

#                 # ==========================
#                 # 5. 计算说话人相似度
#                 # ==========================
#                 similarity = float(np.dot(emb_target, emb_generated))

#                 # 打印当前对的相似度
#                 print(f"  相似度 (cosine similarity) = {similarity:.6f}")

#                 # 保存结果用于平均计算 & 输出 CSV
#                 similarities.append(similarity)
#                 output_rows.append(
#                     {
#                         "audio_file": gen_path,
#                         "target_audio_file": tgt_path,
#                         "speaker_name": speaker_name,
#                         "similarity": similarity,
#                     }
#                 )

#             except Exception as e:
#                 # 任意异常都记录下来并继续下一个样本
#                 print(f"  [ERROR] 处理该样本时出错，已跳过。错误信息: {e}")
#                 continue

#     # ==========================
#     # 6. 计算整体平均相似度
#     # ==========================
#     if len(similarities) == 0:
#         avg_similarity = float("nan")
#         summary_text = (
#             "===========================================\n"
#             "未成功计算任何一对音频的说话人相似度，请检查输入数据。\n"
#             "===========================================\n"
#         )
#         print(summary_text)
#     else:
#         avg_similarity = float(np.mean(similarities))

#         summary_text = (
#             "===========================================\n"
#             f"共成功计算 {len(similarities)} 对音频的说话人相似度。\n"
#             f"平均说话人相似度: {avg_similarity:.6f}\n"
#             "===========================================\n"
#         )

#         # 打印到控制台
#         print(summary_text)

#     # 将最终结果写入 spk_sim_sum.txt 文件
#     with open(summary_text_file, "w", encoding="utf-8") as f_sum:
#         f_sum.write(summary_text)

#     print(f"已将总体结果写入：{summary_text_file}\n")

#     # ==========================
#     # 7. 将结果写入输出 CSV
#     # ==========================
#     # 输出文件包含以下列：
#     #   - audio_file          : 生成音频路径
#     #   - target_audio_file   : 对应的目标音频路径
#     #   - speaker_name        : 说话人 ID / 名字
#     #   - similarity          : 说话人相似度（cosine similarity）
#     fieldnames = ["audio_file", "target_audio_file", "speaker_name", "similarity"]

#     # 创建输出目录（如果不存在）
#     out_dir = os.path.dirname(OUTPUT_CSV_PATH)
#     if out_dir and not os.path.exists(out_dir):
#         os.makedirs(out_dir, exist_ok=True)

#     with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
#         writer = csv.DictWriter(f_out, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in output_rows:
#             writer.writerow(r)

#     print(f"已将每对音频的相似度写入：{OUTPUT_CSV_PATH}")
#     print("脚本运行结束。")


# if __name__ == "__main__":
#     main()







# 用Resemblyzer计算说话人相似度(单个音频)
# from resemblyzer import VoiceEncoder, preprocess_wav
# from pathlib import Path
# import numpy as np

# # 1. 加载并预处理语音
# fpath1 = Path("/mnt/hd/zjy/test/target_audios/ESD/0020/Angry/0020_000299.wav")
# fpath2 = Path("/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Angry/0020_000299.wav")
# wav1 = preprocess_wav(fpath1)
# wav2 = preprocess_wav(fpath2)

# # 2. 构建 encoder（第一次调用会自动下载预训练模型）
# encoder = VoiceEncoder()

# # 3. 提取每段语音的 embedding（每个是 256 维向量）
# embed1 = encoder.embed_utterance(wav1)
# embed2 = encoder.embed_utterance(wav2)

# # 4. 计算相似度（Resemblyzer 的 embedding 默认做了 L2 归一化，
# #    所以向量点积 = 余弦相似度）
# similarity = np.dot(embed1, embed2)

# print("Similarity:", similarity)






# 批量计算平均说话人相似性（并对计算结果进行保存）
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_speaker_similarity_from_csv_with_csv_output.py

功能概述（基于你当前的需求）：
  - 从 CSV (audio_file|text|speaker_name) 读取目标音频路径（GT）：
      * 若 CSV 中 audio_file 为绝对路径（以 / 开头），则直接使用；
      * 若为相对路径，则用 BASE_GT_PREFIX + audio_file 拼接得到绝对路径。
  - 对应生成音频路径（GEN）的构造规则：
      * 假设 GT 为：/mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
      * 则 GEN 为：/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
      * 即：在 GT 路径中找到 "/wavs/"，保留其后面的相对子路径 "ESD/0020/Sad/xxx.wav"，
        然后挂到 GENERATE_PATH（/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en）后面。
  - 对每对 (GEN, GT) 单独调用 model.get_conditioning_latents(...) 提取说话人 embedding（**不做缓存**）。
  - 计算并打印每对余弦相似度（保留 4 位小数）。
  - 汇总并打印 mean/median/std（忽略 NaN），并按 speaker 统计平均相似度。
  - 将 Summary 文本保存到： os.path.join(GENERATE_PATH, "spk_sim_summary.txt")
  - 将每对 (gen_path, gt_path, similarity) 写入 CSV： os.path.join(GENERATE_PATH, "spk_sim_result.csv")
    CSV 使用逗号分隔，similarity 写为 4 位小数（字符串格式）。
  - 若任意一条在检查时路径不存在或模型调用失败，则**立即抛出异常**以便立刻排查。
"""

# import os
# import sys
# import traceback
# from typing import List, Tuple, Dict, Any, Optional

# import csv
# import numpy as np
# import torch
# from torch.nn import functional as F
# import pandas as pd

# # -------------------- 配置区域（请按需修改） --------------------
# CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"   # <- 你的 CSV 文件路径（包含 audio_file|text|speaker_name）
# CSV_DELIM = "|"                              # CSV 分隔符（示例用 '|'）

# # 如果 CSV 中 audio_file 为相对路径，可用这个前缀拼出绝对路径；
# # 如果 CSV 中已经是绝对路径（例如 /mnt/hd/zjy/wavs/...），就不会用到这个前缀。
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset"

# # 生成音频的“根目录”：
# # 比如目标路径 /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav，
# # 则生成路径是 /mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
# # GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en"
# # GENERATE_PATH = "/mnt/hd/zjy/test/no_emo_labels_en"
# GENERATE_PATH = "/mnt/hd/zjy/test/5_emo_labels_en"
# # GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_295020"
# # GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_375480"

# # XTTS 模型文件（checkpoint/config/vocab）   
# # 由于微调XTTS过程中，不会训练说话人编码器，所以使用预训练模型来计算说话人嵌入
# # 计算说话人嵌入时用不到情感标签，所以用XTTS预训练的config.json和vocab.json即可
# MODEL_CHECKPOINT = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/model.pth"
# MODEL_CONFIG = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/config_original.json"
# MODEL_VOCAB = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/vocab_original.json"

# # 输出开关：是否打印逐对信息（True/False）
# PRINT_PROGRESS = True

# # 输出文件（Summary 文本 & CSV），基于 GENERATE_PATH
# RESULTS_TXT_PATH = os.path.join(GENERATE_PATH, "spk_sim_summary.txt")
# RESULTS_CSV_PATH = os.path.join(GENERATE_PATH, "spk_sim_result.csv")
# # -------------------------------------------------------------

# # ---------- 导入 XTTS（与合成时相同环境） ----------
# try:
#     from TTS.tts.configs.xtts_config import XttsConfig
#     from TTS.tts.models.xtts import Xtts
# except Exception as e:
#     print("Error: 无法 import XTTS 模块。请确认你的 Python 环境中能导入 TTS 包（与合成时一致）。")
#     raise

# # 全局模型句柄（加载后赋值）
# XTTS_MODEL = None


# def clear_gpu_cache():
#     """如果有 GPU 则释放缓存，降低 OOM 风险。"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def load_xtts_model(checkpoint: str, config_path: str, vocab_path: str):
#     """
#     加载 XTTS 模型并将其实例保存到全局变量 XTTS_MODEL。
#     - checkpoint/config/vocab 必须提供且路径正确。
#     - 返回加载的 model 对象（便于测试/外部调用）。
#     """
#     global XTTS_MODEL
#     clear_gpu_cache()
#     if not checkpoint or not config_path or not vocab_path:
#         raise ValueError("请提供 checkpoint/config/vocab 的路径。")

#     cfg = XttsConfig()
#     cfg.load_json(config_path)
#     model = Xtts.init_from_config(cfg)

#     print(f"Loading XTTS checkpoint: {checkpoint}")
#     # 调用模型的 checkpoint 加载接口（与合成脚本保持一致）
#     model.load_checkpoint(
#         cfg,
#         checkpoint_path=checkpoint,
#         vocab_path=vocab_path,
#         strict=False,
#         use_deepspeed=False,
#     )

#     # 若有 CUDA 可用，将模型移到 GPU 上（可加速 embedding 提取）
#     if torch.cuda.is_available():
#         model.cuda()

#     XTTS_MODEL = model
#     print("XTTS model loaded.")
#     return model


# def _ensure_model_loaded():
#     """检查模型是否已经 load。"""
#     if XTTS_MODEL is None:
#         raise RuntimeError("XTTS_MODEL 未加载，请先调用 load_xtts_model(...)。")


# # ---------- path 构造（已改为“wavs -> test/... ”的规则） ----------
# def build_paths_from_csv(
#     csv_path: str,
#     csv_delim: str,
#     base_gt_prefix: str,
#     generate_prefix: str,
# ) -> Tuple[List[str], List[str], List[str]]:
#     """
#     从 CSV 读取列 audio_file 与 speaker_name（text 可忽略），构造：
#       - gt_paths: 目标音频路径（GT）
#           * 如果 CSV 中 audio_file 是绝对路径（以 / 开头），则直接使用该路径；
#           * 如果是相对路径，则用 base_gt_prefix + audio_file 拼出完整路径。
#       - gen_paths: 生成音频路径（GEN）
#           * 在 GT 路径中，找到路径片段 "/wavs/"，并将其后的子路径挂载到 generate_prefix 下：
#                 GT:  /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
#                 GEN: /mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
#           * 换句话说：保持 "ESD/0020/Sad/xxx.wav" 这部分目录结构不变，只替换根目录。
#       - speakers: 从 speaker_name 列读取（如果缺失则填空字符串）

#     注意：
#       - CSV 必须包含 audio_file 列；如果列名不同请先修改或预处理 CSV。
#       - 若在 GT 路径中找不到 "/wavs/" 这一段，将抛出 ValueError，
#         以便你尽早发现路径格式不符合预期。
#     """
#     # 读入 CSV（保持与原脚本一致）
#     df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)
#     if "audio_file" not in df.columns:
#         raise ValueError("CSV must contain column: audio_file (格式: audio_file|text|speaker_name)。")

#     # speaker_name 可选；若不存在则填空字符串
#     speakers = (
#         df["speaker_name"].astype(str).tolist()
#         if "speaker_name" in df.columns
#         else [""] * len(df)
#     )

#     audio_files = df["audio_file"].astype(str).tolist()
#     gt_paths: List[str] = []
#     gen_paths: List[str] = []

#     # 要替换的路径片段关键字；这里专门替换路径中的 "wavs" 这一段
#     wavs_token = "wavs"

#     for af in audio_files:
#         af = af.strip()
#         if not af:
#             raise ValueError("CSV 中存在空的 audio_file 字段，请检查。")

#         # ---------- 1) 构造 GT 路径 ----------
#         # 若为绝对路径（以 / 开头），直接作为 GT 使用
#         if os.path.isabs(af):
#             gt = af
#         else:
#             # 否则用 base_gt_prefix 作为前缀，拼接出完整路径
#             gt = os.path.join(base_gt_prefix, af)

#         # 归一化路径（去掉多余的 .. 或重复的 /）
#         gt = os.path.normpath(gt)
#         gt_paths.append(gt)

#         # ---------- 2) 从 GT 路径推导 GEN 路径 ----------
#         #
#         # 目标：将 GT 中的
#         #   .../wavs/ESD/0020/Sad/0020_000332.wav
#         # 映射为
#         #   generate_prefix + "/ESD/0020/Sad/0020_000332.wav"
#         #
#         # 做法：
#         #   1) 在 GT 字符串中找到子串 "/wavs/"（或 Windows 下的 "\wavs\"）
#         #   2) 取其后面的相对子路径，例如 "ESD/0020/Sad/0020_000332.wav"
#         #   3) 用 generate_prefix + 相对子路径 拼出 GEN 路径
#         #
#         # 这样可以保证 ESD/0020/Sad/... 这层目录结构完整保留。
#         token_with_sep = os.sep + wavs_token + os.sep  # 例如 "/wavs/" 或 "\wavs\"
#         pos = gt.find(token_with_sep)

#         if pos == -1:
#             # 没找到 "/wavs/"，说明路径格式不符合预期；直接抛错，方便你排查
#             raise ValueError(
#                 f"在目标音频路径中未找到 '{token_with_sep}' 片段，"
#                 f"无法根据约定规则构造生成音频路径：{gt}"
#             )

#         # 取出 "wavs" 之后那一段相对路径，比如 "ESD/0020/Sad/0020_000332.wav"
#         rel_subpath = gt[pos + len(token_with_sep):]

#         # 生成路径 = generate_prefix + 相对子路径
#         # generate_prefix 就是 "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en"
#         gen = os.path.join(generate_prefix, rel_subpath)
#         gen = os.path.normpath(gen)
#         gen_paths.append(gen)

#     return gt_paths, gen_paths, speakers


# # ---------- embedding 提取与余弦相似度计算 ----------
# def extract_speaker_embedding_no_cache(audio_path: str, model) -> np.ndarray:
#     """
#     对单个音频路径调用 model.get_conditioning_latents(...)
#     并返回 1-D numpy 数组作为说话人 embedding（不缓存）。

#     - 若 audio_path 不存在，将抛出 FileNotFoundError。
#     - 若 model.get_conditioning_latents 出现错误，会抛出异常（上层决定是否捕获或终止）。
#     - 返回值保证为 1D numpy float 数组。
#     """
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio not found: {audio_path}")

#     # 这些参数名基于 XTTS 模型的 config；保持与你训练/合成时一致
#     gpt_cond, speaker_embedding = model.get_conditioning_latents(
#         audio_path=audio_path,
#         gpt_cond_len=getattr(model.config, "gpt_cond_len"),
#         max_ref_length=getattr(model.config, "max_ref_len"),
#         sound_norm_refs=getattr(model.config, "sound_norm_refs"),
#     )

#     # 兼容返回 torch.Tensor 或 numpy 可迭代对象
#     if isinstance(speaker_embedding, torch.Tensor):
#         emb = speaker_embedding.detach().cpu().numpy()
#     else:
#         emb = np.array(speaker_embedding, dtype=np.float32)
#     return emb.reshape(-1)


# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     使用 PyTorch 对 numpy 向量计算余弦相似度（先归一化再点乘）。
#     返回 float（-1.0 到 1.0）或在输入为零向量时返回 NaN。
#     """
#     a = a.reshape(-1)
#     b = b.reshape(-1)
#     # 安全检查零向量
#     if a.size == 0 or b.size == 0:
#         return float("nan")
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)


# # ---------- 批量计算（遇错立即抛出，以便你能立刻排查） ----------
# def batch_speaker_similarity(
#     model,
#     gen_paths: List[str],
#     gt_paths: List[str],
#     speakers: List[str],
# ) -> List[Dict[str, Any]]:
#     """
#     对每对 (gen, gt, speaker) 执行：
#       - 路径存在性检查（若不存在立即抛出 FileNotFoundError）
#       - 提取 embedding（若模型调用异常会抛出）
#       - 计算余弦相似度并把结果追加到列表
#     返回每项为字典：{'gen_path','gt_path','speaker','similarity'}
#     """
#     if len(gen_paths) != len(gt_paths) or len(gen_paths) != len(speakers):
#         raise ValueError("gen_paths, gt_paths, speakers must have the same length.")

#     results: List[Dict[str, Any]] = []
#     total = len(gen_paths)
#     for i, (gpath, tpath, spk) in enumerate(zip(gen_paths, gt_paths, speakers), start=1):
#         # 路径检查：若不存在，立即抛出（按你要求）
#         if not os.path.exists(gpath):
#             raise FileNotFoundError(f"Generated audio not found: {gpath}")
#         if not os.path.exists(tpath):
#             raise FileNotFoundError(f"Target audio not found: {tpath}")

#         if PRINT_PROGRESS:
#             print(f"[{i}/{total}] gen: {gpath}\n    gt:  {tpath}\n    speaker: {spk}")

#         # 提取 embedding（无缓存）
#         gen_emb = extract_speaker_embedding_no_cache(gpath, model)
#         gt_emb = extract_speaker_embedding_no_cache(tpath, model)

#         # 计算余弦相似度
#         sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#         results.append(
#             {
#                 "gen_path": gpath,
#                 "gt_path": tpath,
#                 "speaker": spk,
#                 "similarity": float(sim),
#             }
#         )

#         if PRINT_PROGRESS:
#             # 打印 4 位小数（NaN 单独打印）
#             if np.isnan(sim):
#                 print(f"    -> similarity = nan")
#             else:
#                 print(f"    -> similarity = {sim:.4f}")

#     return results


# # ---------- 汇总、打印并保存 Summary 到文件（格式化为 4 位小数） ----------
# def summarize_results_and_save(
#     results: List[Dict[str, Any]],
#     out_txt_path: str,
#     out_csv_path: str,
# ):
#     """
#     - 计算整体 mean/median/std（忽略 NaN）
#     - 计算 per-speaker 平均相似度与样本数
#     - 在控制台打印 Summary（保留 4 位小数）
#     - 把 Summary 文本写入 out_txt_path（覆盖写）
#     - 把每对 (gen_path, gt_path, similarity) 写入 out_csv_path（CSV, 逗号分隔，similarity 保留 4 位小数）
#     """
#     sims = np.array([r["similarity"] for r in results], dtype=float)
#     valid = sims[~np.isnan(sims)]
#     mean_sim = float(valid.mean()) if valid.size > 0 else None
#     median_sim = float(np.median(valid)) if valid.size > 0 else None
#     std_sim = float(valid.std(ddof=0)) if valid.size > 0 else None

#     # per-speaker 统计
#     per_spk: Dict[str, Dict[str, Any]] = {}
#     for r in results:
#         spk = (r.get("speaker") or "").strip()
#         per_spk.setdefault(spk, {"count": 0, "sum_sim": 0.0})
#         per_spk[spk]["count"] += 1
#         per_spk[spk]["sum_sim"] += float(r["similarity"])

#     per_spk_stats: Dict[str, Dict[str, Any]] = {}
#     for spk, v in per_spk.items():
#         cnt = v["count"]
#         mean_k = (v["sum_sim"] / cnt) if cnt > 0 else None
#         per_spk_stats[spk] = {"total": cnt, "mean_similarity": mean_k}

#     # 构造 Summary 文本（每行字符串，便于打印与写入）
#     lines: List[str] = []
#     lines.append("=== Summary ===")
#     lines.append(f"pairs total: {len(results)}")
#     if mean_sim is not None:
#         lines.append(f"mean similarity: {mean_sim:.4f}")
#         lines.append(f"median similarity: {median_sim:.4f}")
#         lines.append(f"std similarity: {std_sim:.4f}")
#     else:
#         lines.append("No valid similarity values.")

#     lines.append("per-speaker stats (speaker: total, mean_similarity):")
#     # 为了可预测输出顺序，按 speaker key 排序输出
#     for spk in sorted(per_spk_stats.keys()):
#         stats = per_spk_stats[spk]
#         mean_k_str = (
#             f"{stats['mean_similarity']:.4f}"
#             if stats["mean_similarity"] is not None
#             else "None"
#         )
#         lines.append(f"  {spk}: {stats['total']}, mean_similarity={mean_k_str}")

#     # 打印 Summary 到控制台
#     for ln in lines:
#         print(ln)

#     # 将 Summary 写入文本文件（覆盖写），并确保目录存在
#     try:
#         dirpath = os.path.dirname(out_txt_path)
#         if dirpath:
#             os.makedirs(dirpath, exist_ok=True)
#         with open(out_txt_path, "w", encoding="utf-8") as f:
#             for ln in lines:
#                 f.write(ln + "\n")
#         print(f"Saved summary text to: {out_txt_path}")
#     except Exception as e:
#         # 保存失败不要掩盖主逻辑：打印错误但不抛出（以便你仍可获得控制台输出）
#         print(f"Warning: failed to save summary text to {out_txt_path}: {e}")

#     # 将每对结果写入 CSV（覆盖写），格式： gen_path,gt_path,similarity (similarity 保留 4 位小数)
#     try:
#         dir_csv = os.path.dirname(out_csv_path)
#         if dir_csv:
#             os.makedirs(dir_csv, exist_ok=True)
#         with open(out_csv_path, "w", newline="", encoding="utf-8") as csvf:
#             writer = csv.writer(csvf)
#             # 写表头（3 列）
#             writer.writerow(["gen_path", "gt_path", "similarity"])
#             # 写每一行，similarity 格式化为 4 位小数；若为 nan 则写为空字符串
#             for r in results:
#                 sim_val = r.get("similarity")
#                 if sim_val is None or np.isnan(sim_val):
#                     sim_str = ""
#                 else:
#                     sim_str = f"{sim_val:.4f}"
#                 writer.writerow([r.get("gen_path"), r.get("gt_path"), sim_str])
#         print(f"Saved per-pair CSV to: {out_csv_path}")
#     except Exception as e:
#         print(f"Warning: failed to save CSV to {out_csv_path}: {e}")

#     return {
#         "mean_similarity": mean_sim,
#         "median_similarity": median_sim,
#         "std_similarity": std_sim,
#         "per_speaker": per_spk_stats,
#     }


# # ---------- main ----------
# def main():
#     # 检查 CSV 是否存在
#     if not os.path.exists(CSV_PATH):
#         print(f"Error: CSV not found: {CSV_PATH}")
#         sys.exit(2)

#     try:
#         # 1) 加载 XTTS 模型
#         print("Loading XTTS model...")
#         model = load_xtts_model(MODEL_CHECKPOINT, MODEL_CONFIG, MODEL_VOCAB)
#         print("Model loaded.")

#         # 2) 从 CSV 构造路径（gt_paths, gen_paths, speakers）
#         print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#         gt_paths, gen_paths, speakers = build_paths_from_csv(
#             CSV_PATH,
#             CSV_DELIM,
#             BASE_GT_PREFIX,
#             GENERATE_PATH,
#         )

#         # debug：打印前 5 条映射以便人工核查
#         print("Example mapping (first 5):")
#         for i in range(min(5, len(gt_paths))):
#             print(f"  GT:  {gt_paths[i]}")
#             print(f"  GEN: {gen_paths[i]}")
#             print(f"  SPK: {speakers[i]}")

#         # 3) 逐对计算相似度（遇错即时抛出）
#         results = batch_speaker_similarity(model, gen_paths, gt_paths, speakers)

#         # 4) 汇总、打印、保存 Summary，并把每对写入 CSV
#         summary = summarize_results_and_save(
#             results,
#             RESULTS_TXT_PATH,
#             RESULTS_CSV_PATH,
#         )

#         # 返回 summary 以便外部调用或单元测试使用
#         return summary

#     except Exception:
#         # 打印完整 trace 以便排查问题（模型/路径/权限等）
#         traceback.print_exc()
#         raise


# if __name__ == "__main__":
#     main()








# 批量计算说话人相似性
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# compute_cosine_wayB.py

# 用途（方式 B）：
#   - 直接在脚本顶部用 Python 列表填入两等长的路径列表：
#       GEN_PATHS = ["/path/to/gen1.wav", "/path/to/gen2.wav", ...]
#       REF_PATHS = ["/path/to/ref1.wav", "/path/to/ref2.wav", ...]
#     其中第 i 个生成音对应第 i 个目标音。
#   - 脚本对每一对 (gen, ref) 都**单独**调用 XTTS_MODEL.get_conditioning_latents(...) 提取说话人嵌入（不做缓存）。
#   - 只计算余弦相似度（cosine similarity），在控制台打印每对 cosine 和最终的 mean/std/median。
#   - 不写 CSV，不计算欧式距离（按你要求去掉）。

# 使用：
#   - 编辑下面的 GEN_PATHS / REF_PATHS 列表（确保等长且路径存在）。
#   - 修改 MODEL_* 三个路径为你的 checkpoint/config/vocab（或保持默认示例）。
#   - 运行： python compute_cosine_wayB.py
# """

# import os
# import sys
# import traceback
# from typing import List, Sequence, Tuple, Optional

# import numpy as np
# import torch

# # -------------------- 配置区域：在此处填入你的生成音频列表与参考音频列表 --------------------
# # 注意：两者必须等长，且顺序要一一对应（第 i 个生成音对应第 i 个参考音）。
# # 英文模型  w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10_en      no_ecl-loss_en
# # XTTS_v2.0_original_model_files_original_en
# GEN_PATHS: Optional[List[str]] = [
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/neutral.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/angry.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/happy.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/sad.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/surprise.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/neutral.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/angry.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/happy.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/sad.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/surprise.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/neutral.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/angry.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/happy.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/sad.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/surprise.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/neutral.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/angry.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/happy.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/sad.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/surprise.wav",
# ]
# REF_PATHS: Optional[List[str]] = [
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_000349.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_000699.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001049.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001399.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001749.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000133.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000483.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000833.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_001183.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_001533.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_000306.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_000656.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001006.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001356.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001706.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000071.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000421.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000771.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_001121.wav",
#     "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_001471.wav",
# ]

# # 中文模型  w_ecl_0.5_gpt-last-layer_lr_2e_6_1-10      no_ecl-loss
# # XTTS_v2.0_original_model_files_original
# # GEN_PATHS: Optional[List[str]] = [
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0001_000159/neutral.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0001_000159/angry.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0001_000159/happy.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0001_000159/sad.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0001_000159/surprise.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0004_000047/neutral.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0004_000047/angry.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0004_000047/happy.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0004_000047/sad.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0004_000047/surprise.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0006_000009/neutral.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0006_000009/angry.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0006_000009/happy.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0006_000009/sad.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0006_000009/surprise.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0009_000002/neutral.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0009_000002/angry.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0009_000002/happy.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0009_000002/sad.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original/0009_000002/surprise.wav",
# # ]
# # REF_PATHS: Optional[List[str]] = [
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000159.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000509.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000859.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_001209.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_001559.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000047.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000397.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000747.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_001097.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_001447.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000009.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000359.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000709.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_001059.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_001409.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000002.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000352.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000702.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_001052.wav",
# #     "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_001402.wav",
# # ]

# # XTTS 模型的 checkpoint / config / vocab 路径（按需修改）
# MODEL_CHECKPOINT = "/tmp/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/model.pth"
# MODEL_CONFIG = "/tmp/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/config_original.json"
# MODEL_VOCAB = "/tmp/xtts_ft/run/training/XTTS_v2.0_original_model_files_original/vocab_original.json"
# # ----------------------------------------------------------------------------------------

# # ---------- 以下为实现：无需修改（除非你熟悉脚本并要扩展） ----------
# # 导入你的 XTTS 实现（应与合成时相同的环境）
# try:
#     from TTS.tts.configs.xtts_config import XttsConfig
#     from TTS.tts.models.xtts import Xtts
# except Exception as e:
#     print("Error: 无法 import XTTS 模块。请确认你的 Python 环境中能导入 TTS 包（与合成时一致）。")
#     raise

# XTTS_MODEL = None

# def clear_gpu_cache():
#     """如有 CUDA 则清理缓存，降低 OOM 风险"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# def load_xtts_model(checkpoint: str, config_path: str, vocab_path: str):
#     """
#     加载 XTTS 模型（和你合成时用的是同一套 API）。
#     把模型保存到全局 XTTS_MODEL，并在可用时移到 GPU。
#     """
#     global XTTS_MODEL
#     clear_gpu_cache()
#     if not checkpoint or not config_path or not vocab_path:
#         raise ValueError("请提供 checkpoint/config/vocab 的路径。")

#     cfg = XttsConfig()
#     cfg.load_json(config_path)
#     model = Xtts.init_from_config(cfg)

#     print(f"Loading XTTS checkpoint: {checkpoint}")
#     model.load_checkpoint(cfg, checkpoint_path=checkpoint, vocab_path=vocab_path, strict=False, use_deepspeed=False)

#     if torch.cuda.is_available():
#         model.cuda()

#     XTTS_MODEL = model
#     print("XTTS model loaded.")
#     return model

# def _ensure_model_loaded():
#     if XTTS_MODEL is None:
#         raise RuntimeError("XTTS_MODEL 未加载，请先调用 load_xtts_model(...)。")

# def extract_speaker_embedding_no_cache(audio_path: str, model) -> np.ndarray:
#     """
#     对单个音频路径调用 model.get_conditioning_latents(...)，返回 1-D numpy 嵌入。
#     注意：不做缓存；每次调用都会重新执行特征提取（符合你的要求）。
#     """
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio not found: {audio_path}")

#     # 调用模型的 get_conditioning_latents，使用 model.config 中的参数名
#     gpt_cond, speaker_embedding = model.get_conditioning_latents(
#         audio_path=audio_path,
#         gpt_cond_len=getattr(model.config, "gpt_cond_len"),
#         max_ref_length=getattr(model.config, "max_ref_len"),
#         sound_norm_refs=getattr(model.config, "sound_norm_refs"),
#     )

#     # 统一转换为 numpy 1-D 向量
#     if isinstance(speaker_embedding, torch.Tensor):
#         emb = speaker_embedding.detach().cpu().numpy()
#     else:
#         emb = np.array(speaker_embedding, dtype=np.float32)
#     return emb.reshape(-1)

# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     计算余弦相似度；若任一向量为零向量则返回 np.nan（表示不可用）
#     """
#     a = a.reshape(-1).astype(np.float64)
#     b = b.reshape(-1).astype(np.float64)
#     na = np.linalg.norm(a)
#     nb = np.linalg.norm(b)
#     if na == 0 or nb == 0:
#         return float("nan")
#     return float(np.dot(a, b) / (na * nb))

# def compute_cosines_only_no_cache(
#     gen_paths: Sequence[str],
#     ref_paths: Sequence[str],
#     model,
#     print_progress: bool = True
# ) -> List[Tuple[str, str, float]]:
#     """
#     对等长列表 gen_paths/ref_paths 逐对：
#       - 分别单独提取两段音频的说话人嵌入（不缓存）
#       - 只计算余弦相似度
#       - 返回 [(gen_path, ref_path, cosine), ...]
#     """
#     if len(gen_paths) != len(ref_paths):
#         raise ValueError("gen_paths 与 ref_paths 必须等长。")

#     results: List[Tuple[str, str, float]] = []
#     total = len(gen_paths)
#     for i, (gpath, rpath) in enumerate(zip(gen_paths, ref_paths), start=1):
#         try:
#             if print_progress:
#                 print(f"[{i}/{total}] gen: {gpath}\n    ref: {rpath}")

#             emb_g = extract_speaker_embedding_no_cache(gpath, model)
#             emb_r = extract_speaker_embedding_no_cache(rpath, model)

#             cos = cosine_similarity(emb_g, emb_r)
#             if print_progress:
#                 print(f"    -> cosine = {np.nan_to_num(cos):.4f}")
#             results.append((gpath, rpath, cos))
#         except Exception as e:
#             # 不中断批次，记录 nan，继续下一对
#             print(f"    Warning: failed for pair ({gpath}, {rpath}): {e}")
#             results.append((gpath, rpath, float("nan")))
#     return results

# def summarize_and_print_cosines(results: List[Tuple[str, str, float]]):
#     """汇总并打印 mean/std/median（忽略 nan）"""
#     cosines = np.array([r[2] for r in results], dtype=np.float64)
#     valid = cosines[~np.isnan(cosines)]
#     total = len(cosines)
#     valid_n = valid.size
#     print("\n==== Summary ====")
#     print(f"Total pairs: {total} | Valid cosine values: {valid_n}")
#     if valid_n > 0:
#         print(f"Cosine mean: {valid.mean():.4f} | std: {valid.std():.4f} | median: {np.median(valid):.4f}")
#     else:
#         print("No valid cosine values to summarize (all nan).")

# def main():
#     # 检查用户是否已在脚本中填入路径列表
#     if GEN_PATHS is None or REF_PATHS is None:
#         print("Error: 请在脚本顶部将 GEN_PATHS 与 REF_PATHS 赋值为等长列表（方式 B）。")
#         sys.exit(2)
#     if len(GEN_PATHS) != len(REF_PATHS):
#         print("Error: GEN_PATHS 与 REF_PATHS 长度不一致，请检查。")
#         sys.exit(3)

#     try:
#         # 1) 加载模型
#         model = load_xtts_model(MODEL_CHECKPOINT, MODEL_CONFIG, MODEL_VOCAB)

#         # 2) 逐对计算余弦（不缓存）
#         print(f"Computing cosine similarity for {len(GEN_PATHS)} pairs (no cache)...")
#         res = compute_cosines_only_no_cache(GEN_PATHS, REF_PATHS, model, print_progress=True)

#         # 3) 汇总并打印统计
#         summarize_and_print_cosines(res)

#     except Exception:
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
