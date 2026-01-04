# （只用于测试，没有真正使用）测试从音频路径中获取目标音频的情感标签，而不是根据模型预测获取
# import os
# import re
# import csv
# from typing import List, Tuple, Dict, Any

# import torch
# import numpy as np
# from torch.nn import functional as F
# import pandas as pd

# from funasr import AutoModel

# # ============================================================
# #                      配置区域（可按需修改）
# # ============================================================

# # 情感模型路径（本地或远程）
# MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_large"
# # 模型来源："ms" 表示 modelscope，其它如 "hf" 等按你环境设置
# HUB = "ms"

# # 原始 CSV，至少需要 audio_file 列，路径可以是绝对或相对
# CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# # CSV 分隔符（你原来就是用 "|"）
# CSV_DELIM = "|"

# # 当 CSV 中 audio_file 是相对路径时，用它作为前缀（若 CSV 中全是绝对路径，可以无视）
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# # ✅ 这里仍然用“替换路径片段”的方式构造生成音频路径
# # 例：
# #   /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
# # ->/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
# #
# # 即把路径中 "/wavs/" 替换为 "/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/"
# ORIG_SUBDIR = "wavs"
# GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en"

# # 这个参数在新的路径逻辑中暂时不会用到（为了兼容函数签名保留）
# GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/"

# # 是否保存评估结果为 CSV
# SAVE_RESULTS_CSV = True
# # 评估结果 CSV 路径（实际内容由 evaluate_results 写入）
# RESULTS_CSV_PATH = os.path.join(GENERATE_PATH, "emo_sim_results.csv")
# # 保存统计 Summary 的文本文件路径
# RESULTS_TXT_PATH = os.path.join(GENERATE_PATH, "emo_sim_summary.txt")

# # ============================================================
# #                      标签映射与工具函数
# # ============================================================

# # 将可能出现的中英文标签都归一到统一的 canonical 标签
# LABEL_MAP = {
#     "生气": "angry", "angry": "angry",
#     "开心": "happy", "happy": "happy",
#     "难过": "sad", "sad": "sad",
#     "吃惊": "surprise", "surprise": "surprise",
#     "中立": "neutral", "neutral": "neutral",
#     "厌恶": "disgusted", "disgusted": "disgusted",
#     "恐惧": "fearful", "fear": "fearful", "fearful": "fearful",
#     "其他": "other", "other": "other",
# }

# def _normalize_token(tok: str) -> str:
#     """
#     对标签做基础清洗：
#     - 转字符串
#     - 去掉非字母数字/中文字符
#     - 去除两端空白
#     - 小写
#     """
#     if not tok:
#         return ""
#     return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(tok)).strip().lower()

# def normalize_label_to_canonical(label: str) -> str:
#     """
#     将原始情感标签（可能是中英文混写、多标签、带特殊字符等）映射为 canonical 标签。

#     处理流程：
#     1. 先按 "/" 或 "|" 切分标签，得到多个子标签
#     2. 对每个子标签做 normalize，然后在 LABEL_MAP 中查找
#     3. 如果严格匹配不到，再尝试“包含关系”的弱匹配（如 "very_angry" 中包含 "angry"）
#     4. 如果仍然匹配不到，则返回第一个子标签的归一化结果
#     """
#     if not label:
#         return ""
#     # 先按 / 或 | 切分，多标签情况只取第一个匹配成功的
#     parts = [p.strip() for p in re.split(r"[\/|]", label) if p.strip()]

#     # 第一轮：严格匹配
#     for p in parts:
#         p_norm = _normalize_token(p)
#         if p_norm in LABEL_MAP:
#             return LABEL_MAP[p_norm]

#     # 第二轮：弱匹配（key in p_norm）
#     for p in parts:
#         p_norm = _normalize_token(p)
#         for key in LABEL_MAP:
#             if key in p_norm:
#                 return LABEL_MAP[key]

#     # 都匹配不到时，退而求其次，返回第一项的归一化结果
#     return _normalize_token(parts[0]) if parts else _normalize_token(label)

# def pred_top1_from_labels_and_scores(labels: List[str], scores: List[float]) -> Tuple[str, str]:
#     """
#     从模型输出的 labels + scores 中选出分数最高的标签（top1）。

#     返回:
#         (canonical_pred_label, raw_label_string_at_top)

#     说明：
#     - canonical_pred_label 是经过 normalize_label_to_canonical 处理后的结果
#     - raw_label_string_at_top 是 labels 中原始的 top1 字符串
#     """
#     if not labels:
#         return "", ""
#     idx = None
#     try:
#         # 如果 scores 有效且长度等于 labels，则选分数最大的位置
#         if scores and len(scores) == len(labels):
#             scores_f = [float(s) for s in scores]
#             idx = int(np.argmax(np.array(scores_f)))
#     except Exception:
#         idx = None

#     # 如果 scores 异常或为空，则默认取第一个
#     if idx is None:
#         idx = 0

#     raw_top = str(labels[int(idx)])
#     canonical = normalize_label_to_canonical(raw_top)
#     return canonical, raw_top

# # ====== 新增：从 GT 路径中解析“真实情感标签”的函数 ====================

# def extract_emotion_from_path(gt_path: str) -> str:
#     """
#     从 GT 音频路径中解析情感标签，并做 canonical 归一化。

#     按你的规则：
#         /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
#     情感标签 = 倒数第二级目录名 = "Sad" -> 归一化后 "sad"

#     若路径结构不同，也可以按需要修改：
#         - 例如 /.../emotion/speaker/xxx.wav，则可能取倒数第三级
#     """
#     # 取倒数第二级目录名
#     parent_dir = os.path.basename(os.path.dirname(gt_path))  # 这里应为 "Sad"
#     canon = normalize_label_to_canonical(parent_dir)
#     if not canon:
#         # 如果解析失败，可以按需返回 "other" 或 "_unknown"
#         canon = "other"
#     return canon

# # ============================================================
# #                  模型加载与推理辅助函数
# # ============================================================

# def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
#     """
#     加载情感模型（基于 funasr.AutoModel）。
#     """
#     model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
#     return model

# def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
#     """
#     使用模型对单条音频进行推理，提取：
#       - feats: 情感 embedding（向量）
#       - labels: 模型预测的情感标签候选
#       - scores: 对应标签的置信度/分数

#     如果模型没有返回结果或没有 "feats"，会抛出异常。
#     """
#     res = model.generate(
#         wav_path,
#         output_dir="./outputs_temp",
#         granularity="utterance",
#         extract_embedding=True
#     )
#     if not res or len(res) == 0:
#         raise RuntimeError(f"Model.generate returned empty for {wav_path}")

#     entry = res[0]
#     feats = entry.get("feats", None)
#     if feats is None:
#         raise RuntimeError(f"No 'feats' returned for {wav_path}")

#     labels = entry.get("labels", [])
#     scores = entry.get("scores", [])

#     return np.asarray(feats), labels, scores

# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     使用 PyTorch 计算两个 numpy 向量的余弦相似度。
#     """
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)

# # ============================================================
# #                  批量计算情感相似度
# # ============================================================

# def batch_emotion_similarity(
#     model,
#     gen_paths: List[str],
#     gt_paths: List[str],
#     gt_labels_true: List[str],
# ) -> List[Dict[str, Any]]:
#     """
#     对多对 (gen_path, gt_path) 音频进行批量推理和相似度计算。

#     与之前版本的区别（核心改动）：
#         ✅ GT 的“真实情感标签”不再来自模型预测，
#            而是函数参数 gt_labels_true（即从路径中解析出来的标签）。

#     处理逻辑：
#         1. 检查生成和 GT 音频文件是否存在
#         2. 使用模型分别对生成音频和 GT 音频提取 embedding + labels + scores
#         3. 计算 embedding 的余弦相似度
#         4. 对生成音频的 labels + scores 求 top1，得到预测情感（pred_top1）
#         5. 将一路的真实标签（gt_label_true）保存到结果中，供后续统计使用
#     """
#     if not (len(gen_paths) == len(gt_paths) == len(gt_labels_true)):
#         raise ValueError("gen_paths, gt_paths, gt_labels_true must have the same length!")

#     results: List[Dict[str, Any]] = []

#     for i, (gpath, tpath, gt_label_true) in enumerate(
#         zip(gen_paths, gt_paths, gt_labels_true), start=1
#     ):
#         # 基本防御：生成音频和 GT 音频都必须存在，否则直接报错
#         if not os.path.exists(gpath):
#             raise FileNotFoundError(f"Generated audio not found: {gpath}")
#         if not os.path.exists(tpath):
#             raise FileNotFoundError(f"Target audio not found: {tpath}")

#         print(f"[{i}/{len(gen_paths)}] Processing pair:")
#         print(f"  gen: {gpath}")
#         print(f"  gt:  {tpath}")
#         print(f"  gt_label_true: {gt_label_true}")

#         # 对生成音频推理，得到 embedding 和情感标签分布
#         gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
#         # 对 GT 音频推理：这里只为了拿 embedding 算相似度
#         gt_emb, _, _ = _get_embedding_and_scores(model, tpath)

#         # 计算两者的余弦相似度
#         sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#         # 将模型对“生成音频”的输出转成 canonical 的 top1 标签
#         gen_canon, _ = pred_top1_from_labels_and_scores(gen_labels, gen_scores)

#         result = {
#             "gen_path": gpath,
#             "gt_path": tpath,
#             "similarity": sim,

#             # 真实 GT 标签（从路径解析）
#             "gt_label_true": gt_label_true,

#             # 模型对生成音频的预测标签
#             "pred_top1": gen_canon,
#         }
#         results.append(result)

#         print(f"  similarity = {sim:.4f}  pred_top1={gen_canon}  gt_label_true={gt_label_true}")

#     return results

# # ============================================================
# #                       CSV -> 路径构造
# # ============================================================

# def build_paths_from_csv(
#     csv_path: str,
#     csv_delim: str,
#     base_gt_prefix: str,
#     generate_path: str
# ) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
#     """
#     从 CSV 构造:
#         - gt_paths:      GT 音频路径列表
#         - gen_paths:     生成音频路径列表
#         - gt_labels_true:从 GT 路径解析得到的真实情感标签（canonical）
#         - df:           原始 DataFrame

#     路径规则（与你的描述一致）：
#       1) CSV 中的 audio_file 若为绝对路径，则直接作为 GT 路径
#       2) 若为相对路径，则用 base_gt_prefix 拼接成 GT 路径
#       3) 生成音频路径：
#             在 GT 路径中，将 "/ORIG_SUBDIR/" 替换为 "/GEN_SUBDIR/"
#          例如：
#             /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
#          -> /mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
#       4) 真实情感标签：
#          从 GT 路径的“倒数第二级目录名”解析，例如 "Sad" -> "sad"
#     """
#     df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)

#     if "audio_file" not in df.columns:
#         raise ValueError("CSV must contain column: audio_file (and optionally text).")

#     audio_files = df["audio_file"].astype(str).tolist()

#     gt_paths: List[str] = []
#     gen_paths: List[str] = []
#     gt_labels_true: List[str] = []

#     for af in audio_files:
#         af = af.strip()

#         # ---------- 1) 构造 GT 原始音频路径 ----------
#         if os.path.isabs(af):
#             # 若 CSV 中直接给的是绝对路径，则原样使用
#             gt = af
#         else:
#             # 若是相对路径，则用 base_gt_prefix 拼接
#             gt = os.path.join(base_gt_prefix, af)
#         gt = os.path.normpath(gt)
#         gt_paths.append(gt)

#         # ---------- 2) 由 GT 路径构造生成音频路径 ----------
#         pattern = f"/{ORIG_SUBDIR}/"
#         replacement = f"/{GEN_SUBDIR}/"

#         if pattern not in gt:
#             raise ValueError(
#                 f"GT path does not contain '{pattern}': {gt}，"
#                 f"无法按约定规则替换为生成音频路径，请检查 ORIG_SUBDIR 或 CSV。"
#             )

#         gen = gt.replace(pattern, replacement, 1)
#         gen = os.path.normpath(gen)
#         gen_paths.append(gen)

#         # ---------- 3) 从 GT 路径中解析真实情感标签 ----------
#         gt_label_true = extract_emotion_from_path(gt)
#         gt_labels_true.append(gt_label_true)

#     return gt_paths, gen_paths, gt_labels_true, df

# # ============================================================
# #                       评估与统计汇总
# # ============================================================

# def evaluate_results(
#     results: List[Dict[str, Any]],
#     save_csv: bool = False,
#     out_csv: str = None,
#     out_txt: str = None
# ):
#     """
#     对 batch_emotion_similarity 的结果进行统计和保存。

#     ⚠ 重大逻辑变化（符合你当前需求）：
#         - per-class 的分组与计数完全基于 "gt_label_true"
#           （即从 GT 路径解析来的真实标签），
#           而不是模型对 GT 的预测结果。

#         - top-1 accuracy 度量的是：
#               模型对“生成音频”的预测 (pred_top1)
#               vs
#               真实标签 (gt_label_true)
#           是否一致。

#     统计内容：
#         - 总样本数
#         - 相似度的 mean / median / std
#         - top-1 accuracy
#         - 每个情感类别的：
#             * 样本数（真实标签数）
#             * 平均相似度
#             * 预测正确数
#             * 类内 accuracy
#     """
#     total = len(results)

#     sims = np.array(
#         [float(r["similarity"]) for r in results if r.get("similarity") is not None],
#         dtype=float
#     )
#     mean_sim = float(sims.mean()) if sims.size > 0 else None
#     median_sim = float(np.median(sims)) if sims.size > 0 else None
#     std_sim = float(sims.std(ddof=0)) if sims.size > 0 else None

#     top1_correct = 0  # 预测与真实标签完全一致的样本数

#     # 每类别统计信息（基于 true label）
#     per_class_counts: Dict[str, int] = {}
#     per_class_correct: Dict[str, int] = {}
#     per_class_sim_sums: Dict[str, float] = {}

#     for r in results:
#         gt_label_true = (r.get("gt_label_true") or "").strip().lower()
#         pred_label = (r.get("pred_top1") or "").strip().lower()

#         key = gt_label_true if gt_label_true else "_unknown"
#         per_class_counts.setdefault(key, 0)
#         per_class_correct.setdefault(key, 0)
#         per_class_sim_sums.setdefault(key, 0.0)

#         per_class_counts[key] += 1

#         if r.get("similarity") is not None:
#             per_class_sim_sums[key] += float(r.get("similarity"))

#         # 预测与真实标签完全一致才算正确
#         if gt_label_true and pred_label and gt_label_true == pred_label:
#             top1_correct += 1
#             per_class_correct[key] += 1

#     accuracy = top1_correct / total if total > 0 else None

#     per_class_stats: Dict[str, Dict[str, Any]] = {}
#     for k in per_class_counts.keys():
#         cnt = per_class_counts[k]
#         sim_sum = per_class_sim_sums.get(k, 0.0)
#         correct = per_class_correct.get(k, 0)
#         mean_sim_k = (sim_sum / cnt) if cnt > 0 else None
#         acc_k = (correct / cnt) if cnt > 0 else None

#         per_class_stats[k] = {
#             "total": cnt,
#             "mean_similarity": mean_sim_k,
#             "correct": correct,
#             "accuracy": acc_k,
#         }

#     # ---------- 构造 Summary 文本 ----------
#     lines = []
#     lines.append("=== Summary ===")
#     lines.append(f"pairs total: {total}")
#     if mean_sim is not None:
#         lines.append(f"mean similarity: {mean_sim:.4f}")
#     else:
#         lines.append("mean similarity: None")

#     if median_sim is not None:
#         lines.append(f"median similarity: {median_sim:.4f}")
#     else:
#         lines.append("median similarity: None")

#     if std_sim is not None:
#         lines.append(f"std similarity: {std_sim:.4f}")
#     else:
#         lines.append("std similarity: None")

#     if accuracy is not None:
#         lines.append(f"top-1 accuracy: {top1_correct}/{total} = {accuracy:.4f}")
#     else:
#         lines.append("top-1 accuracy: None")

#     lines.append("per-class stats (label: total, mean_sim, correct, accuracy):")

#     for k in sorted(per_class_stats.keys()):
#         v = per_class_stats[k]
#         cnt = v["total"]
#         mean_sim_k = v["mean_similarity"]
#         correct = v["correct"]
#         acc_k = v["accuracy"]

#         mean_sim_k_str = f"{mean_sim_k:.4f}" if mean_sim_k is not None else "None"
#         acc_k_str = f"{acc_k:.4f}" if acc_k is not None else "None"
#         lines.append(f"  {k}: {cnt}, mean_sim={mean_sim_k_str}, correct={correct}, acc={acc_k_str}")

#     # 控制台打印 Summary
#     for ln in lines:
#         print(ln)

#     # 将 Summary 写入文本文件（可选）
#     if out_txt:
#         try:
#             os.makedirs(os.path.dirname(out_txt), exist_ok=True)
#             with open(out_txt, "w", encoding="utf-8") as f:
#                 for ln in lines:
#                     f.write(ln + "\n")
#             print(f"Saved summary text to: {out_txt}")
#         except Exception as e:
#             print(f"Failed to save summary to {out_txt}: {e}")

#     # 写最终 CSV（只写一次，列名固定）
#     if save_csv and out_csv:
#         fieldnames = ["gen_path", "gt_path", "similarity", "gt_label_true", "pred_top1"]
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in results:
#                 writer.writerow({
#                     "gen_path": r.get("gen_path"),
#                     "gt_path": r.get("gt_path"),
#                     "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.4f}",
#                     "gt_label_true": r.get("gt_label_true") or "",
#                     "pred_top1": r.get("pred_top1") or "",
#                 })
#         print(f"Saved evaluation CSV to: {out_csv}")

#     return {
#         "mean_similarity": mean_sim,
#         "median_similarity": median_sim,
#         "std_similarity": std_sim,
#         "top1_accuracy": accuracy,
#         "per_class": per_class_stats,
#     }

# # ============================================================
# #                           main 主流程
# # ============================================================

# if __name__ == "__main__":
#     # 1) 加载情感模型
#     print("Loading emotion model...")
#     model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
#     print("Model loaded.")

#     # 2) 从 CSV 中读取 audio_file，并构造：
#     #    - GT 路径
#     #    - 生成音频路径
#     #    - 真实情感标签（从路径解析）
#     print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#     gt_paths, gen_paths, gt_labels_true, df = build_paths_from_csv(
#         CSV_PATH, CSV_DELIM, BASE_GT_PREFIX, GENERATE_PATH
#     )

#     # 打印前 5 条样例映射，方便你快速确认路径和标签是否正确
#     print("Example mapping (first 5):")
#     for i in range(min(5, len(gt_paths))):
#         print(f"  GT:  {gt_paths[i]}")
#         print(f"  GEN: {gen_paths[i]}")
#         print(f"  GT_LABEL_TRUE: {gt_labels_true[i]}")

#     # 3) 批量计算相似度与情感预测
#     results = batch_emotion_similarity(model, gen_paths, gt_paths, gt_labels_true)

#     # 4) 汇总评估指标，并按配置写入 CSV 和 Summary 文本
#     eval_summary = evaluate_results(
#         results,
#         save_csv=SAVE_RESULTS_CSV,
#         out_csv=RESULTS_CSV_PATH if SAVE_RESULTS_CSV else None,
#         out_txt=RESULTS_TXT_PATH,
#     )








# 计算情感一致性和情感分类准确率（适配emospeech生成的音频）
# import os
# import re
# import csv
# from typing import List, Tuple, Dict, Any

# import torch
# import numpy as np
# from torch.nn import functional as F
# import pandas as pd

# from funasr import AutoModel

# # ============================================================
# #                      配置区域（可按需修改）
# # ============================================================

# # 情感模型路径（本地或远程）
# MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_large"
# # 模型来源："ms" 表示 modelscope，其它如 "hf" 等按你环境设置
# HUB = "ms"

# # 原始 CSV，至少需要 audio_file 列，路径可以是绝对或相对
# CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# # CSV 分隔符（你原来就是用 "|"）
# CSV_DELIM = "|"

# # 当 CSV 中 audio_file 是相对路径时，用它作为前缀（若 CSV 中全是绝对路径，可以无视）
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# # ✅ 新增：生成音频固定目录（你描述的 deepvk_test_4.1803 generated）
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1803/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1774/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1591/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1453/generated/"
# # GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1446/generated/"
# GEN_DIR_FIXED = "/mnt/hd/zjy/emospeech/app/data/deepvk_test_pretrain_4.1504/generated/"

# # 是否保存评估结果为 CSV
# SAVE_RESULTS_CSV = True

# # 评估结果输出目录：这里用 GEN_DIR_FIXED 作为输出目录（你也可以改成别的）
# RESULTS_CSV_PATH = os.path.join(GEN_DIR_FIXED, "emo_sim_results.csv")
# RESULTS_TXT_PATH = os.path.join(GEN_DIR_FIXED, "emo_sim_summary.txt")

# # ============================================================
# #                      标签映射与工具函数
# # ============================================================

# LABEL_MAP = {
#     "生气": "angry", "angry": "angry",
#     "开心": "happy", "happy": "happy",
#     "难过": "sad", "sad": "sad",
#     "吃惊": "surprise", "surprise": "surprise",
#     "中立": "neutral", "neutral": "neutral",
#     "厌恶": "disgusted", "disgusted": "disgusted",
#     "恐惧": "fearful", "fear": "fearful", "fearful": "fearful",
#     "其他": "other", "other": "other",
# }

# def _normalize_token(tok: str) -> str:
#     if not tok:
#         return ""
#     return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(tok)).strip().lower()

# def normalize_label_to_canonical(label: str) -> str:
#     if not label:
#         return ""
#     parts = [p.strip() for p in re.split(r"[\/|]", label) if p.strip()]

#     for p in parts:
#         p_norm = _normalize_token(p)
#         if p_norm in LABEL_MAP:
#             return LABEL_MAP[p_norm]

#     for p in parts:
#         p_norm = _normalize_token(p)
#         for key in LABEL_MAP:
#             if key in p_norm:
#                 return LABEL_MAP[key]

#     return _normalize_token(parts[0]) if parts else _normalize_token(label)

# def pred_top1_from_labels_and_scores(labels: List[str], scores: List[float]) -> Tuple[str, str]:
#     if not labels:
#         return "", ""
#     idx = None
#     try:
#         if scores and len(scores) == len(labels):
#             scores_f = [float(s) for s in scores]
#             idx = int(np.argmax(np.array(scores_f)))
#     except Exception:
#         idx = None

#     if idx is None:
#         idx = 0

#     raw_top = str(labels[int(idx)])
#     canonical = normalize_label_to_canonical(raw_top)
#     return canonical, raw_top

# # ============================================================
# #                  模型加载与推理辅助函数
# # ============================================================

# def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
#     model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
#     return model

# def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
#     res = model.generate(
#         wav_path,
#         output_dir="./outputs_temp",
#         granularity="utterance",
#         extract_embedding=True
#     )
#     if not res or len(res) == 0:
#         raise RuntimeError(f"Model.generate returned empty for {wav_path}")

#     entry = res[0]
#     feats = entry.get("feats", None)
#     if feats is None:
#         raise RuntimeError(f"No 'feats' returned for {wav_path}")

#     labels = entry.get("labels", [])
#     scores = entry.get("scores", [])
#     return np.asarray(feats), labels, scores

# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)

# # ============================================================
# #                  批量计算情感相似度
# # ============================================================

# def batch_emotion_similarity(
#     model,
#     gen_paths: List[str],
#     gt_paths: List[str],
# ) -> List[Dict[str, Any]]:
#     if len(gen_paths) != len(gt_paths):
#         raise ValueError("gen_paths and gt_paths must be the same length!")

#     results: List[Dict[str, Any]] = []

#     for i, (gpath, tpath) in enumerate(zip(gen_paths, gt_paths), start=1):
#         if not os.path.exists(gpath):
#             raise FileNotFoundError(f"Generated audio not found: {gpath}")
#         if not os.path.exists(tpath):
#             raise FileNotFoundError(f"Target audio not found: {tpath}")

#         print(f"[{i}/{len(gen_paths)}] Processing pair:")
#         print(f"  gen: {gpath}")
#         print(f"  gt:  {tpath}")

#         gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
#         gt_emb, gt_labels, gt_scores = _get_embedding_and_scores(model, tpath)

#         sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#         gen_canon, _ = pred_top1_from_labels_and_scores(gen_labels, gen_scores)
#         gt_canon, _ = pred_top1_from_labels_and_scores(gt_labels, gt_scores)

#         result = {
#             "gen_path": gpath,
#             "gt_path": tpath,
#             "similarity": sim,
#             "pred_top1": gen_canon,
#             "gt_pred_top1": gt_canon,
#         }
#         results.append(result)

#         print(f"  similarity = {sim:.4f}  pred_top1={gen_canon}  gt_pred_top1={gt_canon}")

#     return results

# # ============================================================
# #          ✅ 新增：GT 路径 -> 生成音频路径 的转换封装
# # ============================================================

# # 仅支持你描述的 5 类情感目录 -> id
# EMO2ID = {
#     "neutral": 0,
#     "angry": 1,
#     "happy": 2,
#     "sad": 3,
#     "surprise": 4,
# }

# # 数据集 -> 文本序号偏移
# DATASET_OFFSETS = {
#     "ESD": 0,
#     "generate_dataset_200": 350,
#     "generate_dataset_600": 550,
# }

# def convert_gt_path_to_gen_path(gt_path: str, gen_dir: str = GEN_DIR_FIXED) -> str:
#     """
#     按你描述的规则，将 GT 音频路径转换为生成音频路径。

#     示例：
#       /mnt/hd/zjy/wavs/ESD/0020/Angry/0020_000299.wav
#     ->/mnt/hd/zjy/emospeech/app/data/deepvk_test_4.1803/generated/10_299_1.wav
#     """
#     gt_path = os.path.normpath(gt_path)

#     # ---------- 1) 判断数据集与 offset ----------
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

#     # ---------- 2) 判断情感目录并映射到 emo_id ----------
#     emo_dir = None
#     # 目录名可能是 Angry 或 angry，这里做大小写不敏感匹配
#     gt_lower = gt_path.lower()
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

#     # ---------- 3) 解析文件名 speaker 与 idx ----------
#     base = os.path.basename(gt_path)  # e.g. 0020_000299.wav
#     m = re.match(r"^(\d+)_([0-9]+)\.wav$", base, flags=re.IGNORECASE)
#     if not m:
#         raise ValueError(
#             f"Unexpected filename format: {base}. Expected like 0020_000299.wav"
#         )
#     spk_str, idx_str = m.group(1), m.group(2)

#     spk = int(spk_str)      # 自动去前导 0
#     idx = int(idx_str)      # 自动去前导 0

#     spk_new = spk - 10
#     if spk_new < 0:
#         raise ValueError(f"speaker_id - 10 becomes negative for {base}: {spk} -> {spk_new}")

#     idx_new = idx + offset

#     # ---------- 4) 生成最终路径 ----------
#     gen_filename = f"{spk_new}_{idx_new}_{emo_id}.wav"
#     gen_path = os.path.normpath(os.path.join(gen_dir, gen_filename))
#     return gen_path

# # ============================================================
# #                       CSV -> 路径构造（调用新规则）
# # ============================================================

# def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str):
#     df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)

#     if "audio_file" not in df.columns:
#         raise ValueError("CSV must contain column: audio_file (and optionally text).")

#     audio_files = df["audio_file"].astype(str).tolist()

#     gt_paths = []
#     gen_paths = []

#     for af in audio_files:
#         af = af.strip()

#         # 1) 构造 GT 路径
#         if os.path.isabs(af):
#             gt = af
#         else:
#             gt = os.path.join(base_gt_prefix, af)
#         gt = os.path.normpath(gt)
#         gt_paths.append(gt)

#         # 2) 用新规则构造生成音频路径
#         gen = convert_gt_path_to_gen_path(gt, gen_dir=GEN_DIR_FIXED)
#         gen_paths.append(gen)

#     return gt_paths, gen_paths, df

# # ============================================================
# #                       评估与统计汇总
# # ============================================================

# def evaluate_results(
#     results: List[Dict[str, Any]],
#     save_csv: bool = False,
#     out_csv: str = None,
#     out_txt: str = None
# ):
#     total = len(results)

#     sims = np.array(
#         [float(r["similarity"]) for r in results if r.get("similarity") is not None],
#         dtype=float
#     )
#     mean_sim = float(sims.mean()) if sims.size > 0 else None
#     median_sim = float(np.median(sims)) if sims.size > 0 else None
#     std_sim = float(sims.std(ddof=0)) if sims.size > 0 else None

#     top1_correct = 0

#     per_class_counts: Dict[str, int] = {}
#     per_class_correct: Dict[str, int] = {}
#     per_class_sim_sums: Dict[str, float] = {}

#     for r in results:
#         gt_label = (r.get("gt_pred_top1") or "").strip().lower()
#         pred_label = (r.get("pred_top1") or "").strip().lower()

#         key = gt_label if gt_label else "_unknown"
#         per_class_counts.setdefault(key, 0)
#         per_class_correct.setdefault(key, 0)
#         per_class_sim_sums.setdefault(key, 0.0)

#         per_class_counts[key] += 1

#         if r.get("similarity") is not None:
#             per_class_sim_sums[key] += float(r.get("similarity"))

#         if gt_label and pred_label and gt_label == pred_label:
#             top1_correct += 1
#             per_class_correct[key] += 1

#     accuracy = top1_correct / total if total > 0 else None

#     per_class_stats: Dict[str, Dict[str, Any]] = {}
#     for k in per_class_counts.keys():
#         cnt = per_class_counts[k]
#         sim_sum = per_class_sim_sums.get(k, 0.0)
#         correct = per_class_correct.get(k, 0)
#         mean_sim_k = (sim_sum / cnt) if cnt > 0 else None
#         acc_k = (correct / cnt) if cnt > 0 else None

#         per_class_stats[k] = {
#             "total": cnt,
#             "mean_similarity": mean_sim_k,
#             "correct": correct,
#             "accuracy": acc_k,
#         }

#     lines = []
#     lines.append("=== Summary ===")
#     lines.append(f"pairs total: {total}")
#     lines.append(f"mean similarity: {mean_sim:.4f}" if mean_sim is not None else "mean similarity: None")
#     lines.append(f"median similarity: {median_sim:.4f}" if median_sim is not None else "median similarity: None")
#     lines.append(f"std similarity: {std_sim:.4f}" if std_sim is not None else "std similarity: None")
#     lines.append(f"top-1 accuracy: {top1_correct}/{total} = {accuracy:.4f}" if accuracy is not None else "top-1 accuracy: None")

#     lines.append("per-class stats (label: total, mean_sim, correct, accuracy):")
#     for k in sorted(per_class_stats.keys()):
#         v = per_class_stats[k]
#         cnt = v["total"]
#         mean_sim_k = v["mean_similarity"]
#         correct = v["correct"]
#         acc_k = v["accuracy"]

#         mean_sim_k_str = f"{mean_sim_k:.4f}" if mean_sim_k is not None else "None"
#         acc_k_str = f"{acc_k:.4f}" if acc_k is not None else "None"
#         lines.append(f"  {k}: {cnt}, mean_sim={mean_sim_k_str}, correct={correct}, acc={acc_k_str}")

#     for ln in lines:
#         print(ln)

#     if out_txt:
#         try:
#             os.makedirs(os.path.dirname(out_txt), exist_ok=True)
#             with open(out_txt, "w", encoding="utf-8") as f:
#                 for ln in lines:
#                     f.write(ln + "\n")
#             print(f"Saved summary text to: {out_txt}")
#         except Exception as e:
#             print(f"Failed to save summary to {out_txt}: {e}")

#     if save_csv and out_csv:
#         fieldnames = ["gen_path", "gt_path", "similarity", "gt_pred_top1", "pred_top1"]
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in results:
#                 writer.writerow({
#                     "gen_path": r.get("gen_path"),
#                     "gt_path": r.get("gt_path"),
#                     "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.4f}",
#                     "gt_pred_top1": r.get("gt_pred_top1") or "",
#                     "pred_top1": r.get("pred_top1") or "",
#                 })
#         print(f"Saved evaluation CSV to: {out_csv}")

#     return {
#         "mean_similarity": mean_sim,
#         "median_similarity": median_sim,
#         "std_similarity": std_sim,
#         "top1_accuracy": accuracy,
#         "per_class": per_class_stats,
#     }

# # ============================================================
# #                           main 主流程
# # ============================================================

# if __name__ == "__main__":
#     # 1) 加载情感模型
#     print("Loading emotion model...")
#     model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
#     print("Model loaded.")

#     # 2) 从 CSV 中读取 audio_file，并构造 GT 与生成音频路径
#     print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#     gt_paths, gen_paths, df = build_paths_from_csv(
#         CSV_PATH, CSV_DELIM, BASE_GT_PREFIX
#     )

#     # 打印前 5 条样例映射，方便你快速确认路径是否正确
#     print("Example mapping (first 5):")
#     for i in range(min(5, len(gt_paths))):
#         print(f"  GT:  {gt_paths[i]}")
#         print(f"  GEN: {gen_paths[i]}")

#     # 3) 批量计算相似度与情感预测（GT 情感也由模型推理得到）
#     results = batch_emotion_similarity(model, gen_paths, gt_paths)

#     # 4) 汇总评估指标，并按配置写入 CSV 和 Summary 文本
#     eval_summary = evaluate_results(
#         results,
#         save_csv=SAVE_RESULTS_CSV,
#         out_csv=RESULTS_CSV_PATH if SAVE_RESULTS_CSV else None,
#         out_txt=RESULTS_TXT_PATH,
#     )









# 计算情感一致性和情感分类准确率（适用etd-tts生成音频）
# import os
# import re
# import csv
# from typing import List, Tuple, Dict, Any

# import torch
# import numpy as np
# from torch.nn import functional as F
# import pandas as pd

# from funasr import AutoModel

# # ============================================================
# #                      配置区域（可按需修改）
# # ============================================================

# # 情感模型路径（本地或远程）
# MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_large"
# # 模型来源："ms" 表示 modelscope，其它如 "hf" 等按你环境设置
# HUB = "ms"

# # 原始 CSV，至少需要 audio_file 列，路径可以是绝对或相对
# CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# # CSV 分隔符（你原来就是用 "|"）
# CSV_DELIM = "|"

# # 当 CSV 中 audio_file 是相对路径时，用它作为前缀（若 CSV 中全是绝对路径，可以无视）
# BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# # ✅ GEN 音频固定根目录（你的最新需求）
# # GEN_ROOT = "/mnt/hd/zjy/test/etd-tts_no-ft-hifigan/"
# GEN_ROOT = "/mnt/hd/zjy/test/etd-tts_ft-hifigan/"

# # 评估结果输出目录（建议与 GEN_ROOT 一致，方便管理）
# GENERATE_PATH = GEN_ROOT

# # 是否保存评估结果为 CSV
# SAVE_RESULTS_CSV = True
# # 评估结果 CSV 路径（实际内容由 evaluate_results 写入）
# RESULTS_CSV_PATH = os.path.join(GENERATE_PATH, "emo_sim_results.csv")
# # 保存统计 Summary 的文本文件路径
# RESULTS_TXT_PATH = os.path.join(GENERATE_PATH, "emo_sim_summary.txt")

# # ============================================================
# #      ✅ GT -> GEN 文件名映射（按 offset + 前后缀规则）
# # ============================================================

# # 你的情感集合（Title Case，与目录一致）
# EMOTIONS_TITLE = {"Neutral", "Angry", "Happy", "Sad", "Surprise"}

# # offset 规则（与你给出的最终规则一致）
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
#     假设路径结构：.../wavs/<subset>/...
#     例如：.../wavs/ESD/0020/Angry/0020_000299.wav
#     """
#     parts = gt_path.split("/")
#     try:
#         i = parts.index("wavs")
#         subset = parts[i + 1]
#     except Exception:
#         raise ValueError(f"Cannot detect subset from GT path: {gt_path}")

#     if subset not in OFFSETS:
#         raise ValueError(f"Unknown subset '{subset}' from GT path: {gt_path}")
#     return subset


# def detect_emotion_from_gt_path(gt_path: str) -> str:
#     """
#     从 GT 路径目录名中找 emotion（Neutral/Angry/Happy/Sad/Surprise）。
#     返回 Title Case（与目录一致）。
#     """
#     for p in gt_path.split("/"):
#         if p in EMOTIONS_TITLE:
#             return p
#     raise ValueError(f"Cannot detect emotion from GT path: {gt_path}")


# def build_gen_filename_from_gt_filename(gt_filename: str, subset: str, emotion: str) -> str:
#     """
#     GT: 0020_000299.wav
#     先按 offset 得到：0020_001449.wav
#     再拼接：0020_Angry_0020_001449_generated_e2e.wav
#     """
#     m = FNAME_RE.match(gt_filename)
#     if not m:
#         raise ValueError(f"Bad GT filename format: {gt_filename}")

#     spk = m.group("spk")
#     old_idx = int(m.group("idx"))

#     offset = OFFSETS[subset][emotion]
#     new_idx = old_idx + offset

#     core = f"{spk}_{new_idx:06d}"  # 不带 .wav
#     final_name = f"{spk}_{emotion}_{core}_generated_e2e.wav"
#     return final_name


# # ============================================================
# #                      标签映射与工具函数
# # ============================================================

# LABEL_MAP = {
#     "生气": "angry", "angry": "angry",
#     "开心": "happy", "happy": "happy",
#     "难过": "sad", "sad": "sad",
#     "吃惊": "surprise", "surprise": "surprise",
#     "中立": "neutral", "neutral": "neutral",
#     "厌恶": "disgusted", "disgusted": "disgusted",
#     "恐惧": "fearful", "fear": "fearful", "fearful": "fearful",
#     "其他": "other", "other": "other",
# }


# def _normalize_token(tok: str) -> str:
#     if not tok:
#         return ""
#     return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(tok)).strip().lower()


# def normalize_label_to_canonical(label: str) -> str:
#     if not label:
#         return ""
#     parts = [p.strip() for p in re.split(r"[\/|]", label) if p.strip()]

#     for p in parts:
#         p_norm = _normalize_token(p)
#         if p_norm in LABEL_MAP:
#             return LABEL_MAP[p_norm]

#     for p in parts:
#         p_norm = _normalize_token(p)
#         for key in LABEL_MAP:
#             if key in p_norm:
#                 return LABEL_MAP[key]

#     return _normalize_token(parts[0]) if parts else _normalize_token(label)


# def pred_top1_from_labels_and_scores(labels: List[str], scores: List[float]) -> Tuple[str, str]:
#     if not labels:
#         return "", ""
#     idx = None
#     try:
#         if scores and len(scores) == len(labels):
#             scores_f = [float(s) for s in scores]
#             idx = int(np.argmax(np.array(scores_f)))
#     except Exception:
#         idx = None

#     if idx is None:
#         idx = 0

#     raw_top = str(labels[int(idx)])
#     canonical = normalize_label_to_canonical(raw_top)
#     return canonical, raw_top


# # ============================================================
# #                  模型加载与推理辅助函数
# # ============================================================

# def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
#     model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
#     return model


# def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
#     res = model.generate(wav_path, output_dir="./outputs_temp",
#                          granularity="utterance", extract_embedding=True)
#     if not res or len(res) == 0:
#         raise RuntimeError(f"Model.generate returned empty for {wav_path}")

#     entry = res[0]
#     feats = entry.get("feats", None)
#     if feats is None:
#         raise RuntimeError(f"No 'feats' returned for {wav_path}")

#     labels = entry.get("labels", [])
#     scores = entry.get("scores", [])
#     return np.asarray(feats), labels, scores


# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)


# # ============================================================
# #                  批量计算情感相似度
# # ============================================================

# def batch_emotion_similarity(model, gen_paths: List[str], gt_paths: List[str]) -> List[Dict[str, Any]]:
#     if len(gen_paths) != len(gt_paths):
#         raise ValueError("gen_paths and gt_paths must be the same length!")

#     results: List[Dict[str, Any]] = []

#     for i, (gpath, tpath) in enumerate(zip(gen_paths, gt_paths), start=1):
#         if not os.path.exists(gpath):
#             raise FileNotFoundError(f"Generated audio not found: {gpath}")
#         if not os.path.exists(tpath):
#             raise FileNotFoundError(f"Target audio not found: {tpath}")

#         print(f"[{i}/{len(gen_paths)}] Processing pair:")
#         print(f"  gen: {gpath}")
#         print(f"  gt:  {tpath}")

#         gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
#         gt_emb, gt_labels, gt_scores = _get_embedding_and_scores(model, tpath)

#         sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#         gen_canon, _ = pred_top1_from_labels_and_scores(gen_labels, gen_scores)
#         gt_canon, _ = pred_top1_from_labels_and_scores(gt_labels, gt_scores)

#         result = {
#             "gen_path": gpath,
#             "gt_path": tpath,
#             "similarity": sim,
#             "pred_top1": gen_canon,
#             "gt_pred_top1": gt_canon,
#         }
#         results.append(result)

#         print(f"  similarity = {sim:.4f}  pred_top1={gen_canon}  gt_pred_top1={gt_canon}")

#     return results


# # ============================================================
# #                       CSV -> 路径构造（✅最新：固定 GEN_ROOT + 文件名映射）
# # ============================================================

# def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str, generate_path: str):
#     """
#     从 CSV 构造 gt_paths 与 gen_paths。

#     最新规则：
#       - GT：从 CSV 的 audio_file 得到（绝对路径直接用，相对路径用 base_gt_prefix 拼）
#       - GEN：
#          1) 从 GT 路径解析 subset + emotion
#          2) 用 GT 文件名 + offset 规则生成 gen 文件名：
#               0020_000299.wav -> 0020_Angry_0020_001449_generated_e2e.wav
#          3) gen_path = GEN_ROOT / gen_filename
#     """
#     df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)

#     if "audio_file" not in df.columns:
#         raise ValueError("CSV must contain column: audio_file (and optionally text).")

#     audio_files = df["audio_file"].astype(str).tolist()

#     gt_paths = []
#     gen_paths = []

#     gen_root = os.path.normpath(GEN_ROOT)

#     for af in audio_files:
#         af = af.strip()

#         # ---------- 1) 构造 GT 原始音频路径 ----------
#         if os.path.isabs(af):
#             gt = af
#         else:
#             gt = os.path.join(base_gt_prefix, af)
#         gt = os.path.normpath(gt)
#         gt_paths.append(gt)

#         # ---------- 2) 构造 GEN 路径：固定目录 + 新文件名 ----------
#         subset = detect_subset_from_gt_path(gt)
#         emotion = detect_emotion_from_gt_path(gt)
#         gt_filename = os.path.basename(gt)

#         gen_filename = build_gen_filename_from_gt_filename(gt_filename, subset, emotion)
#         gen = os.path.join(gen_root, gen_filename)
#         gen = os.path.normpath(gen)
#         gen_paths.append(gen)

#     return gt_paths, gen_paths, df


# # ============================================================
# #                       评估与统计汇总
# # ============================================================

# def evaluate_results(results: List[Dict[str, Any]], save_csv: bool = False, out_csv: str = None, out_txt: str = None):
#     total = len(results)

#     sims = np.array([float(r["similarity"]) for r in results if r.get("similarity") is not None], dtype=float)
#     mean_sim = float(sims.mean()) if sims.size > 0 else None
#     median_sim = float(np.median(sims)) if sims.size > 0 else None
#     std_sim = float(sims.std(ddof=0)) if sims.size > 0 else None

#     top1_correct = 0

#     per_class_counts: Dict[str, int] = {}
#     per_class_correct: Dict[str, int] = {}
#     per_class_sim_sums: Dict[str, float] = {}

#     for r in results:
#         gt_label = (r.get("gt_pred_top1") or "").strip().lower()
#         pred_label = (r.get("pred_top1") or "").strip().lower()

#         key = gt_label if gt_label else "_unknown"
#         per_class_counts.setdefault(key, 0)
#         per_class_correct.setdefault(key, 0)
#         per_class_sim_sums.setdefault(key, 0.0)

#         per_class_counts[key] += 1

#         if r.get("similarity") is not None:
#             per_class_sim_sums[key] += float(r.get("similarity"))

#         if gt_label and pred_label and gt_label == pred_label:
#             top1_correct += 1
#             per_class_correct[key] += 1

#     accuracy = top1_correct / total if total > 0 else None

#     per_class_stats: Dict[str, Dict[str, Any]] = {}
#     for k in per_class_counts.keys():
#         cnt = per_class_counts[k]
#         sim_sum = per_class_sim_sums.get(k, 0.0)
#         correct = per_class_correct.get(k, 0)
#         mean_sim_k = (sim_sum / cnt) if cnt > 0 else None
#         acc_k = (correct / cnt) if cnt > 0 else None

#         per_class_stats[k] = {
#             "total": cnt,
#             "mean_similarity": mean_sim_k,
#             "correct": correct,
#             "accuracy": acc_k,
#         }

#     lines = []
#     lines.append("=== Summary ===")
#     lines.append(f"pairs total: {total}")
#     lines.append(f"mean similarity: {mean_sim:.4f}" if mean_sim is not None else "mean similarity: None")
#     lines.append(f"median similarity: {median_sim:.4f}" if median_sim is not None else "median similarity: None")
#     lines.append(f"std similarity: {std_sim:.4f}" if std_sim is not None else "std similarity: None")
#     lines.append(f"top-1 accuracy: {top1_correct}/{total} = {accuracy:.4f}" if accuracy is not None else "top-1 accuracy: None")
#     lines.append("per-class stats (label: total, mean_sim, correct, accuracy):")

#     for k in sorted(per_class_stats.keys()):
#         v = per_class_stats[k]
#         cnt = v["total"]
#         mean_sim_k = v["mean_similarity"]
#         correct = v["correct"]
#         acc_k = v["accuracy"]

#         mean_sim_k_str = f"{mean_sim_k:.4f}" if mean_sim_k is not None else "None"
#         acc_k_str = f"{acc_k:.4f}" if acc_k is not None else "None"
#         lines.append(f"  {k}: {cnt}, mean_sim={mean_sim_k_str}, correct={correct}, acc={acc_k_str}")

#     for ln in lines:
#         print(ln)

#     if out_txt:
#         try:
#             os.makedirs(os.path.dirname(out_txt), exist_ok=True)
#             with open(out_txt, "w", encoding="utf-8") as f:
#                 for ln in lines:
#                     f.write(ln + "\n")
#             print(f"Saved summary text to: {out_txt}")
#         except Exception as e:
#             print(f"Failed to save summary to {out_txt}: {e}")

#     if save_csv and out_csv:
#         fieldnames = ["gen_path", "gt_path", "similarity", "gt_pred_top1", "pred_top1"]
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in results:
#                 writer.writerow({
#                     "gen_path": r.get("gen_path"),
#                     "gt_path": r.get("gt_path"),
#                     "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.4f}",
#                     "gt_pred_top1": r.get("gt_pred_top1") or "",
#                     "pred_top1": r.get("pred_top1") or "",
#                 })
#         print(f"Saved evaluation CSV to: {out_csv}")

#     return {
#         "mean_similarity": mean_sim,
#         "median_similarity": median_sim,
#         "std_similarity": std_sim,
#         "top1_accuracy": accuracy,
#         "per_class": per_class_stats,
#     }


# # ============================================================
# #                           main 主流程
# # ============================================================

# if __name__ == "__main__":
#     print("Loading emotion model...")
#     model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
#     print("Model loaded.")

#     print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#     gt_paths, gen_paths, df = build_paths_from_csv(
#         CSV_PATH, CSV_DELIM, BASE_GT_PREFIX, GENERATE_PATH
#     )

#     print("Example mapping (first 5):")
#     for i in range(min(5, len(gt_paths))):
#         print(f"  GT:  {gt_paths[i]}")
#         print(f"  GEN: {gen_paths[i]}")

#     results = batch_emotion_similarity(model, gen_paths, gt_paths)

#     eval_summary = evaluate_results(
#         results,
#         save_csv=SAVE_RESULTS_CSV,
#         out_csv=RESULTS_CSV_PATH if SAVE_RESULTS_CSV else None,
#         out_txt=RESULTS_TXT_PATH,
#     )

















# 计算情感一致性和情感分类准确率（适用更改文件名后的ESD数据，即序号为0-350）
import os
import re
import csv
from typing import List, Tuple, Dict, Any

import torch
import numpy as np
from torch.nn import functional as F
import pandas as pd

from funasr import AutoModel

# ============================================================
#                      配置区域（可按需修改）
# ============================================================

# 情感模型路径（本地或远程）
MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_large"
# 模型来源："ms" 表示 modelscope，其它如 "hf" 等按你环境设置
HUB = "ms"

# 原始 CSV，至少需要 audio_file 列，路径可以是绝对或相对
CSV_PATH = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/metadata_test.csv"
# CSV 分隔符（你原来就是用 "|"）
CSV_DELIM = "|"

# 当 CSV 中 audio_file 是相对路径时，用它作为前缀（若 CSV 中全是绝对路径，可以无视）
BASE_GT_PREFIX = "/mnt/hd/zjy/y23_zjy_data/xtts_ft/dataset/"

# ✅ 新增：通过“替换路径片段”的方式构造生成音频路径
# 例：
#   /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
# ->/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav
#
# 即把路径中 "/wavs/" 替换为 "/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/"
ORIG_SUBDIR = "wavs"

# GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en"
# GEN_SUBDIR = "test/no_emo_labels_en"
# GEN_SUBDIR = "test/5_emo_labels_en"
# GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_295020"
# GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_364752"
# GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_375480"
# GEN_SUBDIR = "test/no_ecl_emo_labels_gpt-last-layer"
# GEN_SUBDIR = "test/no_pre-ft"
GEN_SUBDIR = "test/target_audios"

# mixed_emotions模型权重
# GEN_SUBDIR = "test/generate_from_pretrain_embedding"
# GEN_SUBDIR = "test/generate_from_test_embedding"
# GEN_SUBDIR = "test/generate_from_train_embedding"

# 这个参数在新的路径逻辑中暂时不会用到（为了兼容函数签名保留）
# GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/"
# GENERATE_PATH = "/mnt/hd/zjy/test/no_emo_labels_en/"
# GENERATE_PATH = "/mnt/hd/zjy/test/5_emo_labels_en/"
# GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_295020/"
# GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_364752/"
# GENERATE_PATH = "/mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en_375480/"
GENERATE_PATH = "/mnt/hd/zjy/test/target_audios/"

# mixed_emotions模型权重
# GENERATE_PATH = "/mnt/hd/zjy/test/generate_from_pretrain_embedding/"
# GENERATE_PATH = "/mnt/hd/zjy/test/generate_from_test_embedding/"
# GENERATE_PATH = "/mnt/hd/zjy/test/generate_from_train_embedding/"

# 是否保存评估结果为 CSV
SAVE_RESULTS_CSV = True
# 评估结果 CSV 路径（实际内容由 evaluate_results 写入）
RESULTS_CSV_PATH = os.path.join(GENERATE_PATH, "emo_sim_results.csv")
# 新增：保存统计 Summary 的文本文件路径
RESULTS_TXT_PATH = os.path.join(GENERATE_PATH, "emo_sim_summary.txt")

# ============================================================
#                      标签映射与工具函数
# ============================================================

# 将可能出现的中英文标签都归一到统一的 canonical 标签
LABEL_MAP = {
    "生气": "angry", "angry": "angry",
    "开心": "happy", "happy": "happy",
    "难过": "sad", "sad": "sad",
    "吃惊": "surprise", "surprise": "surprise",
    "中立": "neutral", "neutral": "neutral",
    "厌恶": "disgusted", "disgusted": "disgusted",
    "恐惧": "fearful", "fear": "fearful", "fearful": "fearful",
    "其他": "other", "other": "other",
}

def _normalize_token(tok: str) -> str:
    """
    对标签做基础清洗：
    - 转字符串
    - 去掉非字母数字/中文字符
    - 去除两端空白
    - 小写
    """
    if not tok:
        return ""
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(tok)).strip().lower()

def normalize_label_to_canonical(label: str) -> str:
    """
    将原始情感标签（可能是中英文混写、多标签、带特殊字符等）映射为 canonical 标签。

    处理流程：
    1. 先按 "/" 或 "|" 切分标签，得到多个子标签
    2. 对每个子标签做 normalize，然后直接在 LABEL_MAP 中查找
    3. 如果完全匹配不到，再尝试“包含关系”的弱匹配（如 "very_angry" 中包含 "angry"）
    4. 如果仍然匹配不到，则返回第一个子标签的归一化结果
    """
    if not label:
        return ""
    # 先按 / 或 | 切分，多标签情况只取第一个匹配成功的
    parts = [p.strip() for p in re.split(r"[\/|]", label) if p.strip()]

    # 第一轮：严格匹配
    for p in parts:
        p_norm = _normalize_token(p)
        if p_norm in LABEL_MAP:
            return LABEL_MAP[p_norm]

    # 第二轮：弱匹配（key in p_norm）
    for p in parts:
        p_norm = _normalize_token(p)
        for key in LABEL_MAP:
            if key in p_norm:
                return LABEL_MAP[key]

    # 都匹配不到时，退而求其次，返回第一项的归一化结果
    return _normalize_token(parts[0]) if parts else _normalize_token(label)

def pred_top1_from_labels_and_scores(labels: List[str], scores: List[float]) -> Tuple[str, str]:
    """
    从模型输出的 labels + scores 中选出分数最高的标签（top1）。

    返回:
        (canonical_pred_label, raw_label_string_at_top)

    说明：
    - canonical_pred_label 是经过 normalize_label_to_canonical 处理后的结果
    - raw_label_string_at_top 是 labels 中原始的 top1 字符串
    """
    if not labels:
        return "", ""
    idx = None
    try:
        # 如果 scores 有效且长度等于 labels，则选分数最大的位置
        if scores and len(scores) == len(labels):
            scores_f = [float(s) for s in scores]
            idx = int(np.argmax(np.array(scores_f)))
    except Exception:
        idx = None

    # 如果 scores 异常或为空，则默认取第一个
    if idx is None:
        idx = 0

    raw_top = str(labels[int(idx)])
    canonical = normalize_label_to_canonical(raw_top)
    return canonical, raw_top

# ============================================================
#                  模型加载与推理辅助函数
# ============================================================

def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
    """
    加载情感模型（基于 funasr.AutoModel）。

    参数:
        model_name_or_path: 模型路径或名称
        hub: 模型来源（如 "ms"）

    返回:
        已加载的模型对象，可以直接调用 model.generate(...)
    """
    model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
    return model

def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
    """
    使用模型对单条音频进行推理，提取：
      - feats: 情感 embedding（向量）
      - labels: 模型预测的情感标签候选
      - scores: 对应标签的置信度/分数

    如果模型没有返回结果或没有 "feats"，会抛出异常。
    """
    # granularity="utterance"：按整句级别提取特征
    # extract_embedding=True：让模型返回 embedding
    res = model.generate(wav_path, output_dir="./outputs_temp",
                         granularity="utterance", extract_embedding=True)
    if not res or len(res) == 0:
        raise RuntimeError(f"Model.generate returned empty for {wav_path}")

    entry = res[0]
    feats = entry.get("feats", None)
    if feats is None:
        raise RuntimeError(f"No 'feats' returned for {wav_path}")

    labels = entry.get("labels", [])
    scores = entry.get("scores", [])

    # feats 一般是 list / np.array，这里统一转成 numpy 数组
    return np.asarray(feats), labels, scores

def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """
    使用 PyTorch 计算两个 numpy 向量的余弦相似度。

    步骤:
    1. numpy -> torch tensor
    2. L2 归一化
    3. 点积得到 cos 值
    """
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    a_n = F.normalize(a_t, dim=-1)
    b_n = F.normalize(b_t, dim=-1)
    cos = (a_n * b_n).sum().item()
    return float(cos)

# ============================================================
#                  批量计算情感相似度
# ============================================================

def batch_emotion_similarity(
    model,
    gen_paths: List[str],
    gt_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    对多对 (gen_path, gt_path) 音频进行批量推理和相似度计算。

    处理逻辑：
        1. 对每对音频，先检查文件是否存在；不存在直接抛异常
        2. 使用模型分别对生成音频和 GT 音频提取 embedding + labels + scores
        3. 计算两者 embedding 的余弦相似度
        4. 通过 labels + scores 得到 top1 情感标签（均 canonical 化）
        5. 将结果保存到列表中返回

    重要特性：
        - 若任意一条音频对在推理或文件检查中失败，会立刻抛出异常并中断流程
    """
    if len(gen_paths) != len(gt_paths):
        raise ValueError("gen_paths and gt_paths must be the same length!")

    results: List[Dict[str, Any]] = []

    for i, (gpath, tpath) in enumerate(zip(gen_paths, gt_paths), start=1):
        # 基本防御：生成音频和 GT 音频都必须存在，否则直接报错
        if not os.path.exists(gpath):
            raise FileNotFoundError(f"Generated audio not found: {gpath}")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"Target audio not found: {tpath}")

        print(f"[{i}/{len(gen_paths)}] Processing pair:")
        print(f"  gen: {gpath}")
        print(f"  gt:  {tpath}")

        # 对生成音频推理，得到 embedding 和情感标签分布
        gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
        # 对 GT 音频推理
        gt_emb, gt_labels, gt_scores = _get_embedding_and_scores(model, tpath)

        # 计算两者的余弦相似度
        sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

        # 将模型输出的 labels + scores 转成 canonical 的 top1 标签
        gen_canon, _ = pred_top1_from_labels_and_scores(gen_labels, gen_scores)
        gt_canon, _ = pred_top1_from_labels_and_scores(gt_labels, gt_scores)

        result = {
            "gen_path": gpath,
            "gt_path": tpath,
            "similarity": sim,
            "pred_top1": gen_canon,     # 生成音频的预测情感（canonical）
            "gt_pred_top1": gt_canon,   # GT 音频的预测情感（canonical）
        }
        results.append(result)

        print(f"  similarity = {sim:.4f}  pred_top1={gen_canon}  gt_pred_top1={gt_canon}")

    return results

# ============================================================
#                       CSV -> 路径构造
# ============================================================

def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str, generate_path: str):
    """
    从 CSV 构造 gt_paths 与 gen_paths。

    新规则（按你的要求）：
      - CSV 中的 audio_file 若为绝对路径，则直接作为 GT 路径
      - 若为相对路径，则用 base_gt_prefix 拼接成 GT 路径
      - 生成音频路径 gen_path：
            在 GT 路径中，将 "/ORIG_SUBDIR/" 替换为 "/GEN_SUBDIR/"
        例如：
            ORIG_SUBDIR = "wavs"
            GEN_SUBDIR = "test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en"

            /mnt/hd/zjy/wavs/ESD/0020/Sad/0020_000332.wav
        ->  /mnt/hd/zjy/test/w_ecl_0.5_gpt-last-layer_lr_2e-6_en/ESD/0020/Sad/0020_000332.wav

    参数:
        csv_path:   原始 CSV 文件路径
        csv_delim:  分隔符（如 "|"）
        base_gt_prefix: 当 CSV 中 audio_file 为相对路径时，作为前缀使用
        generate_path:  为了保持函数签名不变暂时保留，不在此逻辑中使用

    返回:
        gt_paths: GT 音频的完整路径列表
        gen_paths: 生成音频的完整路径列表
        df: 读取的原始 DataFrame
    """
    df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)

    # 只强制要求有 audio_file 列，text 等其它列变为可选
    if "audio_file" not in df.columns:
        raise ValueError("CSV must contain column: audio_file (and optionally text).")

    audio_files = df["audio_file"].astype(str).tolist()

    gt_paths = []
    gen_paths = []

    for af in audio_files:
        af = af.strip()

        # ---------- 1) 构造 GT 原始音频路径 ----------
        if os.path.isabs(af):
            # 如果 CSV 中直接给的是绝对路径，则原样使用
            gt = af
        else:
            # 如果是相对路径，则用 base_gt_prefix 拼接
            gt = os.path.join(base_gt_prefix, af)
        gt = os.path.normpath(gt)
        gt_paths.append(gt)

        # ---------- 2) 由 GT 路径构造生成音频路径 ----------
        # 我们期望 GT 路径中包含 "/wavs/" 这段（或你配置的 ORIG_SUBDIR）
        pattern = f"/{ORIG_SUBDIR}/"
        replacement = f"/{GEN_SUBDIR}/"

        if pattern not in gt:
            # 若匹配不到，说明路径结构不符合预期，直接报错提醒你检查配置或 CSV 内容
            raise ValueError(
                f"GT path does not contain '{pattern}': {gt}，"
                f"无法按约定规则替换为生成音频路径，请检查 ORIG_SUBDIR 或 CSV。"
            )

        # 只替换第一次出现的 "/wavs/"，防止路径中其它位置误替换
        gen = gt.replace(pattern, replacement, 1)
        gen = os.path.normpath(gen)
        gen_paths.append(gen)

    return gt_paths, gen_paths, df

# ============================================================
#                       评估与统计汇总
# ============================================================

def evaluate_results(
    results: List[Dict[str, Any]],
    save_csv: bool = False,
    out_csv: str = None,
    out_txt: str = None
):
    """
    对 batch_emotion_similarity 的结果进行统计和保存。

    统计内容：
        - 总样本数
        - 相似度的 mean / median / std
        - top-1 accuracy (pred_top1 == gt_pred_top1)
        - 每个情感类别的：
            * 样本数
            * 平均相似度
            * 预测正确数
            * 类内 accuracy

    输出：
        - 控制台打印 Summary
        - 若 out_txt 不为空：将 Summary 写入指定 txt 文件
        - 若 save_csv 且 out_csv 不为空：写入包含结果的 CSV 文件

    所有数值保存时格式统一为四位小数。
    """
    total = len(results)

    # 收集所有有效的相似度值
    sims = np.array(
        [float(r["similarity"]) for r in results if r.get("similarity") is not None],
        dtype=float
    )
    mean_sim = float(sims.mean()) if sims.size > 0 else None
    median_sim = float(np.median(sims)) if sims.size > 0 else None
    std_sim = float(sims.std(ddof=0)) if sims.size > 0 else None

    top1_correct = 0  # 预测与 GT 完全一致的样本数

    # 每类别统计信息
    per_class_counts: Dict[str, int] = {}
    per_class_correct: Dict[str, int] = {}
    per_class_sim_sums: Dict[str, float] = {}

    for r in results:
        gt_label = (r.get("gt_pred_top1") or "").strip().lower()
        pred_label = (r.get("pred_top1") or "").strip().lower()

        # 将缺失标签归为 "_unknown" 类，防止 key 错误
        key = gt_label if gt_label else "_unknown"
        per_class_counts.setdefault(key, 0)
        per_class_correct.setdefault(key, 0)
        per_class_sim_sums.setdefault(key, 0.0)

        per_class_counts[key] += 1

        if r.get("similarity") is not None:
            per_class_sim_sums[key] += float(r.get("similarity"))

        # 严格匹配：只有当 canonical 标签完全一致时才算正确
        if gt_label and pred_label and gt_label == pred_label:
            top1_correct += 1
            per_class_correct[key] += 1

    accuracy = top1_correct / total if total > 0 else None

    # 汇总每个类别的统计
    per_class_stats: Dict[str, Dict[str, Any]] = {}
    for k in per_class_counts.keys():
        cnt = per_class_counts[k]
        sim_sum = per_class_sim_sums.get(k, 0.0)
        correct = per_class_correct.get(k, 0)
        mean_sim_k = (sim_sum / cnt) if cnt > 0 else None
        acc_k = (correct / cnt) if cnt > 0 else None

        per_class_stats[k] = {
            "total": cnt,
            "mean_similarity": mean_sim_k,
            "correct": correct,
            "accuracy": acc_k,
        }

    # ---------- 构造 Summary 文本 ----------
    lines = []
    lines.append("=== Summary ===")
    lines.append(f"pairs total: {total}")
    if mean_sim is not None:
        lines.append(f"mean similarity: {mean_sim:.4f}")
    else:
        lines.append("mean similarity: None")

    if median_sim is not None:
        lines.append(f"median similarity: {median_sim:.4f}")
    else:
        lines.append("median similarity: None")

    if std_sim is not None:
        lines.append(f"std similarity: {std_sim:.4f}")
    else:
        lines.append("std similarity: None")

    if accuracy is not None:
        lines.append(f"top-1 accuracy: {top1_correct}/{total} = {accuracy:.4f}")
    else:
        lines.append("top-1 accuracy: None")

    lines.append("per-class stats (label: total, mean_sim, correct, accuracy):")

    # 为了便于阅读，按标签字典序输出
    for k in sorted(per_class_stats.keys()):
        v = per_class_stats[k]
        cnt = v["total"]
        mean_sim_k = v["mean_similarity"]
        correct = v["correct"]
        acc_k = v["accuracy"]

        mean_sim_k_str = f"{mean_sim_k:.4f}" if mean_sim_k is not None else "None"
        acc_k_str = f"{acc_k:.4f}" if acc_k is not None else "None"
        lines.append(f"  {k}: {cnt}, mean_sim={mean_sim_k_str}, correct={correct}, acc={acc_k_str}")

    # 控制台打印 Summary
    for ln in lines:
        print(ln)

    # 将 Summary 写入文本文件（可选）
    if out_txt:
        try:
            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            with open(out_txt, "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
            print(f"Saved summary text to: {out_txt}")
        except Exception as e:
            print(f"Failed to save summary to {out_txt}: {e}")

    # 写最终 CSV（只写一次，列名固定）
    if save_csv and out_csv:
        fieldnames = ["gen_path", "gt_path", "similarity", "gt_pred_top1", "pred_top1"]
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "gen_path": r.get("gen_path"),
                    "gt_path": r.get("gt_path"),
                    "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.4f}",
                    "gt_pred_top1": r.get("gt_pred_top1") or "",
                    "pred_top1": r.get("pred_top1") or "",
                })
        print(f"Saved evaluation CSV to: {out_csv}")

    return {
        "mean_similarity": mean_sim,
        "median_similarity": median_sim,
        "std_similarity": std_sim,
        "top1_accuracy": accuracy,
        "per_class": per_class_stats,
    }

# ============================================================
#                           main 主流程
# ============================================================

if __name__ == "__main__":
    # 1) 加载情感模型
    print("Loading emotion model...")
    model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
    print("Model loaded.")

    # 2) 从 CSV 中读取 audio_file，并构造 GT 与生成音频路径
    print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
    gt_paths, gen_paths, df = build_paths_from_csv(
        CSV_PATH, CSV_DELIM, BASE_GT_PREFIX, GENERATE_PATH
    )

    # 打印前 5 条样例映射，方便你快速确认路径是否正确
    print("Example mapping (first 5):")
    for i in range(min(5, len(gt_paths))):
        print(f"  GT:  {gt_paths[i]}")
        print(f"  GEN: {gen_paths[i]}")

    # 3) 批量计算相似度与情感预测（GT 情感也由模型推理得到）
    results = batch_emotion_similarity(model, gen_paths, gt_paths)

    # 4) 汇总评估指标，并按配置写入 CSV 和 Summary 文本
    eval_summary = evaluate_results(
        results,
        save_csv=SAVE_RESULTS_CSV,
        out_csv=RESULTS_CSV_PATH if SAVE_RESULTS_CSV else None,
        out_txt=RESULTS_TXT_PATH,
    )







# 计算情感一致性和情感分类准确率（适用原ESD数据）
# import os
# import re
# import csv
# from typing import List, Tuple, Dict, Any

# import torch
# import numpy as np
# from torch.nn import functional as F
# import pandas as pd

# from funasr import AutoModel

# # --------------------- 配置区域 ---------------------
# # 请根据你的环境修改以下路径/参数
# MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_base"
# HUB = "ms"                           # 模型来源类型（ms/modelscope/hf 等）
# CSV_PATH = "/tmp/xtts_ft/dataset/test.csv"  # 原始目标 CSV（包含 audio_file,text,...）
# CSV_DELIM = "|"                      # CSV 分隔符（示例用了竖线）
# BASE_GT_PREFIX = "/tmp/xtts_ft/dataset/"    # 拼接 CSV 中相对路径得到真实 gt 音频路径的前缀
# GENERATE_PATH = "/mnt/hd/zjy/test/Ft_XTTS_new_dataset_no-label/"  # 生成音频所在目录（会拼接 basename）
# SAVE_RESULTS_CSV = True
# RESULTS_CSV_PATH = os.path.join(GENERATE_PATH, "emo_sim_results.csv")
# RESULTS_TXT_PATH = os.path.join(GENERATE_PATH, "emo_sim_summary.txt")  # <--- 新增：保存 Summary 文本的路径
# # ----------------------------------------------------

# # ---------- 标签映射与工具函数 ----------
# LABEL_MAP = {
#     "生气": "angry", "angry": "angry",
#     "开心": "happy", "happy": "happy",
#     "难过": "sad", "sad": "sad",
#     "吃惊": "surprise", "surprise": "surprise",
#     "中立": "neutral", "neutral": "neutral",
#     "厌恶": "disgusted", "disgusted": "disgusted",
#     "恐惧": "fearful", "fear": "fearful", "fearful": "fearful",
#     "其他": "other", "other": "other",
# }

# def _normalize_token(tok: str) -> str:
#     if not tok:
#         return ""
#     return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(tok)).strip().lower()

# def normalize_label_to_canonical(label: str) -> str:
#     if not label:
#         return ""
#     parts = [p.strip() for p in re.split(r"[\/|]", label) if p.strip()]
#     for p in parts:
#         p_norm = _normalize_token(p)
#         if p_norm in LABEL_MAP:
#             return LABEL_MAP[p_norm]
#     for p in parts:
#         p_norm = _normalize_token(p)
#         for key in LABEL_MAP:
#             if key in p_norm:
#                 return LABEL_MAP[key]
#     return _normalize_token(parts[0]) if parts else _normalize_token(label)

# def pred_top1_from_labels_and_scores(labels: List[str], scores: List[float]) -> Tuple[str, str]:
#     """
#     返回 (canonical_pred_label, raw_label_string_at_top)
#     但注意：主流程会只保留 canonical_pred_label。
#     """
#     if not labels:
#         return "", ""
#     idx = None
#     try:
#         if scores and len(scores) == len(labels):
#             scores_f = [float(s) for s in scores]
#             idx = int(np.argmax(np.array(scores_f)))
#     except Exception:
#         idx = None
#     if idx is None:
#         idx = 0
#     raw_top = str(labels[int(idx)])
#     canonical = normalize_label_to_canonical(raw_top)
#     return canonical, raw_top

# # --------------------- 模型加载与推理工具 ---------------------
# def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
#     model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
#     return model

# def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
#     res = model.generate(wav_path, output_dir="./outputs_temp", granularity="utterance", extract_embedding=True)
#     if not res or len(res) == 0:
#         raise RuntimeError(f"Model.generate returned empty for {wav_path}")
#     entry = res[0]
#     feats = entry.get("feats", None)
#     if feats is None:
#         raise RuntimeError(f"No 'feats' returned for {wav_path}")
#     labels = entry.get("labels", [])
#     scores = entry.get("scores", [])
#     # debug: 如果你想看 label/score，脚本默认会打印（保留你原先的调试打印）
#     # print(labels)
#     # print(scores)
#     return np.asarray(feats), labels, scores

# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)

# # --------------------- 批量计算（不再写中间 CSV） ---------------------
# def batch_emotion_similarity(
#     model,
#     gen_paths: List[str],
#     gt_paths: List[str],
# ) -> List[Dict[str, Any]]:
#     """
#     - 对每对 (gen, gt) 做推理
#     - 直接在这里从 model 的输出获得 GT 的 top1 canonical label（不再来自 CSV 文本）
#     - 仅保存规范化后的标签（pred_top1, gt_pred_top1）
#     - 若任意一条出现异常，立即抛出（不在结果中记录 error）
#     """
#     if len(gen_paths) != len(gt_paths):
#         raise ValueError("gen_paths and gt_paths must be the same length!")

#     results: List[Dict[str, Any]] = []
#     for i, (gpath, tpath) in enumerate(zip(gen_paths, gt_paths), start=1):
#         # 发生错误直接抛出（按你的要求）
#         if not os.path.exists(gpath):
#             raise FileNotFoundError(f"Generated audio not found: {gpath}")
#         if not os.path.exists(tpath):
#             raise FileNotFoundError(f"Target audio not found: {tpath}")

#         print(f"[{i}/{len(gen_paths)}] Processing pair:\n  gen: {gpath}\n  gt:  {tpath}")

#         # 提取 embedding, labels, scores（可能抛异常）
#         # print("生成音频：")
#         gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
#         # print("目标音频：")
#         gt_emb, gt_labels, gt_scores = _get_embedding_and_scores(model, tpath)

#         # 计算余弦相似度
#         sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#         # 从 labels/scores 中选出 canonical top1（只保留 canonical）
#         gen_canon, _ = pred_top1_from_labels_and_scores(gen_labels, gen_scores)
#         gt_canon, _ = pred_top1_from_labels_and_scores(gt_labels, gt_scores)

#         result = {
#             "gen_path": gpath,
#             "gt_path": tpath,
#             "similarity": sim,
#             "pred_top1": gen_canon,     # canonical only
#             "gt_pred_top1": gt_canon,   # canonical only
#         }
#         results.append(result)

#         print(f"  similarity = {sim:.4f}  pred_top1={gen_canon}  gt_pred_top1={gt_canon}")

#     return results

# # ----------------- CSV -> 路径构造 -----------------
# def build_paths_from_csv(csv_path: str, csv_delim: str, base_gt_prefix: str, generate_path: str):
#     """
#     从 CSV 构造 gt_paths 与 gen_paths（不再尝试从 text 解析 GT 情感）
#     返回 (gt_paths, gen_paths, df_original)
#     """
#     df = pd.read_csv(csv_path, sep=csv_delim, dtype=str, keep_default_na=False)
#     # Only require audio_file column; text is no longer required because GT emotion comes from model
#     if "audio_file" not in df.columns:
#         raise ValueError("CSV must contain column: audio_file (and optionally text).")
#     audio_files = df["audio_file"].astype(str).tolist()

#     gt_paths = []
#     gen_paths = []

#     for af in audio_files:
#         af = af.strip()
#         if os.path.isabs(af):
#             gt = af
#         else:
#             gt = os.path.join(base_gt_prefix, af)
#         gt = os.path.normpath(gt)
#         gt_paths.append(gt)

#         name = os.path.basename(af)
#         gen = os.path.join(generate_path, name)
#         gen = os.path.normpath(gen)
#         gen_paths.append(gen)

#     return gt_paths, gen_paths, df

# # ----------------- 评估统计（写入唯一最终 CSV & 保存 Summary 文本） -----------------
# def evaluate_results(results: List[Dict[str, Any]], save_csv: bool = False, out_csv: str = None, out_txt: str = None):
#     """
#     - 现在 GT 情感直接来自 results[*]['gt_pred_top1']（模型推理得到的 canonical 标签）
#     - 结果 CSV 只包含规范化标签，不包含原始 labels/scores 或 error 字段
#     - 所有打印与保存到 out_txt 的数值均保留四位小数
#     """
#     total = len(results)

#     # 统计相似度（排除 None，但在本实现中若成功则不会为 None）
#     sims = np.array([float(r["similarity"]) for r in results if r.get("similarity") is not None], dtype=float)
#     mean_sim = float(sims.mean()) if sims.size > 0 else None
#     median_sim = float(np.median(sims)) if sims.size > 0 else None
#     std_sim = float(sims.std(ddof=0)) if sims.size > 0 else None

#     top1_correct = 0
#     per_class_counts: Dict[str, int] = {}
#     per_class_correct: Dict[str, int] = {}
#     per_class_sim_sums: Dict[str, float] = {}

#     for r in results:
#         gt_label = (r.get("gt_pred_top1") or "").strip().lower()
#         pred_label = (r.get("pred_top1") or "").strip().lower()
#         key = gt_label if gt_label else "_unknown"
#         per_class_counts.setdefault(key, 0)
#         per_class_correct.setdefault(key, 0)
#         per_class_sim_sums.setdefault(key, 0.0)

#         per_class_counts[key] += 1
#         if r.get("similarity") is not None:
#             per_class_sim_sums[key] += float(r.get("similarity"))

#         # 严格相等判断
#         if gt_label and pred_label and gt_label == pred_label:
#             top1_correct += 1
#             per_class_correct[key] += 1

#     accuracy = top1_correct / total if total > 0 else None

#     per_class_stats: Dict[str, Dict[str, Any]] = {}
#     for k in per_class_counts.keys():
#         cnt = per_class_counts[k]
#         sim_sum = per_class_sim_sums.get(k, 0.0)
#         correct = per_class_correct.get(k, 0)
#         mean_sim_k = (sim_sum / cnt) if cnt > 0 else None
#         acc_k = (correct / cnt) if cnt > 0 else None
#         per_class_stats[k] = {"total": cnt, "mean_similarity": mean_sim_k, "correct": correct, "accuracy": acc_k}

#     # --- 构造 Summary 文本（所有数值格式化为四位小数）
#     lines = []
#     lines.append("=== Summary ===")
#     lines.append(f"pairs total: {total}")
#     if mean_sim is not None:
#         lines.append(f"mean similarity: {mean_sim:.4f}")
#     else:
#         lines.append("mean similarity: None")
#     if median_sim is not None:
#         lines.append(f"median similarity: {median_sim:.4f}")
#     else:
#         lines.append("median similarity: None")
#     if std_sim is not None:
#         lines.append(f"std similarity: {std_sim:.4f}")
#     else:
#         lines.append("std similarity: None")
#     if accuracy is not None:
#         # also show fraction like 29/30
#         lines.append(f"top-1 accuracy: {top1_correct}/{total} = {accuracy:.4f}")
#     else:
#         lines.append("top-1 accuracy: None")

#     lines.append("per-class stats (label: total, mean_sim, correct, accuracy):")
#     # 按字典顺序输出，保持可预测顺序
#     for k in sorted(per_class_stats.keys()):
#         v = per_class_stats[k]
#         cnt = v["total"]
#         mean_sim_k = v["mean_similarity"]
#         correct = v["correct"]
#         acc_k = v["accuracy"]
#         mean_sim_k_str = f"{mean_sim_k:.4f}" if mean_sim_k is not None else "None"
#         acc_k_str = f"{acc_k:.4f}" if acc_k is not None else "None"
#         lines.append(f"  {k}: {cnt}, mean_sim={mean_sim_k_str}, correct={correct}, acc={acc_k_str}")

#     # 打印 Summary（控制台）
#     for ln in lines:
#         print(ln)

#     # 将 Summary 写入文件（如果指定了 out_txt）
#     if out_txt:
#         try:
#             os.makedirs(os.path.dirname(out_txt), exist_ok=True)
#             with open(out_txt, "w", encoding="utf-8") as f:
#                 for ln in lines:
#                     f.write(ln + "\n")
#             print(f"Saved summary text to: {out_txt}")
#         except Exception as e:
#             print(f"Failed to save summary to {out_txt}: {e}")

#     # 写最终 CSV：只写一次，确保列名固定
#     if save_csv and out_csv:
#         fieldnames = ["gen_path", "gt_path", "similarity", "gt_pred_top1", "pred_top1"]
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in results:
#                 writer.writerow({
#                     "gen_path": r.get("gen_path"),
#                     "gt_path": r.get("gt_path"),
#                     "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.4f}",
#                     "gt_pred_top1": r.get("gt_pred_top1") or "",
#                     "pred_top1": r.get("pred_top1") or "",
#                 })
#         print(f"Saved evaluation CSV to: {out_csv}")

#     return {
#         "mean_similarity": mean_sim,
#         "median_similarity": median_sim,
#         "std_similarity": std_sim,
#         "top1_accuracy": accuracy,
#         "per_class": per_class_stats,
#     }

# # --------------------------- main ---------------------------
# if __name__ == "__main__":
#     print("Loading emotion model...")
#     model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
#     print("Model loaded.")

#     print(f"Reading CSV: {CSV_PATH} (delim='{CSV_DELIM}')")
#     gt_paths, gen_paths, df = build_paths_from_csv(CSV_PATH, CSV_DELIM, BASE_GT_PREFIX, GENERATE_PATH)

#     print("Example mapping (first 5):")
#     for i in range(min(5, len(gt_paths))):
#         print(f"  GT: {gt_paths[i]}")
#         print(f"  GEN:{gen_paths[i]}")

#     # 1) 批量计算相似度与预测（GT 情感由模型输出决定）
#     results = batch_emotion_similarity(model, gen_paths, gt_paths)

#     # 2) 汇总并写最终 CSV（如配置）。任何单条推理失败都会抛出异常并停止（按你的要求）
#     eval_summary = evaluate_results(results, save_csv=SAVE_RESULTS_CSV, out_csv=RESULTS_CSV_PATH if SAVE_RESULTS_CSV else None, out_txt=RESULTS_TXT_PATH)







# 批量计算情感相似度并进行情感分类
"""
Batch emotion-similarity calculator.

用法示例已在 bottom 的 `if __name__ == "__main__":` 中给出。

要求：
- funasr.AutoModel 可用（你的情感识别模型路径/名称保留）
- torch, numpy, pandas (若启用保存CSV)
"""

# import os
# import csv
# from typing import List, Tuple, Dict, Any

# import torch
# import numpy as np
# from torch.nn import functional as F

# from funasr import AutoModel

# # --------------------- 配置区域 ---------------------
# # 这里填你要用的情感识别模型（可以是 HF 名称、ModelScope 名称或本地目录）
# # 示例（你原先的路径）：
# MODEL_NAME_OR_PATH = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_base"
# HUB = "ms"  # "ms" or "modelscope" 或 "hf"/"huggingface"，按你环境选择
# # ----------------------------------------------------

# def load_emotion_model(model_name_or_path: str = MODEL_NAME_OR_PATH, hub: str = HUB):
#     """
#     加载情感识别模型（AutoModel）。
#     返回已经加载好的 model 对象。
#     """
#     model = AutoModel(model=model_name_or_path, hub=hub, disable_update=True)
#     return model


# def _get_embedding_and_scores(model, wav_path: str) -> Tuple[np.ndarray, List[str], List[float]]:
#     """
#     对单个音频文件调用 model.generate(..., extract_embedding=True)
#     返回：embedding (numpy array), labels (list), scores (list)
#     注：model.generate 返回的是一个列表（分片级别），通常取第0项。
#     """
#     # generate 返回类似 [{'feats': ..., 'labels': [...], 'scores': [...]}]
#     res = model.generate(wav_path, output_dir="./outputs_temp", granularity="utterance", extract_embedding=True)
#     if not res or len(res) == 0:
#         raise RuntimeError(f"Model.generate returned empty for {wav_path}")
#     entry = res[0]  # 取第一个 utterance 的结果

#     # entry['feats'] 可能是 numpy 数组
#     feats = entry.get("feats", None)
#     if feats is None:
#         raise RuntimeError(f"No 'feats' returned for {wav_path}")

#     labels = entry.get("labels", [])
#     scores = entry.get("scores", [])
#     return np.asarray(feats), labels, scores


# def _cosine_similarity_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     计算两个 1-D numpy 向量的余弦相似度，返回标量（Python float）
#     """
#     a_t = torch.from_numpy(a).float()
#     b_t = torch.from_numpy(b).float()
#     # 归一化并点乘
#     a_n = F.normalize(a_t, dim=-1)
#     b_n = F.normalize(b_t, dim=-1)
#     cos = (a_n * b_n).sum().item()
#     return float(cos)


# def batch_emotion_similarity(
#     model,
#     gen_paths: List[str],
#     gt_paths: List[str],
#     save_csv: bool = False,
#     csv_path: str = "emotion_similarity_results.csv",
# ) -> List[Dict[str, Any]]:
#     """
#     批量计算情感相似度。
#     输入：
#       - model: 已加载的情感识别 AutoModel
#       - gen_paths: 生成音频路径列表
#       - gt_paths: 目标/参考音频路径列表（长度必须与 gen_paths 相同）
#       - save_csv: 是否把结果写入 csv
#       - csv_path: csv 保存路径（若 save_csv=True）

#     输出：返回一个字典列表，每项包含：
#       {
#         "gen_path": str,
#         "gt_path": str,
#         "similarity": float,
#         "gen_labels": [...],   # emotion labels by the model for generated audio
#         "gen_scores": [...],   # corresponding scores
#         "gt_labels": [...],
#         "gt_scores": [...],
#       }
#     """
#     if len(gen_paths) != len(gt_paths):
#         raise ValueError("gen_paths and gt_paths must be the same length!")

#     results = []

#     for i, (gpath, tpath) in enumerate(zip(gen_paths, gt_paths), start=1):
#         try:
#             if not os.path.exists(gpath):
#                 raise FileNotFoundError(f"Generated audio not found: {gpath}")
#             if not os.path.exists(tpath):
#                 raise FileNotFoundError(f"Target audio not found: {tpath}")

#             print(f"[{i}/{len(gen_paths)}] Processing pair:\n  gen: {gpath}\n  gt:  {tpath}")

#             # 对每个音频分别调用 model.generate，提取 embedding + labels + scores
#             gen_emb, gen_labels, gen_scores = _get_embedding_and_scores(model, gpath)
#             gt_emb, gt_labels, gt_scores = _get_embedding_and_scores(model, tpath)

#             # 计算余弦相似度
#             sim = _cosine_similarity_from_numpy(gen_emb, gt_emb)

#             result = {
#                 "gen_path": gpath,
#                 "gt_path": tpath,
#                 "similarity": sim,
#                 "gen_labels": gen_labels,
#                 "gen_scores": gen_scores,
#                 "gt_labels": gt_labels,
#                 "gt_scores": gt_scores,
#             }
#             results.append(result)

#             print(f"  similarity = {sim:.4f}")
#             # 可选: 打印 top label/scores 简短摘要
#             if gen_labels and gen_scores:
#                 print(f"  gen top: {gen_labels} / {gen_scores}")
#             if gt_labels and gt_scores:
#                 print(f"  gt  top: {gt_labels} / {gt_scores}")

#         except Exception as e:
#             # 捕获单对异常并记录，继续处理下一对
#             print(f"  ERROR processing pair idx={i}: {e}")
#             results.append(
#                 {
#                     "gen_path": gpath,
#                     "gt_path": tpath,
#                     "similarity": None,
#                     "gen_labels": None,
#                     "gen_scores": None,
#                     "gt_labels": None,
#                     "gt_scores": None,
#                     "error": str(e),
#                 }
#             )

#     # 可选：把结果保存为 CSV（扁平化 labels/scores 为字符串）
#     if save_csv:
#         fieldnames = [
#             "gen_path",
#             "gt_path",
#             "similarity",
#             "gen_labels",
#             "gen_scores",
#             "gt_labels",
#             "gt_scores",
#             "error",
#         ]
#         with open(csv_path, "w", newline="", encoding="utf-8") as cf:
#             writer = csv.DictWriter(cf, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in results:
#                 row = {
#                     "gen_path": r.get("gen_path"),
#                     "gt_path": r.get("gt_path"),
#                     "similarity": "" if r.get("similarity") is None else f"{r.get('similarity'):.6f}",
#                     "gen_labels": ";".join(map(str, r.get("gen_labels") or [])),
#                     "gen_scores": ";".join(map(str, r.get("gen_scores") or [])),
#                     "gt_labels": ";".join(map(str, r.get("gt_labels") or [])),
#                     "gt_scores": ";".join(map(str, r.get("gt_scores") or [])),
#                     "error": r.get("error", ""),
#                 }
#                 writer.writerow(row)
#         print(f"Saved results to CSV: {csv_path}")

#     return results


# # ------------------- 示例主程序 -------------------
# if __name__ == "__main__":
#     # 示例：你会提供两列表，分别为生成音频路径和对应的目标音频路径
#     # no_ecl-loss   w_ecl_0.5_lr_5e-5   w_ecl_1.0_lr_5e-5  w_ecl_1.5_lr_5e-5
#     # w_ecl_0.5_lr_1e-4_5e-5    w_ecl_0.5_lr_7e-5_2e-4    w_ecl_0.5_lr_7e-5_5e-4
#     # w_ecl_0.5_lr_7e-5_3e-4  w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10
#     # w_ecl_0.5_gpt-last-two-layer_lr_2e-6_1-28    w_ecl_0.5_gpt-last-layer_lr_1e-6_1-46
#     # w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10_en   no_ecl-loss_en
#     # XTTS_v2.0_original_model_files_original_en  
#     # 5_emo_labels_zh_new
    
#     # 英文
#     # 生成音频路径
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0011_000349/surprise.wav",
#     ] """
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0014_000133/surprise.wav",
#     ] """
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0015_000306/surprise.wav",
#     ] """
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/XTTS_v2.0_original_model_files_original_en/0018_000071/surprise.wav",
#     ] """

#     # 目标音频路径
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_000349.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_000699.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001049.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001399.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0011/0011_001749.wav",
#     ] """
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000133.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000483.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_000833.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_001183.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0014/0014_001533.wav",
#     ] """
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_000306.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_000656.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001006.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001356.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0015/0015_001706.wav",
#     ] """
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000071.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000421.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_000771.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_001121.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0018/0018_001471.wav",
#     ] """

#     # 中文
#     # 生成音频路径
#     gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10/0001_000159/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10/0001_000159/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10/0001_000159/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10/0001_000159/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/w_ecl_0.5_gpt-last-layer_lr_2e-6_1-10/0001_000159/surprise.wav",
#     ]
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0004_000047/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0004_000047/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0004_000047/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0004_000047/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0004_000047/surprise.wav",
#     ] """
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0006_000009/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0006_000009/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0006_000009/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0006_000009/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0006_000009/surprise.wav",
#     ] """
#     """ gen_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0009_000002/neutral.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0009_000002/angry.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0009_000002/happy.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0009_000002/sad.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/5_emo_labels_zh_new/0009_000002/surprise.wav",
#     ] """
    
#     # 目标音频路径
#     gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000159.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000509.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_000859.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_001209.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0001/0001_001559.wav",
#     ]
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000047.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000397.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_000747.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_001097.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0004/0004_001447.wav",
#     ] """
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000009.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000359.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_000709.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_001059.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0006/0006_001409.wav",
#     ] """
#     """ gt_paths_example = [
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000002.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000352.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_000702.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_001052.wav",
#         "/mnt/hd/zjy/CoquiTTS/test_emo/0009/0009_001402.wav",
#     ] """

#     # 1) 加载模型（只需一次）
#     print("Loading emotion model...")
#     model = load_emotion_model(MODEL_NAME_OR_PATH, HUB)
#     print("Model loaded.")

#     # 2) 调用批量相似度计算
#     results = batch_emotion_similarity(model, gen_paths_example, gt_paths_example, save_csv=False, csv_path="emo_sim_results.csv")

#     # 3) 简要打印所有结果
#     print("\nAll results:")
#     for r in results:
#         print(f"{os.path.basename(r['gen_path'])} vs {os.path.basename(r['gt_path'])} -> sim={r['similarity']}")
