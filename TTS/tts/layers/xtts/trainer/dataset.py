import os
import random
import sys

import torch
import torch.nn.functional as F
import torch.utils.data

import traceback

from TTS.tts.models.xtts import load_audio

torch.set_num_threads(1)


def key_samples_by_col(samples, col):
    """Returns a dictionary of samples keyed by language."""
    samples_by_col = {}
    for sample in samples:
        col_val = sample[col]
        assert isinstance(col_val, str)
        if col_val not in samples_by_col:
            samples_by_col[col_val] = []
        samples_by_col[col_val].append(sample)
    return samples_by_col


def get_prompt_slice(gt_path, max_sample_length, min_sample_length, sample_rate, is_eval=False):
    rel_clip = load_audio(gt_path, sample_rate)
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
    cond_idxs = [rand_start, rand_end]
    return rel_clip, rel_clip.shape[-1], cond_idxs

class XTTSDataset(torch.utils.data.Dataset):
    def __init__(self, config, samples, tokenizer, sample_rate, is_eval=False):
        self.config = config
        model_args = config.model_args
        self.failed_samples = set()
        self.debug_failures = model_args.debug_loading_failures
        self.max_conditioning_length = model_args.max_conditioning_length
        self.min_conditioning_length = model_args.min_conditioning_length
        self.is_eval = is_eval
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_wav_len = model_args.max_wav_length
        self.max_text_len = model_args.max_text_length
        self.use_masking_gt_prompt_approach = model_args.gpt_use_masking_gt_prompt_approach
        assert self.max_wav_len is not None and self.max_text_len is not None

        self.samples = samples
        if not is_eval:
            random.seed(config.training_seed)
            # random.shuffle(self.samples)
            random.shuffle(self.samples)
            # order by language
            self.samples = key_samples_by_col(self.samples, "language")
            print(" > Sampling by language:", self.samples.keys())
        else:
            # for evaluation load and check samples that are corrupted to ensures the reproducibility
            self.check_eval_samples()

    def check_eval_samples(self):
        print(" > Filtering invalid eval samples!!")
        new_samples = []
        for sample in self.samples:
            try:
                tseq, _, wav, _, _, _, _ = self.load_item(sample)
                # print(f"tseq:{tseq}")
                # print(f"wav:{wav}")
            except Exception as e:
                print("详细追踪信息:")
                traceback.print_exc()  # 这会打印完整的调用栈信息
                continue
            # except:
            #     continue

            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            if wav is None:
                print("wav is None")
                continue
            if self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len:
                print("self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len")
                continue
            if self.max_text_len is not None and tseq.shape[0] > self.max_text_len:
                print("self.max_text_len is not None and tseq.shape[0] > self.max_text_len")
                continue
            # if (
            #     wav is None
            #     or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
            #     or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
            # ):
            #     continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def get_text(self, text, lang):
        # print(f"BPE分词前为：{text}")
        tokens = self.tokenizer.encode(text, lang)
        # print(f"BPE分词后的结果为：{tokens}")
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def load_item(self, sample):
        text = str(sample["text"])
        tseq = self.get_text(text, sample["language"])
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if text is None or len(text.strip()) == 0:
            raise ValueError
        if wav is None or wav.shape[-1] < (0.5 * self.sample_rate):
            # Ultra short clips are also useless (and can cause problems within some models).
            raise ValueError

        ref_audiopath = audiopath

        if self.use_masking_gt_prompt_approach:
            # # -------（最初替换逻辑）替换输给Perceiver conditioner的情感音频为对应的中性音频---------
            # # 定义标签与偏移的映射关系
            # offset_map = {
            #     6682: 350,
            #     6683: 700,
            #     6684: 1050,
            #     6685: 1400
            # }
            # # 遍历所有可能的标签，检查是否存在于 tseq 中
            # offset = None
            # for token_id, off in offset_map.items():
            #     if (tseq == token_id).any():
            #         offset = off
            #         break  # 一旦找到匹配的标签，就退出循环

            # # 如果匹配到了 offset，则执行对应操作
            # if offset is not None:   ##把情感音频路径替换成对应的中性音频路径
            #     # print(f"替换前：{audiopath}")
            #     # 将 /tmp/.../wavs/0008_000525.wav 拆分成目录 + 文件名
            #     dirpath, filename = os.path.split(audiopath)
            #     name, ext = os.path.splitext(filename)  # name = "0008_000525", ext = ".wav"
            #     # 再把 name 拆成两段
            #     prefix, idx_str = name.split("_")      # prefix = "0008", idx_str = "000525"
            #     width = len(idx_str)                   # 6
            #     # 做减法，并还原成相同宽度的字符串
            #     new_idx = int(idx_str) - offset        # 525 - 350 = 175
            #     if new_idx < 0:
            #         raise ValueError(f"索引减去 {offset} 后变为负数：{new_idx}")
            #     new_idx_str = str(new_idx).zfill(width)  # "000175"
            #     # 拼回新的文件名和路径
            #     new_filename = f"{prefix}_{new_idx_str}{ext}"
            #     # 拼成新的文件路径
            #     new_audiopath = os.path.join(dirpath, new_filename)
            #     # print(f"替换后：{new_audiopath}")

            #     ref_audiopath = new_audiopath

            #     # get a slice from GT to condition the model
            #     cond, _, cond_idxs = get_prompt_slice(
            #         new_audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            #     )
            # else:
            #     # print(f"没替换：{audiopath}")
            #     # get a slice from GT to condition the model
            #     cond, _, cond_idxs = get_prompt_slice(
            #         audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            #     )
            # # -------------------------------------


            # ---------- 新版：根据路径上的情绪目录替换为 Neutral ----------
            # 需要替换为 Neutral 的情绪子目录名
            emotion_dirs = ["Surprise", "Angry", "Happy", "Sad"]

            # 遍历检查是否包含这些情绪目录
            for emo in emotion_dirs:
                # 用 os.sep 保证在 Linux 和 Windows 都能正常工作
                marker = os.sep + emo + os.sep
                if marker in audiopath:
                    # 只替换一次该情绪目录为 Neutral
                    neutral_marker = os.sep + "Neutral" + os.sep
                    ref_audiopath = audiopath.replace(marker, neutral_marker, 1)
                    break  # 找到一个情绪目录就够了，退出循环

            # 如果原来路径里就是 Neutral，ref_audiopath 会保持为 audiopath，不做任何修改

            # 如果你想确保中性音频一定存在，可以加一个存在性检查（可选）：
            # if not os.path.exists(ref_audiopath):
            #     raise ValueError(f"中性音频不存在: {ref_audiopath}")

            # get a slice from GT to condition the model
            cond, _, cond_idxs = get_prompt_slice(
                ref_audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            ) 

            """ # get a slice from GT to condition the model
            cond, _, cond_idxs = get_prompt_slice(
                audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            ) """
            
            # if use masking do not use cond_len
            cond_len = torch.nan
        else:
            ref_sample = (
                sample["reference_path"]
                if "reference_path" in sample and sample["reference_path"] is not None
                else audiopath
            )
            cond, cond_len, _ = get_prompt_slice(
                ref_sample, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
            )
            # if do not use masking use cond_len
            cond_idxs = torch.nan

        # sample["ref_audiopath"] = ref_audiopath

        return tseq, audiopath, wav, cond, cond_len, cond_idxs,ref_audiopath

    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
            sample_id = str(index)
        else:
            # select a random language
            lang = random.choice(list(self.samples.keys()))
            # select random sample
            index = random.randint(0, len(self.samples[lang]) - 1)
            sample = self.samples[lang][index]
            # a unique id for each sampel to deal with fails
            sample_id = lang + "_" + str(index)

        # ignore samples that we already know that is not valid ones
        if sample_id in self.failed_samples:
            if self.debug_failures:
                print(f"Ignoring sample {sample['audio_file']} because it was already ignored before !!")
            # call get item again to get other sample
            return self[1]

        # try to load the sample, if fails added it to the failed samples list
        try:
            tseq, audiopath, wav, cond, cond_len, cond_idxs,ref_audiopath = self.load_item(sample)
        except:
            if self.debug_failures:
                print(f"error loading {sample['audio_file']} {sys.exc_info()}")
            self.failed_samples.add(sample_id)
            return self[1]

        # check if the audio and text size limits and if it out of the limits, added it failed_samples
        if (
            wav is None
            or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
            or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
        ):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            if self.debug_failures and wav is not None and tseq is not None:
                print(
                    f"error loading {sample['audio_file']}: ranges are out of bounds; {wav.shape[-1]}, {tseq.shape[0]}"
                )
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            # 'real_text': text,
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long)
            if cond_len is not torch.nan
            else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
            "ref_audiopath":ref_audiopath,
        }
        return res

    def __len__(self):
        if self.is_eval:
            return len(self.samples)
        return sum([len(v) for v in self.samples.values()])

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch) 

        #处理之前是一个样本在一个集合中，处理之后是把相同字段的内容放到一个集合中
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape  把Tensor都提取到最外面
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["cond_idxs"] = torch.stack(batch["cond_idxs"])

        if torch.any(batch["cond_idxs"].isnan()):
            batch["cond_idxs"] = None

        if torch.any(batch["cond_lens"].isnan()):#把cond_lens:tensor([[nan],[nan],[nan],[nan]])设置为'cond_lens': None
            batch["cond_lens"] = None

        max_text_len = batch["text_lengths"].max()#取'text_lengths': tensor([16, 23, 13,  8])中的最大值，如tensor(23)
        max_wav_len = batch["wav_lengths"].max()#同上

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded #把batch["wav"] 替换成 wav_padded
        batch["padded_text"] = text_padded #batch["padded_text"] 设为 text_padded
        return batch
