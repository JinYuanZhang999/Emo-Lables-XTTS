from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import os
import librosa
import time

from torch.nn.utils.rnn import pad_sequence
from funasr import AutoModel
import numpy as np
import math, random

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader,Sampler,Dataset
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec
from TTS.tts.models.xtts import load_audio

# 用于保证每个batch包含5个情感类别（在get_data_loader方法中会被用到）
EMOTION_TOKEN_IDS = [6681, 6682, 6683, 6684, 6685]
E = len(EMOTION_TOKEN_IDS)

@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    test_sentences: List[dict] = field(default_factory=lambda: [])


@dataclass
class XttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050


@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    tokenizer_file: str = ""
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # overide vocoder key on the config to avoid json write issues


def callback_clearml_load_save(operation_type, model_info):
    # return None means skip the file upload/log, returning model_info will continue with the log/upload
    # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
    assert operation_type in ("load", "save")
    # print(operation_type, model_info.__dict__)

    if "similarities.pth" in model_info.__dict__["local_model_path"]:
        return None

    return model_info

# ------------------ Dataset Wrapper（在get_data_loader方法中会用到） ------------------
class IndexedXTTSDataset(Dataset):
    """把原始 dataset 包装成可以通过索引访问的 Dataset"""
    def __init__(self, base_dataset, flat_samples):
        self.base = base_dataset
        self.flat = flat_samples

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, idx):
        sample = self.flat[idx]
        tseq, audiopath, wav, cond, cond_len, cond_idxs, ref_audiopath = self.base.load_item(sample)
        return {
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long) if cond_len is not torch.nan else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
            "ref_audiopath": ref_audiopath,
        }

# ------------------ Sampler（在get_data_loader方法中会用到） ------------------
class EmotionBalancedBatchSampler(Sampler):
    """每个 epoch 在 __iter__ 里随机打乱每个桶，并产出包含所有情感的 batch"""
    def __init__(self, buckets_dict, batch_size, num_batches=None):
        assert isinstance(buckets_dict, dict)
        assert batch_size == len(buckets_dict), "batch_size must equal number of emotions"
        self.buckets = {k: list(v) for k, v in buckets_dict.items()}
        self.tids = list(self.buckets.keys())
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        # 每次 epoch 调用时重新 shuffle
        for tid in self.tids:
            random.shuffle(self.buckets[tid])
        total = sum(len(v) for v in self.buckets.values())
        E_local = len(self.tids)
        num_batches = self.num_batches or max(1, math.ceil(total / float(E_local)))
        ptrs = {tid: 0 for tid in self.tids}
        for _ in range(num_batches):
            batch = []
            for tid in self.tids:
                lst = self.buckets[tid]
                p = ptrs[tid]
                if p < len(lst):
                    batch.append(lst[p])
                    ptrs[tid] += 1
                else:
                    batch.append(random.choice(lst))  # 不够就补
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.buckets.values())
        E_local = len(self.tids)
        return max(1, math.ceil(total / float(E_local)))
# ------------------------------end----------------------------------------

class GPTTrainer(BaseTTS):
    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.xtts = Xtts(self.config)
        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()

        # 调用funasr
        emo_model_path = "/mnt/hd/zjy/.cache/modelscope/hub/models/iic/emotion2vec_plus_base"
        self.emo_auto = AutoModel(
            model=emo_model_path,
            hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
            disable_update=True
        )
        self.emotion_model = self.emo_auto.model.to(self.device)   # self.device e.g. "cuda:0"
        self.emotion_model.eval()

        # ------------------ 在这里初始化 speaker-embedding 缓存 ------------------
        # 简单字典（小项目 / 单机训练可用）
        self._spk_emb_cache = {}  # path -> torch.Tensor (存放 CPU 上)

        # ------------------ 在这里初始化 emotion-embedding 缓存 ------------------
        self._emotion_emb_cache = {}  # path -> torch.Tensor (CPU, detached, shape (C,))


        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

        # set mel stats
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            print("是否存在gpt_checkpoint")
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device("cpu"))
            # deal with coqui Trainer exported model
            if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
                print("Coqui Trainer checkpoint detected! Converting it!")
                gpt_checkpoint = gpt_checkpoint["model"]
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if "gpt." in key:
                        new_key = key.replace("gpt.", "")
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                "text_embedding.weight" in gpt_checkpoint
                and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                    self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print(">> GPT weights restored from:", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        losses = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )
        return losses

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.config,
                    s_info["speaker_wav"],
                    s_info["language"],
                    gpt_cond_len=3,
                )["wav"]
                test_audios["{}-audio".format(idx)] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]
        
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # print("cond_mels 形状:", paired_conditioning_mel.shape)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)
        # print("audio_codes 形状:", codes.shape)

        batch["audio_codes"] = codes
        # delete useless batch tensors
        del batch["padded_text"]
        del batch["wav"]
        del batch["conditioning"]
        return batch

    def compute_speaker_embeddings_from_paths(self, ref_paths, load_sr=16000, max_ref_length=30, sound_norm_refs=False, librosa_trim_db=None):
        """
        输入:
        ref_paths: list[str]，len = batch_size
        load_sr: 说话人编码器所需采样率为16000。而推理时，是先以22050采样率加载音频，然后重采样至16000输给说话人编码器，因为计算条件向量需要22050采样率。
        sound_norm_refs, librosa_trim_db: 与推理保持一致以保证相同 embedding

        返回:
        Tensor (B, C, 1) 在 self.device 上
        """
        device = self.device
        emb_list = [None] * len(ref_paths)
        to_compute = []

        # 1) 查缓存
        for i, path in enumerate(ref_paths):
            emb = self._spk_emb_cache.get(path, None)
            if emb is None:
                to_compute.append((i, path))
            else:
                emb_list[i] = emb.to(device)  # 已是 tensor，期望放在 device

        # 2) 逐条计算未缓存项（稳健简单做法）
        if len(to_compute) > 0:
            # 确保 speaker encoder 在 eval 且不需要梯度
            self.xtts.hifigan_decoder.speaker_encoder.eval()
            for idx, path in to_compute:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"ref audio not found: {path}")
                # load_audio 与推理保持一致（返回 Tensor, shape (1, N)）
                audio = load_audio(path, load_sr)  
                if audio is None:
                    raise RuntimeError(f"Unable to load audio: {path}")
                # 截取 & normalize（和推理一致）
                audio = audio[:, : load_sr * max_ref_length].to(device)
                if sound_norm_refs:
                    audio = (audio / torch.abs(audio).max()) * 0.75
                if librosa_trim_db is not None:
                    audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

                # speaker encoder 不需要参数梯度 -> 用 no_grad()
                spk_emb = self.xtts.get_speaker_embedding(audio, load_sr)  # 返回 (C,1) 已 to(self.device) in your impl
                spk_emb = spk_emb.to(device)

                # ----- 最小补丁：去掉前导 batch dim（如果存在） -----
                if spk_emb.dim() == 3 and spk_emb.shape[0] == 1:
                    spk_emb = spk_emb.squeeze(0)  # -> (C,1)
                if spk_emb.dim() == 1:
                    spk_emb = spk_emb.unsqueeze(-1)
                # ----------------------------------------------------

                # cache and assign
                self._spk_emb_cache[path] = spk_emb.detach().cpu()
                emb_list[idx] = spk_emb
        
        # 3) pack 成 (B, C, 1)
        # 每个 emb_list[i] 应为 (C,1)
        assert all(e is not None for e in emb_list), "Some speaker embeddings missing!"
        emb_stack = torch.stack([e.squeeze(-1) for e in emb_list], dim=0)  # (B, C)
        emb_stack = emb_stack.unsqueeze(-1)  # (B, C, 1)
        return emb_stack

    def prepare_batch_for_emotion2vec(
            self, 
            wav_batch,
            device=None,
            sr_in=None,
            sr_target=16000,
            require_grad_inputs=False,
            reduce_channels=True,
            apply_cfg_normalize=False,
        ):
            """
            将一个 batch 的 waveform 规范化为 Emotion2vec 期待的输入格式（唯一负责输入规范化，供后续批量 forward 使用）。
            返回：
            - padded: Tensor [B, T] float32，在 device 上
            - lengths: LongTensor [B] 每条样本的真实长度
            - wave_list: list of 1D Tensor（每条样本），保留 requires_grad（如果需要）
            - padding_mask_wave: BoolTensor [B, T]（waveform-level，True 表示该位置为 padding）
            说明：此函数对应源代码中 load_audio_text_image_video + inference 前对 source 的处理步骤（device、normalize、view(1,-1) 等）。
            """

            # 确定 device
            if device is None:
                first = None
                if isinstance(wav_batch, torch.Tensor):
                    first = wav_batch
                elif isinstance(wav_batch, (list, tuple)) and len(wav_batch) > 0:
                    first = wav_batch[0]
                device = first.device if (isinstance(first, torch.Tensor) and first.device is not None) else (
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

            # 内部函数：把单个 item 转成 1D torch.Tensor 并放到 device（做通道合并、可选重采样、设置 requires_grad）
            def _norm_single(item):
                if isinstance(item, (list, tuple)):
                    item = torch.tensor(item, dtype=torch.float32)
                if isinstance(item, (int, float)):
                    item = torch.tensor([item], dtype=torch.float32)
                if isinstance(item, torch.Tensor):
                    tw = item
                else:
                    tw = torch.tensor(item, dtype=torch.float32)

                # 多通道处理：对应 load_audio_text_image_video 中 reduce_channels 行为
                if tw.dim() == 3 and tw.size(0) == 1 and tw.size(1) == 1:
                    tw = tw.squeeze(0).squeeze(0)
                elif tw.dim() == 2 and (tw.size(0) == 1 or tw.size(1) == 1):
                    if tw.size(0) == 1:
                        tw = tw.squeeze(0)
                    else:
                        if reduce_channels:
                            tw = tw.mean(0)
                        else:
                            tw = tw.view(-1)
                elif tw.dim() > 2:
                    tw = tw.view(-1)

                tw = tw.to(torch.float32)

                # 如需重采样（对应源码末尾 audio_fs != fs 的行为）
                if (sr_in is not None) and (sr_in != sr_target):
                    # 在当前 tw 所在 device（可能是 cuda）上做重采样，避免 detach
                    # 注意：torchaudio Resample 支持 GPU，如果你的 torchaudio 版本支持并编译了相应后端
                    resampler = torchaudio.transforms.Resample(sr_in, sr_target).to(tw.device)
                    # tw 可能是一维 [T]，Resample 期望 [1, T]
                    tw = resampler(tw.unsqueeze(0))[0]  # 仍然保留原始的计算图（不会 detach）
                else:
                    tw = tw.to(device)

                # 是否设置输入可微（用于让 loss.backward 回到 HiFiGAN 输出）
                if require_grad_inputs and not tw.requires_grad:
                    tw.requires_grad_(True)

                return tw

            # 统一把 wav_batch 规范成 list of tensors
            if isinstance(wav_batch, torch.Tensor):
                if wav_batch.dim() >= 2 and wav_batch.size(0) > 1:
                    t = wav_batch
                    if t.dim() == 3 and t.size(1) == 1:
                        t = t.squeeze(1)  # [B, T]
                    waves = [t[i] for i in range(t.size(0))]
                else:
                    waves = [wav_batch]
            elif isinstance(wav_batch, (list, tuple)):
                waves = list(wav_batch)
            else:
                raise ValueError("wav_batch 必须是 tensor 或包含 tensor 的列表")

            # 对每条样本做规范化
            wave_list = [_norm_single(w) for w in waves]

            # 如果要与源码中 per-sample normalize 对齐，按 time 维做 layer_norm（可传 model.cfg.get("normalize", False)）
            if apply_cfg_normalize:
                wave_list = [F.layer_norm(w, (w.shape[0],)) for w in wave_list]

            # lengths 与 pad
            lengths = torch.tensor([w.shape[-1] for w in wave_list], dtype=torch.long, device=device)
            padded = pad_sequence(wave_list, batch_first=True)  # [B, max_T]

            # waveform-level padding mask（True 表示 padded）
            max_T = padded.size(1)
            idx = torch.arange(max_T, device=device).unsqueeze(0)
            padding_mask_wave = idx >= lengths.unsqueeze(1)  # [B, T]

            # 设备一致性检查
            if padded.device != device:
                padded = padded.to(device)
            if padding_mask_wave.device != device:
                padding_mask_wave = padding_mask_wave.to(device)

            return padded, lengths, wave_list, padding_mask_wave


    # 严格对照源码，每条样本单独 forward，传入 extract_features
    def extract_emotion_feats_only_batch(
            self, 
            model,
            wav_batch,
            device=None,
            granularity="utterance",
            extract_embedding=True,
            require_grad=False,
            output_dir=None,
            sr_in=None,
            sr_target=16000,
            require_grad_inputs=False,
        ):
        """
        严格对齐源码 inference 行为：
        - 每条样本单独 forward，传入 extract_features(source, padding_mask=None)
        - 不做 batch pad，不传 padding_mask
        - 保持可微分（只要 require_grad=True 且 require_grad_inputs=True）
        返回：
        - results: list of dict，每个 dict 包含 "feats"
        """

        if device is None:
            device = next(model.parameters()).device

        # 1) 预处理，得到逐条样本（wave_list）
        _, _, wave_list, _ = self.prepare_batch_for_emotion2vec(
            wav_batch,
            device=device,
            sr_in=sr_in,
            sr_target=sr_target,
            require_grad_inputs=require_grad_inputs,
            reduce_channels=True,
            apply_cfg_normalize=getattr(getattr(model, "cfg", None), "normalize", False),
        )

        # print(f"修改后数据处理结果：{wave_list}")

        results = []
        model.eval()

        for i, w in enumerate(wave_list):
            # 按源码要求：reshape 成 [1, T]
            source = w.view(1, -1)

            with torch.set_grad_enabled(require_grad):
                feats_dict = model.extract_features(source, padding_mask=None)  # 和源码一致
                x = feats_dict["x"]  # [1, T_out, C]

            sample_x = x.squeeze(0)  # [T_out, C]

            # 聚合
            if granularity == "utterance":
                feats = sample_x.mean(dim=0)  # [C]
            else:
                feats = sample_x  # [T_out, C]

            # 如果不需要梯度 → 保留为 detached torch.Tensor（放到 CPU，便于缓存/节省 GPU 显存）
            if not require_grad:
                feats = feats.detach().cpu()   # tensor on CPU, requires_grad=False
            else:
                feats = feats.to(device)      # 保证可求导时在目标 device（通常是 GPU）


            results.append({"feats": feats})

            if output_dir and extract_embedding:
                save_feats = feats.detach().cpu().numpy() if require_grad else feats
                np.save(os.path.join(output_dir, f"{i}.npy"), save_feats)

        return results

    def compute_emotion_embeddings_from_paths_simple(
        self,
        paths,                      # list[str] 音频文件路径
        model=None,
        device=None,                 # emotion2vec 模型实例（若 None 则用 self.emotion_model）
        sr_in=16000,
        sr_target=16000,
        require_grad=False,
        require_grad_inputs=False,  # 是否让 emotion2vec 的输入保持 requires_grad（通常 False）
        cache_detach=True,          # 是否以 detached(cpu) 的形式缓存（默认 True）
    ):
        """
        简单缓存策略：path -> cpu detached tensor（shape (C,)），返回 tensor 在 self.device 上 (B, C)
        - 如果需要在后续对 orig_embs 反传梯度，**不要**使用缓存（或使用 cache_detach=False 并谨慎）。
        """
        model = model
        device = device

        emb_list = [None] * len(paths)
        to_compute = []

        # 1) 查缓存
        for i, path in enumerate(paths):
            emb_cpu = self._emotion_emb_cache.get(path, None)
            if emb_cpu is not None:
                emb_list[i] = emb_cpu.to(device)  # 将 CPU cached tensor 移到 device（无梯度）
            else:
                to_compute.append((i, path))

        # 2) 批量计算未缓存项（只做最小实现 — 使用已有的 extract_emotion_feats_only_batch）
        if len(to_compute) > 0:
            # 准备 wav list
            wavs = []
            idxs = []
            for idx, path in to_compute:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"ref audio not found: {path}")
                wav = load_audio(path, sr_in)  # 期望返回 1D tensor
                if wav is None:
                    raise RuntimeError(f"Unable to load audio: {path}")
                wavs.append(wav)      # 保持原始长度，交给 extract_emotion_feats_only_batch 做重采样/处理
                idxs.append(idx)

            # 调用已有批量/逐样本提取函数（我们这里要求 require_grad=False，因为要缓存结果）
            # 注意：extract_emotion_feats_only_batch 的返回格式是 list of dict with "feats"
            # 我们这里 force require_grad=False 保证结果可 detach 并移动到 cpu 存储
            results = self.extract_emotion_feats_only_batch(
                model,
                wavs,
                device=device,
                require_grad=require_grad,            # 计算缓存时不保留梯度
                require_grad_inputs=require_grad_inputs,
                sr_in=sr_in,
                sr_target=sr_target,
            )

            # 存缓存并填充 emb_list
            for (idx, path), res in zip(to_compute, results):
                feats = res["feats"]
                feats = feats.to(torch.float32)
                # 存为 CPU detached（默认）
                cpu_detached = feats.detach().cpu()
                if cache_detach:
                    # 存缓存的始终为 detached CPU tensor（安全，不占 GPU）
                    self._emotion_emb_cache[path] = cpu_detached
                    emb_list[idx] = cpu_detached.to(device)
                else:
                    # 仍把缓存做为 cpu_detached（避免占 GPU），但返回值可以是无需 detach 的 feats（已在 device）
                    self._emotion_emb_cache[path] = cpu_detached
                    emb_list[idx] = feats.to(device)

        # 3) 打包并返回 (B, C)
        assert all(e is not None for e in emb_list), "Some emotion embeddings missing!"
        # emb_list 中的元素是 (C,) 张量（device 上），stack 成 (B, C)
        emb_stack = torch.stack([e.to(device) for e in emb_list], dim=0)
        return emb_stack  # shape [B, C]，在 self.device 上

    def emotion_consistency_loss(self,gen_embs, gt_embs):
        """
        gen_embs: [B, D] 生成音频的情感embedding (可回传梯度)
        gt_embs:  [B, D] 原始音频的情感embedding (不需要梯度，detach即可)
        """
        gen_embs = F.normalize(gen_embs, dim=-1)
        gt_embs = F.normalize(gt_embs, dim=-1)

        # 逐样本计算 cos_sim
        cos_sim = (gen_embs * gt_embs).sum(dim=-1)  # [B]

        # 取均值，损失改成1 - cos_sim
        # cos_sim越大——》loss越小（与CE损失方向一致）
        loss = 1-cos_sim.mean()

        return loss

    # 原代码
    # def train_step(self, batch, criterion):
    #     loss_dict = {}
    #     cond_mels = batch["cond_mels"]
    #     text_inputs = batch["text_inputs"]
    #     text_lengths = batch["text_lengths"]
    #     audio_codes = batch["audio_codes"]
    #     wav_lengths = batch["wav_lengths"]
    #     cond_idxs = batch["cond_idxs"]
    #     cond_lens = batch["cond_lens"]

    #     # loss_text, loss_mel, _ = self.forward(
    #     #     text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
    #     # )
    #     loss_text, loss_mel, _ ,latent= self.forward(
    #         text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
    #     )

    #     loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
    #     loss_dict["loss_mel_ce"] = loss_mel * self.args.gpt_loss_mel_ce_weight
    #     loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
    #     return {"model_outputs": None}, loss_dict

    # 修改版引入ECL损失
    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]

        loss_text, loss_mel, _ ,latent= self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
        )
        # print("latent.shape:", latent.shape, "device:",latent.device, "requires_grad:", latent.requires_grad)
        # print(latent)
        # print(f"train_step方法中接收的mel_latent:{latent}")

        # 去除laten中的pad token
        pad_token_id = 1011
        code_lengths_wo3 = (audio_codes != pad_token_id).sum(dim=1).tolist()  # list[int], 每个样本的有效帧数

        # 冻结 HiFiGAN 参数并设为 eval()（可选，但推荐）
        self.xtts.hifigan_decoder.eval()
        for p in self.xtts.hifigan_decoder.parameters():
            p.requires_grad = False

        # --------计算每个batch中参考音频的说话人嵌入--------
        ref_paths = batch["ref_audiopath"]  # list[str], len=B (B=4)
        # -------------注意：max_ref_length和sound_norm_refs的值应该设为多少—----------------
        spk_embs = self.compute_speaker_embeddings_from_paths(ref_paths, load_sr=16000, max_ref_length=30, sound_norm_refs=False)
        # print("spk_embs的形状、类型和设备:", spk_embs.shape, spk_embs.dtype, spk_embs.device)
        # print("spk_embs sample stats:", spk_embs[0].detach().cpu().min().item(), spk_embs[0].detach().cpu().max().item())
        # print(f"打印spk_embs:{spk_embs}")
        
        # 把latent和说话人嵌入还原成语音波形
        wav_pred_list = []
        for i, real_T in enumerate(code_lengths_wo3):
            latent_i = latent[i, :real_T, :].unsqueeze(0)  # [1, T_i, 1024]
            # print(latent_i.shape)
            spk_emb_i = spk_embs[i].unsqueeze(0)           # [1, 256]
            # print(spk_emb_i.shape)
            
            wav_i = self.xtts.hifigan_decoder(latent_i, g=spk_emb_i)  # 不用 no_grad，保持梯度
            
            # --- slice 到原始音频长度，确保情感一致性 loss 对齐 ---
            # wav_i = wav_i[..., :wav_lengths[i]]  # 最后一维 slice

            wav_pred_list.append(wav_i)
        

            """ # --- 诊断打印（用于比较还原音频和原音频的长度） ---
            print(f"DIAG sample {i}:")
            print(f"  latent slice length = {latent_i.shape[1]}")
            print(f"  effective code length = {real_T}")
            print(f"  original wav length (samples) = {wav_lengths[i]}")

            # decode 后长度
            wav_len_i = wav_i.shape[-1]  # 如果 shape [1, L] 或 [1, 1, L]，取最后一维
            print(f"  HiFiGAN decoded wav length = {wav_len_i}")

            print(f"wav_{i}的形状和类型:", wav_i.shape, wav_i.dtype)
            print(f"打印wav_{i}:{wav_i}") """

        """ # 保存音频
        filenames = batch["filenames"]
        save_dir = '/mnt/hd/zjy/CoquiTTS/'
        os.makedirs(save_dir, exist_ok=True)

        for i, wav_i in enumerate(wav_pred_list):
            # 从路径中提取文件名，如 "/tmp/xxx/0001_000004.wav" -> "0001_000004.wav"
            base_name = os.path.basename(filenames[i])
            # 构造新的文件名
            new_name = "reconstruct_" + base_name
            save_file = os.path.join(save_dir, new_name)

            # wav_i shape: [1, L]，转成 [L] 并保存
            wav_tensor = wav_i.squeeze(0).detach().cpu()  # [L]
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # [channels=1, time=L]
            torchaudio.save(save_file, wav_tensor, 24000)
            print(f"Saved reconstructed wav: {save_file}") """
    
        origin_paths = batch["filenames"]
        wav_origin_list = []
        for i, origin_audio_path in enumerate(origin_paths):
            origin_wav = load_audio(origin_audio_path, 16000)
            wav_origin_list.append(origin_wav)
            # print(f"origin_wav_{i}的形状和类型:", origin_wav.shape, origin_wav.dtype)
            # print(f"打印origin_wav_{i}:{origin_wav}")

        # print(wav_origin_list)

        model = self.emotion_model
        for p in model.parameters():
            p.requires_grad = False

        # 官方调用方式，用于和修改代码的输出进行对比，验证修改代码是否正确
        # emo_model = self.emo_auto
        # wav_file = "/mnt/hd/zjy/CoquiTTS/TTS/data.txt"
        # rec_result = emo_model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=True)
        # print(f"原方法得到的result：{rec_result}")

        # HiFiGAN还原音频通过emotion2vec得到的情感表示
        results_reconstruct = self.extract_emotion_feats_only_batch(model, wav_pred_list,device=self.device, require_grad=True, require_grad_inputs=True,sr_in=24000, sr_target=16000)
        pred_embs = torch.stack([r["feats"] for r in results_reconstruct], dim=0)  # [B, C]
        # print(f"修改后得到的results：{results_reconstruct}")

        # 原音频通过emotion2vec得到的情感表示
        # results_origin = self.extract_emotion_feats_only_batch(model, wav_origin_list,device=self.device, require_grad=True, require_grad_inputs=False,sr_in=16000, sr_target=16000)
        # orig_embs = torch.stack([r["feats"] for r in results_origin], dim=0).detach()  # [B, C]
        # print(f"修改后得到的results：{results_origin}")

        # 使用 path-key 缓存（路径必须唯一且稳定）
        orig_embs = self.compute_emotion_embeddings_from_paths_simple(
            origin_paths,
            model=model,
            sr_in=16000,
            sr_target=16000,
            device=self.device,
            require_grad=False,
            require_grad_inputs=False,
            cache_detach=True
        )  # 返回 [B, C] 在 self.device 上（已 detach）

        # 计算原音频和HiFiGAN还原音频之间的情感一致性损失(余弦相似度)
        loss_ecl = self.emotion_consistency_loss(pred_embs, orig_embs)  
        # print(f"情感一致性损失为：{loss_ecl}")

        # 通过warmup动态增长情感一致性损失的权重
        global_step = getattr(self, "_trainer_global_step", None)
        # warmup_step = 300
        warmup_step = 500
        weight_target = 0.5
        if(global_step <= warmup_step):
            # 保证浮点运算与边界
            weight_ecl = weight_target * float(min(1.0, max(0.0, float(global_step) / float(warmup_step))))
        else:
            weight_ecl = weight_target
        if global_step % 50 == 0:  # 每50步打印一次，避免刷屏
            print(f"[Warmup_weight_ecl] step={global_step}, weight_ecl={weight_ecl:.5f}")
        
        loss_dict["loss_ecl"] = loss_ecl * weight_ecl

        # loss_dict["loss_ecl"] = loss_ecl * 0.5


        # global_step = getattr(self, "_trainer_global_step", None)
        # global_step = global_step - 44355
        # warmup_step = 500
        # weight_target = 0.05
        # if(global_step <= warmup_step):
        #     # 保证浮点运算与边界
        #     weight_ecl = weight_target * float(min(1.0, max(0.0, float(global_step) / float(warmup_step))))
        # else:
        #     weight_ecl = weight_target
        # # 让情感权重从0.1开始，不执行这一步情感权重将从0开始
        # weight_ecl = weight_ecl + 0.2
        # if global_step % 50 == 0:  # 每50步打印一次，避免刷屏
        #     print(f"[Warmup] step={global_step+44355}, weight_ecl={weight_ecl:.5f}")
        # loss_dict["loss_ecl"] = loss_ecl * weight_ecl

        # loss_dict["loss_ecl"] = loss_ecl * 0.25
        
        
        

        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"] = loss_mel * self.args.gpt_loss_mel_ce_weight
        # 只使用mel和text损失
        # loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]

        # 使用ECL损失和mel、text损失
        loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"] + loss_dict["loss_ecl"]
        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        trainer.model.eval()  # the whole model to eval
        # put gpt model in training mode
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            trainer.model.module.xtts.gpt.train()
        else:
            trainer.model.xtts.gpt.train()

    def on_init_end(self, trainer):  # pylint: disable=W0613
        # ignore similarities.pth on clearml save/upload
        if self.config.dashboard_logger.lower() == "clearml":
            from clearml.binding.frameworks import WeightsFileHandler

            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    # ------------------ 在get_data_loader方法中会用到 ------------------
    def flatten_samples(self,samples):
        """把 dict / list 格式的 samples 展平成 list"""
        if isinstance(samples, dict):
            flat = []
            for _, lst in samples.items():
                flat.extend(lst)
            return flat
        return list(samples)

    def build_emotion_buckets(self,dataset, flat_samples):
        """根据 emotion 或 token，把样本分到不同的桶"""
        buckets = {tid: [] for tid in EMOTION_TOKEN_IDS}
        for idx, sample in enumerate(flat_samples):
            tid = None
            if isinstance(sample, dict) and "emotion" in sample:
                em = sample.get("emotion")
                if isinstance(em, (list, tuple)):
                    for e_val in em:
                        if e_val in EMOTION_TOKEN_IDS:
                            tid = e_val
                            break
                elif em in EMOTION_TOKEN_IDS:
                    tid = em
            if tid is None:
                try:
                    toks = dataset.get_text(str(sample["text"]), sample.get("language", None))
                    for t in EMOTION_TOKEN_IDS:
                        if (toks == t).any():
                            tid = t
                            break
                except Exception:
                    tid = None
            if tid is not None:
                buckets[tid].append(idx)
        return buckets
    # --------------------------end------------------------------------

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)
            # print(f"dataset:{dataset}")
            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # ------------------- begin: emotion-balanced batch sampler -------------------
            flat_samples = self.flatten_samples(dataset.samples)
            loader = None

            if len(flat_samples) > 0:
                buckets = self.build_emotion_buckets(dataset, flat_samples)
                empty_buckets = [tid for tid, lst in buckets.items() if len(lst) == 0]

                if len(empty_buckets) == 0:
                    # balanced sampler
                    dataset_for_loader = IndexedXTTSDataset(dataset, flat_samples)
                    batch_size = config.eval_batch_size if is_eval else config.batch_size
                    assert batch_size == E, f"batch_size ({batch_size}) must equal number of emotions ({E})"
                    batch_sampler = EmotionBalancedBatchSampler(buckets, batch_size)

                    loader = DataLoader(
                        dataset_for_loader,
                        batch_sampler=batch_sampler,
                        collate_fn=dataset_for_loader.base.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )

            # fallback
            if loader is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )

            return loader
            # ------------------- end: emotion-balanced sampler (cleaned) -------------------

            """ #原实现逻辑
            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)
            # print(f"sampler:{sampler}")

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size = config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
             
            return loader """

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        #原始加载代码
        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # —— 在这里插入“手动扩容”逻辑 —— 
        # 注意：下文用到的属性名要和 Xtts 实现里一致。通常 embedding 在 self.xtts.gpt.text_embedding，text_head 在 self.xtts.gpt.text_head。

        # 1.1 拿到“当前模型”里 text_embedding/text_head 的 target 张量（新形状：6682 × hidden_dim）
        tgt_emb = self.xtts.gpt.text_embedding.weight.data       # shape = [new_vocab_size, hidden_dim]
        tgt_head_w = self.xtts.gpt.text_head.weight.data         # shape = [new_vocab_size, hidden_dim]
        tgt_head_b = self.xtts.gpt.text_head.bias.data           # shape = [new_vocab_size]

        # 1.2 从 checkpoint state 里取出“旧”的权重张量（旧形状：6681 × hidden_dim）
        #     具体 key 名称可根据实际打印 state.keys() 确认，这里假设是下面这几个：
        old_emb = state["gpt.text_embedding.weight"]             # shape = [old_vocab_size, hidden_dim]
        old_head_w = state["gpt.text_head.weight"]               # shape = [old_vocab_size, hidden_dim]
        old_head_b = state["gpt.text_head.bias"]                 # shape = [old_vocab_size]

        # 1.3 如果行数不同，就把旧权重拷到一个新张量的前半部分，后面一行随机/零初始化
        if old_emb.size(0) != tgt_emb.size(0):
            new_emb = tgt_emb.clone()                            # 先 clone 出一个新的 [new_vocab_size, hidden_dim]
            new_emb[: old_emb.size(0), :] = old_emb               # 前 old_vocab_size 行填入旧权重
            state["gpt.text_embedding.weight"] = new_emb         # 覆盖回 state

        if old_head_w.size(0) != tgt_head_w.size(0):
            new_head_w = tgt_head_w.clone()
            new_head_w[: old_head_w.size(0), :] = old_head_w
            state["gpt.text_head.weight"] = new_head_w

        if old_head_b.size(0) != tgt_head_b.size(0):
            new_head_b = tgt_head_b.clone()
            new_head_b[: old_head_b.size(0)] = old_head_b
            state["gpt.text_head.bias"] = new_head_b

        # —— 扩容完成 —— 
        
        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        """ #验证是否正确手动扩容
        # Step1：Shape 检查（修正版）
        # 1.1 拿到三个张量的 shape
        emb_shape    = self.xtts.gpt.text_embedding.weight.shape   # e.g. (6682, D)
        head_w_shape = self.xtts.gpt.text_head.weight.shape       # e.g. (6682, D)
        head_b_shape = self.xtts.gpt.text_head.bias.shape         # e.g. (6682,)
        # 1.2 直接从 emb_shape[0] 作为 “vocab_size”
        new_vocab_size = emb_shape[0]
        print(f"[Check1] text_embedding: {emb_shape}")
        print(f"[Check1] text_head.weight: {head_w_shape}")
        print(f"[Check1] text_head.bias:   {head_b_shape}")
        # 1.3 断言
        assert emb_shape[0]    == new_vocab_size, "❌ text_embedding 行数不对"
        assert head_w_shape[0] == new_vocab_size, "❌ text_head.weight 行数不对"
        assert head_b_shape[0] == new_vocab_size, "❌ text_head.bias 长度不对"
        print("✅ Step1: shape 校验通过")

        # —— Step2：数值差异检查 —— 
        # 2.1 定位旧 token 和新 token 的索引
        old_idx = emb_shape[0] - 2  # 比如 6682→old_idx=6680
        new_idx = emb_shape[0] - 1  # 6681
        # 2.2 拿 embedding 张量
        emb_tensor = self.xtts.gpt.text_embedding.weight.data  # [6682, D]
        # 2.3 比较
        if torch.allclose(emb_tensor[new_idx], emb_tensor[old_idx], atol=1e-6):
            raise RuntimeError(f"❌ Step2失败：新行 (#{new_idx}) 与旧行 (#{old_idx}) 相同")
        else:
            print(f"✅ Step2: embedding 行 #{new_idx} 与行 #{old_idx} 数值不同，初始化 OK")

        # —— Step3：前向推理 —— 
        #  拿到 encode 结果并定位 new_id
        new_token = "<ANGRY>"
        ids = self.xtts.tokenizer.encode(new_token, "zh")
        expected_id = emb_shape[0] - 1
        assert expected_id in ids, f"❌ Step3 失败：{expected_id} 不在 {ids}"
        new_id = expected_id
        print(f"[Check3] 确认到 <ANGRY> 的 ID = {new_id}") """



        #过滤gpt相关权重的加载代码
        # # 1) 加载所有模型权重
        # state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)
        # # 2) 过滤 GPT
        # filtered_state = {k: v for k, v in state.items() if not k.startswith("gpt.")}
        # # 3) 真正加载
        # load_info = self.xtts.load_state_dict(filtered_state, strict=False)
        # # 4) 收集统计信息
        # model_keys   = set(self.xtts.state_dict().keys())
        # all_ckpt_keys = set(state.keys())
        # loaded_keys   = set(filtered_state.keys()) & model_keys
        # skipped_keys  = {k for k in all_ckpt_keys if k.startswith("gpt.")}
        # # load_info.missing_keys: 在模型里有但你没提供的
        # # load_info.unexpected_keys: 你提供了但模型里没有的
        # # 5) 打印
        # print(f">> ✅ 非 GPT 模块权重加载完成")
        # print(f">> 📥 实际加载的参数数量: {len(loaded_keys)}")
        # for k in sorted(loaded_keys):
        #     print(f"   [已加载] {k}")
        # print(f">> 🧪 跳过的 GPT 参数数量: {len(skipped_keys)}")
        # for k in sorted(skipped_keys):
        #     print(f"   [跳过 GPT] {k}")
        # print(f">> ⚠️ 缺失的参数数量: {len(load_info.missing_keys)}")
        # for k in sorted(load_info.missing_keys):
        #     print(f"   [未命中] {k}")
        # if load_info.unexpected_keys:
        #     print(f">> ❓ 意外的 checkpoint 参数: {len(load_info.unexpected_keys)}")
        #     for k in sorted(load_info.unexpected_keys):
        #         print(f"   [未注册] {k}")

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTTrainer(config)
