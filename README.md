# End-to-End Text-to-Speech Based on Emotion Label Control


## Abstract
Emotional expression plays a vital role in improving the naturalness and expressiveness of text-to-speech (TTS) systems. However, most existing emotional TTS approaches rely on emotional reference speech or emotion-labeled datasets with limited scale, which often leads to the entanglement of emotional information and speaker timbre, resulting in unstable emotion control. In this letter, we propose an explicit emotion label-based emotional control framework and construct a high-quality emotion-labeled speech dataset, termed ELDB (Emotion-Labeled Database). By introducing explicit emotion labels on the text side and employing neutral reference speech to represent speaker timbre, the proposed method effectively disentangles emotional characteristics from speaker identity. In addition, a two-stage fine-tuning strategy is adopted, together with an emotion embedding consistency loss, to enhance the robustness of emotion modeling. Experimental results demonstrate that the proposed approach consistently outperforms the baseline system and existing emotional TTS methods in both subjective and objective evaluations, achieving improved emotional expressiveness while maintaining high speech naturalness and speaker consistency.

Audio samples:https://huggingface.co/Jinyuan0910/Emo-Lables-XTTS

We also provide the [pretrained models](https://1drv.ms/f/c/87587ec0bae9be5a/Ek_2ur6Uwr5Lq1g-C5-5FFUB5JkhHHhLPg9iQxKxFvHm0w?e=Zpcxec).

## Model Architecture
<table style="width:100%; text-align:center;">
  <tr>
    <td style="text-align:center;"><img src="./Model_Architecture.png" alt="model framework" style="width:100%;"></td>
  </tr>
  <tr>
    <th>Model Framework</th>
  </tr>
</table>


## Pre-requisites

1. Clone this repo:

```python
   git clone https://github.com/JinYuanZhang999/Emo-Lables-XTTS.git
   cd Emo-Lables-XTTS
```

```bash
  git clone https://github.com/JinYuanZhang999/Emo-Lables-XTTS.git
  cd Emo-Lables-XTTS
```

3. CD into this repo: `cd Emo-Lables-XTTS`

4. Install python requirements: `pip install -r requirements.txt`

5. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset (for training only)


## Inference Example

Download the pretrained checkpoints and run:

```python
# inference with NeuralVC
# Replace the corresponding parameters
convert.ipynb
```

## Training Example

1. Preprocess

```python

# run this if you want a different train-val-test split
python preprocess_flist.py

# run this if you want to use pretrained speaker encoder
python preprocess_spk.py

# run this if you want to use a different content feature extractor.
python preprocess_code.py

```

2. Train

```python
# train NeuralVC
python train.py


```

## References

- [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](https://arxiv.org/abs/2406.04904)
- [VECL-TTS: Voice identity and Emotional style controllable Cross-Lingual Text-to-Speech](https://arxiv.org/abs/2406.08076)
- [Characteristic-Specific Partial Fine-Tuning for Efficient Emotion and Speaker Adaptation in Codec Language Text-to-Speech Models](https://arxiv.org/abs/2501.14273)
- https://github.com/coqui-ai/TTS
- https://github.com/neonbjb/tortoise-tts
- https://github.com/ddlBoJack/emotion2vec















