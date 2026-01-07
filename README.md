# End-to-End Text-to-Speech With Emotion Label-Based Control

[![githubio](https://img.shields.io/badge/GitHub.io-Demo-blue?logo=Github&style=flat-square)](https://jinyuanzhang999.github.io/Emo-Lables-XTTS_Demo/)  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/Jinyuan0910/Emo-Lables-XTTS/tree/main)  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-blue?logo=huggingface)](https://huggingface.co/datasets/Jinyuan0910/ELDB/tree/main)


## Abstract
<p align="justify">
Emotional expression plays a vital role in improving the naturalness and expressiveness of text-to-speech (TTS) systems. However, most existing emotional TTS approaches rely on emotional reference speech or emotion-labeled datasets with limited scale, which often leads to the entanglement of emotional information and speaker timbre, resulting in unstable emotion control. In this letter, we propose an emotion control method based on emotion labels and construct a high-quality emotion-labeled speech database, termed ELDB (Emotion-Labeled Database). By introducing emotion labels on the text side and employing neutral reference speech to represent speaker timbre, the proposed method effectively disentangles emotional characteristics from speaker identity. In addition, a two-stage fine-tuning strategy is adopted, together with an emotion consistency loss (ECL), to enhance the robustness of emotion control. Experimental results demonstrate that the proposed approach consistently outperforms the baseline system and existing emotional TTS methods in both subjective and objective evaluations, achieving improved emotional expressiveness while maintaining high speech naturalness and speaker consistency. Audio samples are available on the Emo-Labels-XTTS demo page.
</p>


## Model Architecture
<table style="width:100%; text-align:center;">
  <tr>
    <td style="text-align:center;"><img src="./Model_architecture.png" alt="model framework" style="width:100%;"></td>
  </tr>
</table>


## Pre-requisites

<p align="justify">
1. Clone this repo
</p>

```bash
  git clone https://github.com/JinYuanZhang999/Emo-Lables-XTTS.git
  cd Emo-Lables-XTTS
```

<p align="justify">
2. Create a Conda environment
</p>

```bash
  conda create -n Emo-Lables-XTTS python=3.9
  conda activate Emo-Lables-XTTS
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
```

<p align="justify">
3. Download the [pretrained models](https://huggingface.co/Jinyuan0910/Emo-Lables-XTTS/tree/main)
</p>

<p align="justify">
4. Download the [ELDB](https://huggingface.co/datasets/Jinyuan0910/ELDB/tree/main) dataset
</p>


## Inference

Download the pretrained Model weights and run:

```bash
  # inference with Emo-Labels-XTTS
  # run and provide relevant parameters in the inference interface of the WebUI
  python ./TTS/demos/xtts_ft_demo.py
```

## Training 

### Stage 1 : Pre-fine-tune using the constructed emotion label dataset

<p align="justify">
1. rename the gpt_trainer_for_stage1-train.py file (which does not incorporate the ECL loss) located in the project root directory to gpt_trainer.py,and     replace the gpt_trainer.py file in ./TTS/tts/layers/xtts/trainer/ with it
</p>

<p align="justify">
2. rename the trainer_for_stage1-train.py file (which performs full fine-tuning of GPT-2) in the project root directory to trainer.py, and replace the trainer.py file located at [your Conda environment directory]/lib/python3.9/site-packages/trainer/trainer.py with it
</p>

<p align="justify">
3. run and modify the relevant parameters
</p>


```bash

  python ./TTS/demos/xtts_ft_demo.py

```

### Stage 2 : Introduce the Emotion Consistency Loss (ECL) and fine-tune only the last layer of GPT-2

<p align="justify">
1. rename the gpt_trainer_for_stage2-train.py file (which incorporate the ECL loss) located in the project root directory to gpt_trainer.py,and     replace the gpt_trainer.py file in ./TTS/tts/layers/xtts/trainer/ with it
</p>

<p align="justify">
2. rename the trainer_for_stage2-train.py file (which fine-tunes only the last layer of GPT-2) in the project root directory to trainer.py, and replace the trainer.py file located at [your Conda environment directory]/lib/python3.9/site-packages/trainer/trainer.py with it
</p>

<p align="justify">
3. run and modify the relevant parameters
</p>

```bash

  python ./TTS/demos/xtts_ft_demo.py

```

## References

- [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](https://arxiv.org/abs/2406.04904)
- [VECL-TTS: Voice identity and Emotional style controllable Cross-Lingual Text-to-Speech](https://arxiv.org/abs/2406.08076)
- [Characteristic-Specific Partial Fine-Tuning for Efficient Emotion and Speaker Adaptation in Codec Language Text-to-Speech Models](https://arxiv.org/abs/2501.14273)
- https://github.com/coqui-ai/TTS
- https://github.com/neonbjb/tortoise-tts
- https://github.com/ddlBoJack/emotion2vec





























































