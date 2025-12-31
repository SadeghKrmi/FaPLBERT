# Persian Phoneme-Level BERT (training)
> This model is based on [Vaguye](https://github.com/SadeghKrmi/vaguye) phonemizer for Persian!

GPU RTX A6000 (1x) is used for training, dataset of 1.3 milion sentences, chunked, normalized from wikipedia dataset for Farsi. 
```bash
ds = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
```

Training run for 305K steps with the following result (around the following values)
```bash
Step [305000/2000000], Loss: 1.38343, Vocab Loss: 0.34593, Token Loss: 1.25434
```

dataset and model is stored in HF: `SadeghK/Fa-PLBERT`


For dataset preparation from wikipedia, refer to `wikipedia-dataset-styletts2-preparation`
```bash
wikipedia entry -> [normalize] -> [chunk] -> [hamnevise] -> [Zirneshane]
```

For further details of preprocessing, training refer to `preprocess_fa.ipynb` and `train_fa.ipynb`

---

## Phoneme-Level BERT for Enhanced Prosody of Text-to-Speech with Grapheme Predictions

### Yinghao Aaron Li, Cong Han, Xilin Jiang, Nima Mesgarani

> Large-scale pre-trained language models have been shown to be helpful in improving the naturalness of text-to-speech (TTS) models by enabling them to produce more naturalistic prosodic patterns. However, these models are usually word-level or sup-phoneme-level and jointly trained with phonemes, making them inefficient for the downstream TTS task where only phonemes are needed. In this work, we propose a phoneme-level BERT (PL-BERT) with a pretext task of predicting the corresponding graphemes along with the regular masked phoneme predictions. Subjective evaluations show that our phoneme-level BERT encoder has significantly improved the mean opinion scores (MOS) of rated naturalness of synthesized speech compared with the state-of-the-art (SOTA) StyleTTS baseline on out-of-distribution (OOD) texts.

Paper: [https://arxiv.org/abs/2301.08810](https://arxiv.org/abs/2301.08810)

Audio samples: [https://pl-bert.github.io/](https://pl-bert.github.io/)

## dataset preparation
Refer to the [wikipedia-dataset-styletts2-preparation.ipynb](https://github.com/SadeghKrmi/FaPLBERT/blob/main/wikipedia-dataset-styletts2-preparation.ipynb)

## Preprocessing
Refer to the [preprocess_fa.ipynb](https://github.com/SadeghKrmi/FaPLBERT/blob/main/preprocess_fa.ipynb)

## Trianing
Refer to the [train_fa.ipynb](https://github.com/SadeghKrmi/FaPLBERT/blob/main/train_fa.ipynb)

## References
- [NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)
- [tomaarsen/TTSTextNormalization](https://github.com/tomaarsen/TTSTextNormalization)
