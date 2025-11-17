# 🛰️ RSICD: Remote Sensing Image Captioning with BLIP

This project fine-tunes **BLIP (Bootstrapping Language–Image Pre-training)** on the **RSICD (Remote Sensing Image Captioning Dataset)** to generate clear, spatially aware captions for satellite and aerial imagery. Training is accelerated using Hugging Face’s `accelerate` library for efficient GPU utilisation.

---

## 🌐 Overview

Remote sensing images require a model that detects subtle geometric and land-use cues, such as the texture of farmland, the density of buildings, or the meandering of a river cutting through terrain. BLIP already excels at concise, factual descriptions. With fine-tuning on RSICD, the model adapts to the particular “visual language” of remote sensing.

---

## 🔍 Approach

### Model & Dataset

| Component            | Details                                 | Notes                                                                            |
| -------------------- | --------------------------------------- | -------------------------------------------------------------------------------- |
| **Model**            | `Salesforce/blip-image-captioning-base` | Strong spatial reasoning, robust pretrained vision–language encoder–decoder.     |
| **Dataset**          | `rsicd` (Hugging Face)                  | Standard train/validation/test splits used for fine-tuning and final evaluation. |
| **Dataset Handling** | Custom PyTorch `Dataset` classes        | Designed to use *all available captions* per image for richer supervision.       |

---

## 📈 Results

Fine-tuning produced consistent improvements over the base BLIP model, particularly for landscape structure and object-density descriptions.

| Model                            | Split      | BLEU Score(0-100)|
| -------------------------------- | ---------- | ---------- |
| **Base BLIP**                    | Validation | **4.66**   |
| **Fine-tuned BLIP (best epoch)** | Validation | **19.41**   |

The gain may look incremental, but in remote sensing captioning, even small BLEU improvements often correspond to noticeably clearer and more accurate descriptions.

---

## ⚙️ Training Setup

The training pipeline utilises `accelerate` for mixed-precision execution and multi-GPU adaptability, while maintaining a lightweight notebook workflow.

### Key Hyperparameters

| Parameter         | Value             | Explanation                                                                 |
| ----------------- | ----------------- | --------------------------------------------------------------------------- |
| **Learning Rate** | `5e-7`            | Extremely low to avoid erasing BLIP’s pretrained visual–language structure. |
| **Optimiser**     | AdamW             | Stable and well-behaved for transformer fine-tuning.                        |
| **Scheduler**     | ReduceLROnPlateau | Automatically lowers LR when validation loss stalls.                        |
| **Epochs**        | 5                 | Enough time for meaningful adjustments without overfitting.                 |
| **Precision**     | fp16              | Reduced epoch time from ~4 hours → ~2 hour (single Nvidia P100 16GB GPU).                    |

### Implementation Notes

* Training via `accelerate.notebook_launcher` ensured clean kernel memory usage and faster iteration.
* Full code is available in the notebook below.

---

## 📓 Notebooks

| Notebook              | Link                                                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Training Notebook** | [https://www.kaggle.com/code/rajarshi2712/rsicd-blip-training](https://www.kaggle.com/code/rajarshi2712/rsicd-blip-training) |
| **Testing Notebook**  | [https://www.kaggle.com/code/rajarshi2712/rsicd-blip-testing](https://www.kaggle.com/code/rajarshi2712/rsicd-blip-testing) |

---

## 📦 Model

| Resource                     | Link                                 |
| ---------------------------- | -------------------------------------- |
| **Fine-tuned Model Weights** | https://www.kaggle.com/models/rajarshi2712/rsicd-blip-image-caption-fine-tuned |

---

## 📄 License

This project is released under **MIT License**.

BLIP’s pretrained weights follow the original Salesforce Research license—refer to the model card on Hugging Face for details.

---
