# 🛰️ RSICD\_Remote\_Sensing\_BLIP\_Image\_Captioning

Fine-tuning the **BLIP** (Bootstrapping Language-Image Pre-training) model on the **RSICD** (Remote Sensing Image Captioning Dataset) to generate accurate and context-aware captions for satellite and aerial imagery.

This project focuses on optimising the training process using the Hugging Face `accelerate` library for high performance on single or multi-GPU setups within notebook environments.

---

## 💡 Approach and Results

### Model & Data
* **Model:** [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
* **Dataset:** [`rsicd`](https://huggingface.co/datasets/arampacha/rsicd) from Hugging Face. The **`train`** and **`valid`** splits were used for fine-tuning and evaluation.

### Performance (BLEU Score)
The fine-tuned BLIP model significantly improved its performance on the remote sensing data compared to its base capabilities:

| Model | Split Used | BLEU Score |
| :--- | :--- | :--- |
| Base BLIP | **Validation** | 0.51 |
| **Finetuned BLIP** (Best) | **Validation** | **0.56 (placeholder value)** |

**Conclusion:** The BLIP model was chosen over ViT-GPT2 because it demonstrated a superior ability to capture **spatial context** and generate short, crisp captions, which is critical for interpreting complex remote sensing images.

---

## 🛠️ Training Optimisation and Code Details

The training was executed using the **`accelerate`** library to minimise runtime and maximise GPU resource utilisation in a single notebook process, allowing for fast, checkpointed training.

### Key Training Parameters
* **Optimiser:** **AdamW** (Selected for robust optimisation and metric-based loss reduction).
* **Learning Rate:** **$5e-7$** (A low learning rate was found to be optimal for stable fine-tuning and proper mapping of remote sensing features).
* **Scheduler:** **ReduceLROnPlateau** (Used to dynamically prevent overfitting by adjusting the learning rate).
* **Efficiency Gain:** The use of `accelerate` and **Mixed Precision (`fp16`)** dropped the training time per epoch from approximately **4 hours to 1 hour**.

### Code Details
* **Training Notebook:** [`rsicd-blip-training.ipynb`](https://github.com/Raydir27/RSICD_Remote_Sensing_BLIP_Image_Captioning/blob/main/training/rsicd-blip-training.ipynb)
    * Uses **`accelerate.notebook_launcher`** to ensure optimal utilization of the GPU (P100/T4, etc.).
    * Features a custom `CustomDataset` class to handle the multiple captions per image in the RSICD dataset.

---

## 🔗 Links

* **Finetuned Model:** {"Github/Hugginface link of model coming soon... "}
* **Kaggle Notebooks:**
    * [Training Notebook](https://www.kaggle.com/code/rajarshi2712/rsicd-blip-training)
    * [Testing Notebook](https://www.kaggle.com/code/rajarshi2712/rsicd-blip-training)

---

## 📜 License

This code is released under the **Apache License 2.0**.

***Note:** The underlying BLIP model weights are subject to their original Salesforce Research license. Please refer to the Hugging Face model card for details.*
