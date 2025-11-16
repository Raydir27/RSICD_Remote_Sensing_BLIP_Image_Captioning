## 🛰️ RSICD: Remote Sensing BLIP Image Captioning

This project focuses on **fine-tuning the BLIP (Bootstrapping Language-Image Pre-training) model** on the **RSICD (Remote Sensing Image Captioning Dataset)** to generate accurate, context-aware, and concise captions for satellite and aerial imagery. The training process was optimized for high-performance using the Hugging Face $\text{accelerate}$ library.

---

### 💡 Approach and Key Results

#### Model Selection & Data

| Component | Details | Rationale/Context |
| :--- | :--- | :--- |
| **Model** | $\text{Salesforce/blip-image-captioning-base}$ | Selected for its **superior capability in capturing spatial context** and generating short, crisp, fact-based captions. |
| **Dataset** | $\text{rsicd}$ from Hugging Face | Used the $\text{train}$ and $\text{valid}$ splits for fine-tuning and the $\text{test}$ split for final parameter testing. |
| **Custom Implementation** | $\text{CustomDataset}$ classes | Implemented to efficiently handle the **multiple captions** provided for each image in the RSICD dataset, ensuring full textual context utilization. |

#### Performance (BLEU Score)

The fine-tuned BLIP model shows a **significant improvement** in capturing remote sensing features compared to the base model:

| Model | Split Used | **BLEU Score** |
| :--- | :--- | :--- |
| **Base BLIP** | Validation | **0.51** |
| **Finetuned BLIP (Best)** | Validation | **0.56** |

---

### 🛠️ Training Optimization and Hyperparameters

The entire training process was executed using the $\text{accelerate}$ library to **minimize runtime** and **maximize GPU resource utilization**, enabling fast, checkpointed training within a notebook environment.

#### Key Training Parameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | $5e-7$ | **Very low** to prevent catastrophic forgetting of the base model's strong representations, allowing for stable and precise fine-tuning to the remote sensing domain. |
| **Optimizer** | $\text{AdamW}$ | Chosen for its **robust performance** in deep learning fine-tuning, effectively leveraging training metrics. |
| **Scheduler** | $\text{ReduceLROnPlateau}$ | Dynamically reduces the learning rate when validation loss plateaus, acting as a form of **early stopping regularization** to prevent overfitting. |
| **Number of Epochs** | 5 | Necessary due to the low learning rate, ensuring **sufficient iterations** for the model to take small, precise steps and fully adapt the weights without destabilizing pre-trained knowledge. |
| **Efficiency Gain** | **Mixed Precision (fp16)** | Using $\text{accelerate}$ with Mixed Precision **dropped the training time per epoch from $\sim 4$ hours to $\sim 1$ hour**. |

#### Code Details

* **Training Notebook:** $\text{rsicd-blip-training.ipynb}$
* Uses $\text{accelerate.notebook\_launcher}$ for optimal utilization of GPU resources.

---

### 🔗 Project Links

| Resource | Link |
| :--- | :--- |
| **Finetuned Model** | Github/Hugginface link of model coming soon... |
| **Kaggle Notebooks** | $\text{Training Notebook}$ $\text{Testing Notebook}$ |

---

### 📜 License

This code is released under the **Apache License 2.0**.

> Note: The underlying BLIP model weights are subject to their original Salesforce Research license. Please refer to the Hugging Face model card for details.
