🛰️ RSICD: Remote Sensing BLIP Image Captioning

Fine-tuning the BLIP (Bootstrapping Language-Image Pre-training) model on the RSICD (Remote Sensing Image Captioning Dataset) to generate accurate, context-aware, and concise captions for satellite and aerial imagery.

This project focuses on optimizing the training process using the Hugging Face accelerate library for high-performance fine-tuning on single or multi-GPU environments.

💡 Approach and Key Results

Model Selection & Data

Model: Salesforce/blip-image-captioning-base

Dataset: rsicd from Hugging Face. The train and valid splits were used for fine-tuning, with final parameter testing conducted on the dedicated test split.

Custom Dataset: Custom CustomDataset classes were implemented to efficiently handle the multiple captions provided for each image in the RSICD dataset, ensuring all textual context is utilized during training.

Performance (BLEU Score)

The fine-tuned BLIP model shows a significant improvement in capturing remote sensing features compared to the base model:

|

| Model | Split Used | BLEU Score (Example) |
| Base BLIP | Validation | 0.51 |
| Finetuned BLIP (Best) | Validation | 0.56 |

Conclusion: The BLIP architecture was selected over alternatives like ViT-GPT2 due to its superior capability in capturing spatial context and generating the short, crisp, and fact-based captions that are essential for interpreting complex remote sensing images.

🛠️ Training Optimization and Hyperparameters

The entire training process was executed using the accelerate library to minimize runtime and maximize GPU resource utilization, enabling fast, checkpointed training within a notebook environment.

Key Training Parameters

| Parameter | Value | Rationale |
| Learning Rate | $5e-7$ | A very low learning rate was found to be optimal. It prevents catastrophic forgetting of the base model's strong image representations while allowing for stable and precise fine-tuning to map generic vision-language features to the specifics of remote sensing images. |
| Optimizer | AdamW | Chosen for its robust performance in deep learning fine-tuning, effectively leveraging training metrics and minimizing loss. |
| Scheduler | ReduceLROnPlateau | Dynamically reduces the learning rate when the validation loss plateaus, acting as a form of early stopping regularization to prevent the model from overfitting. |
| Number of Epochs | 5 | Given the low learning rate ($5e-7$), a higher number of epochs (5) was necessary to allow the model sufficient iterations to converge. A small learning rate means the model takes smaller, more precise steps in the parameter space, requiring more passes over the data to fully adapt the model weights to the new domain without destabilizing the pre-trained knowledge. |
| Efficiency Gain | Mixed Precision (fp16) | The use of accelerate with Mixed Precision (fp16) significantly dropped the training time per epoch from approximately 4 hours to 1 hour. |

Code Details

Training Notebook: rsicd-blip-training.ipynb

Uses accelerate.notebook_launcher for optimal utilization of GPU resources.

🔗 Project Links

Finetuned Model: Github/Hugginface link of model coming soon...

Kaggle Notebooks:

Training Notebook

Testing Notebook

📜 License

This code is released under the Apache License 2.0.

Note: The underlying BLIP model weights are subject to their original Salesforce Research license. Please refer to the Hugging Face model card for details.
