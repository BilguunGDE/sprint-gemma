# Cardio-Gemma 🫀🚀
**Google TPU Sprint 2026 Submission**

Cardio-Gemma is a domain-specific Large Language Model fine-tuned for Cardiovascular Medicine. This project demonstrates how to post-train Google's `Gemma-3` model using **JAX, Tunix, and Google Cloud TPUs (v5e)**, and deploy it for high-throughput inference using **vLLM**.

## 🌟 Key Highlights
* **Domain:** Cardiology (Trained on a filtered subset of the MedQA dataset).
* **Framework:** JAX / Flax / Tunix / Qwix (LoRA).
* **Hardware:** Google Cloud TPU v5e (`v5litepod-4`).
* **Engineering Hack:** Implemented a custom `TunixBypassWrapper` to solve the JAX 3D Sharding Rank Mismatch issue when passing 2D text tokens into Tunix's FSDP mesh.

## 📂 Repository Files
1. `prepare_data.py` - Downloads and formats the MedQA dataset for Gemma instruction tuning.
2. `train.py` - The core JAX/Tunix training script (includes the 3D Sharding Hack Wrapper).
3. `test_model.py` - Evaluates the fine-tuned model's medical accuracy.
4. `requirements.txt` - Required Python packages.

## ⚙️ How to Run This Project

If you want to replicate this project on a Google Cloud TPU VM, follow these steps:

### 1. Environment Setup
SSH into your TPU VM and install the required dependencies:

```bash
pip install -r requirements.txt
```
2. Hugging Face Login
You must log in to Hugging Face to download the dataset and the base Gemma-3 and Gemma-4 models.
```bash
huggingface-cli login
```
(Enter your Hugging Face Access Token when prompted).

3. Prepare the Dataset
Run the data preparation script to filter the cardiology questions and format them into JSONL.
```bash
python prepare_data.py
```
4. Train the Model
Start the JAX/Tunix training loop. Note: Do NOT use torchrun on the TPU VM, or you will hit a Device or resource busy crash. JAX handles the TPU mesh natively within a single process.
```bash
python train.py
```
5. Serve the API using vLLM
Deploy the fine-tuned model across all 4 TPU chips for high-speed inference:

```bash
vllm serve ./cardio_gemma_tunix_final \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --port 8000
```
✍️ Full Tutorial
For a complete step-by-step guide on how this was built, including the debugging process for JAX sharding constraints, please read my full Medium article here: https://medium.com/@bilguun.js/fine-tuning-gemma-4-on-google-tpu-v5e-using-jax-a-complete-step-by-step-guide-2d513a13fbdd
