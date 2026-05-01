Cardio-Gemma 🫀🚀
Google TPU Sprint 2026 Submission

Cardio-Gemma is a domain-specific Large Language Model fine-tuned for Cardiovascular Medicine. This project demonstrates how to post-train Google's Gemma-3 model using JAX, Tunix, and Google Cloud TPUs (v5e), and deploy it for high-throughput inference using vLLM.

🌟 Key Highlights
Domain: Cardiology (Trained on a filtered subset of the MedQA dataset).
Framework: JAX / Flax / Tunix / Qwix (LoRA).
Hardware: Google Cloud TPU v5e (v5litepod-4).
Engineering Hack: Implemented a custom TunixBypassWrapper to solve the JAX 3D Sharding Rank Mismatch issue when passing 2D text tokens into Tunix's FSDP mesh.
📂 Repository Files
prepare_data.py - Downloads and formats the MedQA dataset for Gemma instruction tuning.
train.py - The core JAX/Tunix training script (includes the 3D Sharding Hack Wrapper).
test_model.py - Evaluates the fine-tuned model's medical accuracy.
requirements.txt - Required Python packages.
⚙️ How to Run This Project
If you want to replicate this project on a Google Cloud TPU VM, follow these steps:

1. Environment Setup
SSH into your TPU VM and install the required dependencies:

pip install -r requirements.txt
2. Hugging Face Login
You must log in to Hugging Face to download the dataset and the base Gemma-3 model.

huggingface-cli login
(Enter your Hugging Face Access Token when prompted).

3. Prepare the Dataset
Run the data preparation script to filter the cardiology questions and format them into JSONL.

python prepare_data.py
This will generate a file named cardio_data.jsonl.

4. Train the Model
Start the JAX/Tunix training loop.
Note: Do NOT use torchrun on the TPU VM, or you will hit a Device or resource busy crash. JAX handles the TPU mesh natively within a single process.

python train.py
This will train the model and save the final merged weights to ./cardio_gemma_tunix_final.

5. Test the Model
Ask the model a medical question to verify the fine-tuning was successful:

python test_model.py
6. Serve the API using vLLM
Deploy the fine-tuned model across all 4 TPU chips for high-speed inference:

vllm serve ./cardio_gemma_tunix_final \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --port 8000
You can test the live API in a new terminal window:

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "./cardio_gemma_tunix_final", "messages": [{"role": "user", "content": "Patient presents with STEMI. What is the immediate treatment?"}]}'
✍️ Full Tutorial
For a complete step-by-step guide on how this was built, including the debugging process for JAX sharding constraints, please read my full Medium article here: [INSERT_YOUR_MEDIUM_LINK_HERE]

Instructions for you:
Copy everything inside the block above.
Go to your GitHub repository in your web browser.
Click on the pencil icon (Edit) on your README.md file.
Paste this text.
Replace [INSERT_YOUR_MEDIUM_LINK_HERE] at the very bottom with the actual link to your Medium article.
Click Commit changes.
