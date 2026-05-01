from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_cardio_gemma():
    model_path = "./cardio_gemma_tunix_final"
    print(f"Loading Cardio-Gemma from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    prompt = (
        "<start_of_turn>user\n"
        "You are an expert cardiologist. A 55-year-old male arrives at the ER with severe, "
        "crushing chest pain radiating to his left arm. His ECG shows ST elevation in leads V1 to V4. "
        "What is the immediate medical intervention required?\n<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🩺 Cardio-Gemma Response:")
    print("-" * 40)
    print(response.split("model\n")[-1])

if __name__ == "__main__":
    test_cardio_gemma()
