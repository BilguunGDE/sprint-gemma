from datasets import load_dataset

def prepare_cardio_dataset():
    print("Loading MedQA dataset from Hugging Face...")
    dataset = load_dataset("openlifescienceai/medqa", split="train")

    cardio_keywords = ["heart", "cardiac", "myocardial", "arrhythmia", "stemi", "aorta", "ventricle", "blood pressure"]
    cardio_dataset = dataset.filter(lambda x: any(kw in str(x.get('Question', '')).lower() for kw in cardio_keywords))
    cardio_dataset = cardio_dataset.select(range(min(500, len(cardio_dataset)))) 

    def format_for_gemma(example):
        opts = example.get('Options', {})
        prompt = (
            f"<start_of_turn>user\n"
            f"You are an expert cardiologist. Answer the medical question:\n"
            f"{example.get('Question', '')}\n\n"
            f"Options:\n"
            f"A: {opts.get('A', '')}\nB: {opts.get('B', '')}\n"
            f"C: {opts.get('C', '')}\nD: {opts.get('D', '')}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        response = f"The correct answer is {example.get('Correct Option', '')}: {example.get('Correct Answer', '')}<end_of_turn>"
        return {"text": prompt + response}

    prepared_dataset = cardio_dataset.map(format_for_gemma)
    prepared_dataset.to_json("cardio_data.jsonl", orient="records", lines=True)
    print(f"Success! {len(prepared_dataset)} examples saved to cardio_data.jsonl")

if __name__ == "__main__":
    prepare_cardio_dataset()
