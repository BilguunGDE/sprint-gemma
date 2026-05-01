import os
import jax
import jax.numpy as jnp
import optax
import qwix
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from flax import nnx
from flax.serialization import to_bytes

# Gemma 4 Imports
from tunix.models.gemma4 import params_safetensors as params_safetensors_lib
from tunix.models.gemma4 import model as gemma4_lib
from tunix.sft import peft_trainer

def main():
    print("🚀 Initializing JAX & Tunix for GEMMA 4...")
    
    NUM_TPUS = len(jax.devices())
    MESH = [(1, NUM_TPUS), ("fsdp", "tp")] if NUM_TPUS > 1 else [(1, 1), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
    print(f" Found {NUM_TPUS} TPU chips. Mesh shape: {mesh.shape}")

    MODEL_ID = "google/gemma-4-E2B-it"
    
    print(f"\nDownloading base model & Tokenizer: {MODEL_ID}...")
    local_model_path = snapshot_download(
        repo_id=MODEL_ID, 
        ignore_patterns=["*.pth", "*.pt", "*.bin"]
    )
    
    print("Loading Tokenizer (Gemma 3 vocabulary for stability)...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading Gemma 4 into TPU HBM (using bfloat16)...")
    model_config = gemma4_lib.ModelConfig.gemma4_e2b() 
    
    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, 
            model_config, 
            mesh,
            dtype=jnp.bfloat16
        )

        print("\nApplying LoRA adapters with Qwix (MLP only)...")
        lora_provider = qwix.LoraProvider(
            module_path=".*gate_proj|.*down_proj|.*up_proj",
            rank=16,
            alpha=32,
        )
        
        dummy_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
        dummy_positions = jnp.zeros((1, 1), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, 1, 1), dtype=jnp.int32)
        
        lora_model = qwix.apply_lora_to_model(
            base_model, 
            lora_provider, 
            tokens=dummy_tokens, 
            positions=dummy_positions,
            attention_mask=dummy_mask,
            rngs=nnx.Rngs(0)
        )

        print("\nSetting up Trainer...")
        training_config = peft_trainer.TrainingConfig(
            max_steps=50,
            eval_every_n_steps=50
        )
        
        optimizer = optax.adamw(learning_rate=2e-4)
        
        trainer = peft_trainer.PeftTrainer(
            model=lora_model,
            optimizer=optimizer,
            training_config=training_config
        )

        print("\n Success: Gemma 4 Model Initialized! Loading dataset...")
        
        raw_ds = load_dataset("json", data_files="cardio_data.jsonl", split="train")
        
        def tokenize_function(examples):
            encoded = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
            
            batch_size = len(encoded["input_ids"])
            seq_length = 256
            
            positions = [list(range(seq_length)) for _ in range(batch_size)]
            mask_3d = [[[x] for x in mask] for mask in encoded["attention_mask"]]
            
            return {
                "input_tokens": encoded["input_ids"],
                "input_mask": encoded["attention_mask"],
                "attention_mask": mask_3d,
                "positions": positions
            }

        tokenized_ds = raw_ds.map(tokenize_function, batched=True)
        
        columns_to_keep = ["input_tokens", "input_mask", "attention_mask", "positions"]
        train_ds = tokenized_ds.remove_columns([col for col in tokenized_ds.column_names if col not in columns_to_keep])
        train_ds.set_format("numpy")

        def get_batched_dataloader(dataset, batch_size=1):
            while True:
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i+batch_size]
                    yield {k: jnp.array(v) for k, v in batch.items()}

        print("\n Starting Tunix Training Loop on TPU...")
        train_dataloader = get_batched_dataloader(train_ds, batch_size=1)
        trainer.train(train_dataloader, None)

        print("\n Training Complete! Saving Cardio-Gemma checkpoint via msgpack...")
        
        graphdef, model_state = nnx.split(trainer.model)
        
        def state_to_dict(node):
            if hasattr(node, 'items'):
                return {k: state_to_dict(v) for k, v in node.items()}
            return node
            
        pure_state_dict = state_to_dict(model_state)
        
        save_path = "./cardio_gemma_adapter.msgpack"
        with open(save_path, "wb") as f:
            f.write(to_bytes(pure_state_dict))
            
        print(f" Success: Model weights safely written to {save_path}!")

if __name__ == "__main__":
    main()
