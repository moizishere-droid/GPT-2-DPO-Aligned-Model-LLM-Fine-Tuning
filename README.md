# 🤖 GPT-2 DPO-Aligned Model

A **DPO fine-tuned GPT-2 model** for preference-aligned text generation. This project demonstrates **Direct Preference Optimization (DPO)** to align GPT-2 outputs with **human preferences**, optimized for **Google Colab free GPU**.

---

## 🚀 Highlights

* **Human Preference Alignment**: GPT-2 fine-tuned on human preference data using **DPO**.
* **Preference-Based Optimization**: Trains model to prefer outputs humans rate higher.
* **8-bit Quantization**: Memory-efficient inference for consumer GPUs.
* **Colab-Friendly**: Small base model (GPT-2) and datasets suitable for free GPU.
* **Reusable Dataset**: Includes `train_preferences.json` and `test_preferences.json`.
* **Comparison Functionality**: Compare original GPT-2 vs DPO-aligned model outputs.
* **Deployed Model**: [Access the aligned model on Hugging Face](https://huggingface.co/Abdulmoiz123/Tweet_Eval_Sentiment_LLM_Full_Fine-tuning)

---

## 📊 Training Configuration

| Parameter             | Value                   |
| --------------------- | ----------------------- |
| Base Model            | GPT-2 (124M parameters) |
| Dataset               | Human preference data   |
| DPO Training Epochs   | 3                       |
| Batch Size            | 2                       |
| Gradient Accumulation | 8                       |
| Learning Rate         | 5e-7                    |
| Max Sequence Length   | 512 tokens              |
| Max Prompt Length     | 256 tokens              |
| Optimizer             | AdamW                   |
| Precision             | FP16 (Mixed Precision)  |
| KL Coefficient (beta) | 0.1                     |

---

## ⚙️ Usage

### Load Model and Generate Responses

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "project4-rlhf/models/aligned-model"  # Local path
BASE_MODEL = "gpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)
model.eval()

def generate_response(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
prompt = "Explain quantum computing to a 10-year-old:"
print("💬 Response:", generate_response(prompt))
```

---

### Compare Original GPT-2 vs DPO-Aligned Model

```python
# Load original GPT-2
tokenizer_orig = AutoTokenizer.from_pretrained("gpt2")
tokenizer_orig.pad_token = tokenizer_orig.eos_token
model_orig = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
model_orig.eval()

# Test prompts
test_prompts = [
    "Explain quantum computing to a 10-year-old:",
    "Help me write a professional email:",
    "What's the capital of France?"
]

for prompt in test_prompts:
    print(f"📝 Prompt: {prompt}")
    print("🤖 Original GPT-2:", generate_response(prompt, model=model_orig, tokenizer=tokenizer_orig))
    print("✨ DPO-Aligned GPT-2:", generate_response(prompt))
    print("-"*70)
```

---

## 🔄 Reproducibility

* DPO fine-tuned GPT-2 trained for **3 epochs** on human preference data
* 8-bit quantization for memory efficiency
* Gradient accumulation simulates larger batch size
* Full training configuration saved for exact replication

---

## 💡 Use Cases

* Preference-aligned chatbots and assistants
* Educational AI tools
* Instruction-following AI for text generation
* Lightweight, Colab-compatible model for experimentation

---

## 📈 Advantages of DPO Fine-Tuning

* Aligns outputs with human preferences
* Memory-efficient: 8-bit quantization reduces GPU usage
* Faster and safer than full RLHF
* Easy comparison between base and aligned models
