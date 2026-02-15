# Qwen/Qwen2.5-14B-Instruct via Unsloth FastLanguageModel
import os
# Set HuggingFace cache to writable location
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache/hub'

import time
import torch
from pathlib import Path
from typing import Optional, List, Union
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(42)

# Qwen chat template (ChatML)
QWEN_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

class AAgent(object):
    def __init__(self, **kwargs):
        self.max_seq_length = kwargs.get('max_seq_length', 2048)
        
        # Check potential paths for fine-tuned model
        possible_paths = [
            "logical_reasoning/logical_reasoning_rocm_merged",           # From root
            "../logical_reasoning/logical_reasoning_rocm_merged",        # From agents/ dir
            "/workspace/AAIPL/logical_reasoning/logical_reasoning_rocm_merged",
            "./logical_reasoning_rocm_merged",
        ]
        
        ft_model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                ft_model_path = path
                print(f"✅ Found fine-tuned model at: {path}")
                break
        
        # Default to fine-tuned model if found, otherwise base model
        default_model = ft_model_path if ft_model_path else "Qwen/Qwen2.5-14B-Instruct"
        
        # If using base model, try to use local cache path
        if default_model == "Qwen/Qwen2.5-14B-Instruct":
            local_cache_path = "/tmp/hf_cache/models--Qwen--Qwen2.5-14B-Instruct/snapshots/main"
            if os.path.exists(local_cache_path):
                default_model = local_cache_path
                print(f"📂 Using local base model from: {default_model}")

        self.model_name = kwargs.get('model_name', default_model)
        print(f"🤖 Loading Answer Model: {self.model_name}")
        
        # Load model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=kwargs.get('load_in_4bit', False),
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Ensure padding side is left for generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Enable native 2x faster inference
        FastLanguageModel.for_inference(self.model)
        
        # 🔥 WARMUP STEP to eliminate first-run latency
        print("🔥 Warming up model for faster inference...")
        with torch.inference_mode():
            dummy_input = self.tokenizer(["Warmup"], return_tensors="pt").to(self.model.device)
            self.model.generate(**dummy_input, max_new_tokens=1)
        print("✅ Model warmed up and ready!")

    def _format_prompt(self, user_message: str, system_prompt: str) -> str:
        """Format prompt using Qwen chat template"""
        return QWEN_TEMPLATE.format(
            system_prompt=system_prompt,
            user_message=user_message
        )

    def generate_response(self, message: Union[str, List[str]], system_prompt: Optional[str] = None, **kwargs) -> tuple:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers multiple choice questions accurately. Provide only the letter answer (A, B, C, or D) followed by a brief justification."
            
        if isinstance(message, str):
            message = [message]

        # Format all messages using the Qwen template
        texts = [self._format_prompt(msg, system_prompt) for msg in message]

        # Enforce left padding just to be safe
        self.tokenizer.padding_side = 'left'

        # Tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.model.device)

        tgps_show_var = kwargs.get('tgps_show', False)
        # Reduce default max tokens to improve speed (128 is usually enough for answer + reason)
        max_new_tokens = kwargs.get('max_new_tokens', 128) 
        temperature = kwargs.get('temperature', 0.1) # Lower temp for consistent answers
        
        if tgps_show_var:
            start_time = time.time()
            
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        if tgps_show_var:
            generation_time = time.time() - start_time

        # Decode the batch
        batch_outs = []
        token_len = 0 if tgps_show_var else None
        
        for input_ids, generated_sequence in zip(model_inputs.input_ids, generated_ids):
            output_ids = generated_sequence[len(input_ids):]
            if tgps_show_var:
                token_len += len(output_ids)
                
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            batch_outs.append(content)

        result = batch_outs[0] if len(batch_outs) == 1 else batch_outs
        
        if tgps_show_var:
            return result, token_len, generation_time
        return result, None, None


if __name__ == "__main__":
    # Test the AAgent
    print("Initializing AAgent...")
    ans_agent = AAgent()
    
    question_with_choices = """Question: What is the capital of France?
    A) London
    B) Berlin
    C) Paris
    D) Madrid"""
    
    print("\n--- Single Answer Generation ---")
    start_t = time.time()
    response, tl, gt = ans_agent.generate_response(question_with_choices, tgps_show=True, max_new_tokens=128)
    total_t = time.time() - start_t
    
    print(f"Single response: {response}")
    if tl and gt:
        print(f"Token length: {tl}, Generation Time (GPU): {gt:.2f}s, Total Call Time: {total_t:.2f}s, T/s: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Batch processing
    questions = [
        """Question: What is 2 + 2?
        A) 3
        B) 4
        C) 5
        D) 6""",
        """Question: Which planet is closest to the sun?
        A) Venus
        B) Earth
        C) Mercury
        D) Mars""",
    ]
    
    print("--- Batch Answer Generation ---")
    start_t = time.time()
    responses, tl, gt = ans_agent.generate_response(questions, max_new_tokens=128, tgps_show=True)
    total_t = time.time() - start_t
    
    print("Responses:")
    if isinstance(responses, str): responses = [responses]
    for i, resp in enumerate(responses):
        print(f"Question {i+1}: {resp}")
    if tl and gt:
        print(f"Token length: {tl}, Generation Time (GPU): {gt:.2f}s, Total Call Time: {total_t:.2f}s, T/s: {tl/gt:.2f}")
