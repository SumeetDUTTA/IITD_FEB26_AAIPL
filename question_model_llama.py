# Qwen/Qwen2.5-14B-Instruct via Unsloth FastLanguageModel
# Improved Q-Agent for generating challenging MCQ questions
import os
# Set HuggingFace cache to writable location (for Jupyter environments with read-only /root/.cache)
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf_cache/hub'

import time
import torch
import re
from typing import Optional, List, Tuple, Union

torch.random.manual_seed(42)

# Check if unsloth is available, fallback to transformers
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print("Warning: Unsloth not available, falling back to transformers")

# Qwen chat template format (ChatML-style)
QWEN_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""


class QAgent(object):
    """Question Generation Agent using Qwen2.5-14B-Instruct"""
    
    def __init__(self, **kwargs):
        self.max_seq_length = kwargs.get('max_seq_length', 2048)
        
        # Check potential paths for fine-tuned model
        # Prioritize the merged model from the tutorial 
        possible_paths = [
            "logical_reasoning/logical_reasoning_rocm_merged",           # From root
            "../logical_reasoning/logical_reasoning_rocm_merged",        # From agents/ dir
            "/workspace/AAIPL/logical_reasoning/logical_reasoning_rocm_merged", # Absolute path example
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
        
        # If base model is used, ensure we point to the local cache path if it exists
        # This handles the case where downloads are restricted/cached manually
        if default_model == "Qwen/Qwen2.5-14B-Instruct":
            local_cache_path = "/tmp/hf_cache/models--Qwen--Qwen2.5-14B-Instruct/snapshots/main"
            if os.path.exists(local_cache_path):
                default_model = local_cache_path
                print(f"📂 Using local base model from: {default_model}")

        self.model_name = kwargs.get('model_name', default_model)
        print(f"🤖 Loading model: {self.model_name}")
        
        if USE_UNSLOTH:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=kwargs.get('load_in_4bit', False),
                device_map={"": 0},  # Force all weights to GPU 0
                trust_remote_code=True,
            )

            self.tokenizer.padding_side = 'left'
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            FastLanguageModel.for_inference(self.model)
        else:
            # Fallback to transformers
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side='left',
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, user_message: str, system_prompt: str) -> str:
        """Format prompt using Qwen chat template"""
        return QWEN_TEMPLATE.format(
            system_prompt=system_prompt,
            user_message=user_message
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response, handling code blocks"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        return text.strip()

    def generate_response(
        self, 
        message: Union[str, List[str]], 
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> Tuple[Union[str, List[str]], Optional[int], Optional[float]]:
        """
        Generate response(s) for the given message(s).
        
        Args:
            message: Single message string or list of messages for batch processing
            system_prompt: Optional system prompt (defaults to expert examiner prompt)
            **kwargs: Generation parameters (max_new_tokens, temperature, top_p, etc.)
            
        Returns:
            Tuple of (response(s), token_count, generation_time)
        """
        if system_prompt is None:
            # Enhanced system prompt for the fine-tuned logical reasoning model
            system_prompt = (
                "You are an expert examiner specializing in creating challenging "
                "multiple-choice questions for competitive exams. Generate questions "
                "that require careful logical reasoning, multi-step problem solving, "
                "and clear step-by-step explanations."
            )
        
        if isinstance(message, str):
            message = [message]
        
        # Format all messages using the Qwen template
        texts = [self._format_prompt(msg, system_prompt) for msg in message]
        
        # Ensure left padding for decoder-only models
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
        max_new_tokens = kwargs.get('max_new_tokens', 1024)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        do_sample = kwargs.get('do_sample', True)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        
        # Measure generation time
        if tgps_show_var:
            start_time = time.time()
        
        # Generate with specified parameters
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        if tgps_show_var:
            generation_time = time.time() - start_time
        
        # Decode the batch
        batch_outs = []
        token_len = 0 if tgps_show_var else None
        
        # Calculate tokens generated (excluding prompt)
        prompt_lengths = [len(ids) for ids in model_inputs.input_ids]
        
        for i, (input_ids, generated_sequence) in enumerate(zip(model_inputs.input_ids, generated_ids)):
            # Extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids):]
            
            # Count tokens generated
            if tgps_show_var:
                token_len += len(output_ids)
            
            # Decode the generated output
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Extract JSON if present
            content = self._extract_json(content)
            batch_outs.append(content)
        
        result = batch_outs[0] if len(batch_outs) == 1 else batch_outs
        
        if tgps_show_var:
            return result, token_len, generation_time
        return result, None, None


if __name__ == "__main__":
    # Test the QAgent
    print("Initializing QAgent...")
    model = QAgent()
    
    # Single example generation - MCQ question
    prompt = """Generate ONE challenging MCQ on the topic: Mixed Series (Alphanumeric)

Return ONLY valid JSON in this exact format:
{
    "topic": "Mixed Series (Alphanumeric)",
    "question": "Your question here?",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A",
    "explanation": "Brief explanation under 80 words"
}"""
    
    print("\n--- Single Question Generation ---")
    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    print("Response:", response)
    if tl and tm:
        print(f"Tokens: {tl}, Time: {tm:.2f}s, Speed: {tl/tm:.2f} tokens/sec")
    
    print("\n" + "=" * 60 + "\n")
    
    # Batch example generation
    prompts = [
        """Generate ONE MCQ on Syllogisms. Return ONLY valid JSON:
{
    "topic": "Syllogisms",
    "question": "Your question?",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A",
    "explanation": "Brief explanation"
}""",
        """Generate ONE MCQ on Seating Arrangements. Return ONLY valid JSON:
{
    "topic": "Seating Arrangements (Linear, Circular)",
    "question": "Your question?",
    "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "B",
    "explanation": "Brief explanation"
}""",
    ]
    
    print("--- Batch Question Generation ---")
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    print("\nBatch Responses:")
    for i, resp in enumerate(responses):
        print(f"\n[{i+1}] {resp}")
        print("-" * 40)
    
    if tl and tm:
        print(f"\nTotal tokens: {tl}, Time: {tm:.2f}s, Speed: {tl/tm:.2f} tokens/sec")
