"""
Answer Agent - Processes questions and generates answers using the answer model.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
# from answer_model_llama import AAgent
from .answer_model_llama import AAgent


def format_question_for_answering(question_obj: Dict[str, Any]) -> str:
    """Format a question object into a string for the answer model."""
    q = question_obj.get("question", "")
    choices = question_obj.get("choices", [])
    
    formatted = f"Question: {q}\n"
    for choice in choices:
        formatted += f"{choice}\n"
    
    return formatted.strip()


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter (A, B, C, or D) from the model response."""
    response = response.strip().upper()
    
    # Look for single letter answer
    if len(response) == 1 and response in "ABCD":
        return response
    
    # Look for letter at the beginning
    if len(response) > 0 and response[0] in "ABCD":
        return response[0]
    
    # Look for pattern like "Answer: A"
    for letter in "ABCD":
        if f"ANSWER: {letter}" in response or f"ANSWER: {letter.lower()}" in response.upper():
            return letter
    
    # Default to first valid letter found
    for char in response:
        if char.upper() in "ABCD":
            return char.upper()
    
    return None


def filter_answers(answers: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter answers to ensure they follow the required format."""
    def basic_checks(a1: Dict[str, str]) -> bool:
        required_keys = ['answer']
        if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
            if len(a1['answer']) == 1 and a1['answer'].upper() not in 'ABCD':
                return False
            check_len = len(a1['answer'].split())
            if check_len < 50:  # Rough token count
                check_len += len(a1.get('reasoning', 'None').split())
                if check_len < 512:
                    return True
        return False

    filtered_answers = []
    for i, a in enumerate(answers):
        if isinstance(a, dict):
            if basic_checks(a):
                filtered_answers.append(a)
            else:
                filtered_answers.append(None)
        elif isinstance(a, str):
            try:
                a1 = json.loads(a)
                if basic_checks(a1):
                    filtered_answers.append(a1)
                else:
                    filtered_answers.append(None)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at index {i}: {a}")
                filtered_answers.append(None)
        else:
            print(f"Skipping unsupported type at index {i}: {type(a)}")
            filtered_answers.append(None)
    
    return filtered_answers


def main():
    parser = argparse.ArgumentParser(description='Answer Agent - Generates answers to questions')
    parser.add_argument('--input_file', type=str, default='outputs/filtered_questions.json',
                        help='Input JSON file containing questions')
    parser.add_argument('--output_file', type=str, default='outputs/answers.json',
                        help='Output JSON file for answers')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='Custom system prompt for the agent')
    
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent # Root of project
    SCRIPT_DIR = Path(__file__).resolve().parent      # Agents directory
    
    # Robust Input Path Resolution
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        # 1. Check relative to CWD (standard behavior)
        if not input_path.exists():
            # 2. Check relative to script directory (agents/)
            if (SCRIPT_DIR / args.input_file).exists():
                input_path = SCRIPT_DIR / args.input_file
            # 3. Check relative to project root
            elif (BASE_DIR / args.input_file).exists():
                input_path = BASE_DIR / args.input_file
            # 4. Fallback: Default to agents output if it doesn't exist yet (for output)
            else:
                 # If input doesn't exist anywhere, we'll error out later.
                 # But let's assume it should be in agents/outputs if running from root
                 input_path = SCRIPT_DIR / args.input_file

    # Robust Output Path Resolution
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        # Default to placing output relative to where input was found, or script dir
        if input_path.parent.name == "outputs":
             output_path = input_path.parent / output_path.name
        else:
             output_path = SCRIPT_DIR / args.output_file
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        # List likely locations to help user debug
        print(f"Searched in:\n - {Path.cwd() / args.input_file}\n - {SCRIPT_DIR / args.input_file}\n - {BASE_DIR / args.input_file}")
        return
    
    with open(input_path, 'r') as f:
        questions = json.load(f)
    
    if args.verbose:
        print(f"Loaded {len(questions)} questions from {args.input_file}")
    
    import yaml
    
    # Load generation config
    config_path = BASE_DIR / "agen.yaml"
    gen_config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            gen_config = yaml.safe_load(f)
            
    # Initialize answer agent
    if args.verbose:
        print("Initializing Answer Agent...")
    
    agent = AAgent()
    
    # Process questions
    answers = []
    
    for idx, question_obj in enumerate(questions):
        if args.verbose and (idx + 1) % max(1, len(questions) // 10) == 0:
            print(f"Processing question {idx + 1}/{len(questions)}")
        
        try:
            # Format question
            formatted_question = format_question_for_answering(question_obj)
            
            # Generate response
            system_prompt = args.system_prompt or (
                "You are an expert at answering multiple-choice logic and reasoning questions. "
                "First, provide a very concise reasoning (under 20 words) explaining your logic. "
                "Then, on a new line, explicitly state the correct option as 'Answer: [Option]' (e.g., 'Answer: A')."
            )
            
            # Merge config with defaults
            generation_kwargs = {
                "max_new_tokens": 64,
                **gen_config
            }
            
            response, _, _ = agent.generate_response(
                formatted_question,
                system_prompt=system_prompt,
                **generation_kwargs
            )
            
            # Extract answer letter
            answer_letter = extract_answer_letter(response)
            
            if answer_letter:
                answer_obj = {
                    "answer": answer_letter,
                    "reasoning": response.strip()
                }
            else:
                answer_obj = {
                    "answer": "A",  # Default to A if extraction fails
                    "reasoning": f"Failed to extract answer from: {response}"
                }
            
            answers.append(answer_obj)
        
        except Exception as e:
            if args.verbose:
                print(f"Error processing question {idx}: {str(e)}")
            answers.append({
                "answer": "A",
                "reasoning": f"Error: {str(e)}"
            })
    
    # Save raw answers
    with open(args.output_file, 'w') as f:
        json.dump(answers, f, indent=4)
    
    if args.verbose:
        print(f"Saved {len(answers)} answers to {args.output_file}")
    
    # Filter answers
    filtered_answers = filter_answers(answers)
    filtered_output = args.output_file.replace('.json', '') + '_filtered.json'
    
    with open(filtered_output, 'w') as f:
        json.dump(filtered_answers, f, indent=4)
    
    num_valid = sum(1 for a in filtered_answers if a is not None)
    if args.verbose:
        print(f"Saved {num_valid} filtered answers to {filtered_output}")
        print(f"Validity rate: {num_valid/len(filtered_answers)*100:.1f}%")


if __name__ == "__main__":
    main()
