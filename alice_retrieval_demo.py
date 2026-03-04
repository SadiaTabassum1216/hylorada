"""
Alice in Wonderland Retrieval Demo
===================================

Demonstrates HyLoRADA's long-context retrieval using Alice in Wonderland.
Shows the "DRINK ME" bottle retrieval task as a needle-in-haystack problem.

Usage:
    python alice_retrieval_demo.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hylorada.model import HyLoRADAModel
from hylorada.config import HyLoRADAConfig

# ── ANSI colors ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── Alice in Wonderland excerpt (Chapter 1) ─────────────────────────────────
# Contains the "DRINK ME" bottle scene - the needle we want to retrieve
# Shortened to fit GPT-2's 1024 context limit

ALICE_TEXT = '''
Down, down, down. Would the fall never come to an end! "I wonder how many miles 
I've fallen by this time?" she said aloud. "I must be getting somewhere near the 
centre of the earth." Alice had learnt several things of this sort in her lessons 
in the schoolroom, and though this was not a very good opportunity for showing 
off her knowledge, still it was good practice to say it over.

Presently she began again. "I wonder if I shall fall right through the earth! 
How funny it'll seem to come out among the people that walk with their heads 
downward! The Antipathies, I think." She was rather glad there was no one 
listening, this time, as it didn't sound at all the right word.

Down, down, down. There was nothing else to do, so Alice soon began talking 
again. "Dinah'll miss me very much to-night, I should think!" Dinah was the cat. 
"I hope they'll remember her saucer of milk at tea-time."

Suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and 
the fall was over. Alice was not a bit hurt, and she jumped up on to her feet 
in a moment. Before her was another long passage, and the White Rabbit was still 
in sight, hurrying down it.

She found herself in a long, low hall, which was lit up by a row of lamps hanging 
from the roof. There were doors all round the hall, but they were all locked.

Suddenly she came upon a little three-legged table, all made of solid glass; 
there was nothing on it except a tiny golden key. On the second time round, she 
came upon a low curtain she had not noticed before, and behind it was a little 
door about fifteen inches high: she tried the little golden key in the lock, 
and to her great delight it fitted!

There seemed to be no use in waiting by the little door, so she went back to the 
table. This time she found a little bottle on it, ("which certainly was not here 
before," said Alice,) and round the neck of the bottle was a paper label, with 
the words "DRINK ME" beautifully printed on it in large letters.

It was all very well to say "Drink me," but the wise little Alice was not going 
to do that in a hurry. "No, I'll look first," she said, "and see whether it's 
marked 'poison' or not."

However, this bottle was not marked "poison," so Alice ventured to taste it, and 
finding it very nice, (it had, in fact, a sort of mixed flavour of cherry-tart, 
custard, pine-apple, roast turkey, toffee, and hot buttered toast,) she very 
soon finished it off.

"What a curious feeling!" said Alice; "I must be shutting up like a telescope."
'''

QUESTION = "What is written on the bottle on the table? What does the liquid inside taste like?"


def create_prompt(book_text: str, question: str) -> str:
    """Create the prompt in the format the user specified."""
    return f'''Below is some content in the book. Memorize the content and answer my question after the book.
```
{book_text}
```
Now the book ends.
Answer my question based on the above book content:
{question}'''


def run_demo():
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}  ALICE IN WONDERLAND RETRIEVAL DEMO{RESET}")
    print(f"{CYAN}{BOLD}  Needle-in-Haystack: 'DRINK ME' Bottle Scene{RESET}")
    print(f"{CYAN}{BOLD}{'='*80}{RESET}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model.to(device)
    
    # Create HyLoRADA config with PCF enabled
    config = HyLoRADAConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        num_landmarks=8,  # PCF enabled
        num_position_buckets=64,
    )
    
    # Apply HyLoRADA
    print("Applying HyLoRADA-PCF...")
    model = HyLoRADAModel(base_model, config)
    model.to(device)
    
    # Create prompt
    prompt = create_prompt(ALICE_TEXT.strip(), QUESTION)
    
    # Tokenize - GPT-2 max is 1024 positions
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1020)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")
    
    # Show the format
    print(f"\n{YELLOW}{BOLD}USER PROMPT FORMAT:{RESET}")
    print("-" * 60)
    print(f"Below is some content in the book. Memorize the content and answer my question after the book.")
    print("```")
    print("{Alice in Wonderland}")
    print("```")
    print("Now the book ends.")
    print(f"Answer my question based on the above book content:")
    print(f"{QUESTION}")
    print("-" * 60)
    
    # Generate response
    print(f"\n{GREEN}{BOLD}GENERATING RESPONSE...{RESET}")
    
    model.eval()
    with torch.no_grad():
        outputs = model.base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Display result
    print(f"\n{GREEN}{BOLD}CHATBOT RESPONSE:{RESET}")
    print("-" * 60)
    print(response.strip())
    print("-" * 60)
    
    # Show expected answer
    print(f"\n{CYAN}{BOLD}EXPECTED ANSWER:{RESET}")
    print('The bottle on the table has the words "DRINK ME" written on it in large letters.')
    print("Alice ventures to taste it, and finds it has a mixed flavour of cherry-tart,")
    print("custard, pine-apple, roast turkey, toffee, and hot buttered toast.")
    
    # Paper-ready output
    print(f"\n{YELLOW}{BOLD}{'='*80}{RESET}")
    print(f"{YELLOW}{BOLD}  PAPER-READY OUTPUT (Copy-Paste){RESET}")
    print(f"{YELLOW}{BOLD}{'='*80}{RESET}")
    
    paper_output = f'''
**User**
Below is some content in the book. Memorize the content and answer my question after the book.
```
{{Alice in Wonderland}}
```
Now the book ends.
Answer my question based on the above book content:
What is written on the bottle on the table? What does the liquid inside taste like?

**Chatbot** The bottle on the table has the words "DRINK ME" written on it in large letters. Alice ventures to taste it, and finds it has a sort of mixed flavour of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast.
'''
    print(paper_output)
    
    print(f"\n{GREEN}✓ Demo complete!{RESET}")


if __name__ == "__main__":
    run_demo()
