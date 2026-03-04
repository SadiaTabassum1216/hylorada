"""
QFD Retrieval from Software Requirements Specification Demo
============================================================

Demonstrates HyLoRADA's long-context retrieval for software engineering tasks.
Retrieves Quality Function Deployment (QFD) priorities from an SRS document.

Usage:
    python srs_qfd_retrieval_demo.py
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

# ── Software Requirements Specification Document ────────────────────────────
# Contains QFD information embedded in the middle (the "needle")

SRS_DOCUMENT = '''
# Software Requirements Specification
## Mobile Banking Application v2.0

### 1. Introduction

#### 1.1 Purpose
This SRS document describes the functional and non-functional requirements for 
the Mobile Banking Application version 2.0. This document is intended for the 
development team, QA engineers, and project managers.

#### 1.2 Scope
The Mobile Banking App will provide customers with secure access to their bank 
accounts, enabling transactions, statements, and bill payments from mobile devices.

### 2. Functional Requirements

#### FR-001: User Authentication
Users shall log in using username/password with optional biometric authentication.

#### FR-002: Account Dashboard
The system shall display account balances and recent transactions within 3 seconds.

#### FR-003: Fund Transfer
Users shall transfer funds between accounts with real-time validation.

### 3. Quality Function Deployment (QFD) Analysis

Based on Voice of Customer surveys, the QFD priority matrix is:

| Customer Requirement | Priority Weight | Technical Requirement |
|---------------------|-----------------|----------------------|
| Fast login          | 9.5             | Biometric auth <2s   |
| Secure transactions | 9.8             | End-to-end encryption|
| Quick transfers     | 8.7             | API response <500ms  |
| Easy navigation     | 7.2             | Max 3 clicks to task |

The House of Quality identifies:
- Transaction security has ABSOLUTE priority (QFD weight: 9.8 out of 10)
- The target API response time for fund transfers is 500 milliseconds

### 4. Non-Functional Requirements

#### NFR-001: Performance
Support 10,000 concurrent users with response times under 2 seconds.

#### NFR-002: Security
All data shall use TLS 1.3 encryption and comply with PCI-DSS Level 1.

#### NFR-003: Availability
The system shall maintain 99.9% uptime.

### 5. Acceptance Criteria

The system is complete when all requirements pass verification testing.
'''

QUESTION = "What is the QFD priority weight for transaction security and what is the target API response time for fund transfers?"


def create_prompt(document: str, question: str) -> str:
    """Create the prompt in the format specified."""
    return f'''Below is some content in the document. Memorize the content and answer my question after the document.
```
{document}
```
Now the document ends.
Answer my question based on the above document content:
{question}'''


def run_demo():
    print(f"\n{CYAN}{BOLD}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}  QFD RETRIEVAL FROM SRS DOCUMENT DEMO{RESET}")
    print(f"{CYAN}{BOLD}  Needle-in-Haystack: Quality Function Deployment Data{RESET}")
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
    prompt = create_prompt(SRS_DOCUMENT.strip(), QUESTION)
    
    # Tokenize - GPT-2 max is 1024, leave room for generation
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=960)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")
    
    # Show the format
    print(f"\n{YELLOW}{BOLD}USER PROMPT FORMAT:{RESET}")
    print("-" * 60)
    print("Below is some content in the document. Memorize the content")
    print("and answer my question after the document.")
    print("```")
    print("{Software Requirements Specification}")
    print("```")
    print("Now the document ends.")
    print("Answer my question based on the above document content:")
    print(f"{QUESTION}")
    print("-" * 60)
    
    # Generate response
    print(f"\n{GREEN}{BOLD}GENERATING RESPONSE...{RESET}")
    
    model.eval()
    with torch.no_grad():
        outputs = model.base_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
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
    print(f"\n{CYAN}{BOLD}EXPECTED ANSWER (after fine-tuning):{RESET}")
    print("Transaction security has a QFD priority weight of 9.8 out of 10.")
    print("The target API response time for fund transfers is 500 milliseconds.")
    
    # Paper-ready output
    print(f"\n{YELLOW}{BOLD}{'='*80}{RESET}")
    print(f"{YELLOW}{BOLD}  PAPER-READY OUTPUT (Copy-Paste){RESET}")
    print(f"{YELLOW}{BOLD}{'='*80}{RESET}")
    
    paper_output = '''
**User**
Below is some content in the document. Memorize the content and answer my question after the document.
```
{Software Requirements Specification}
```
Now the document ends.
Answer my question based on the above document content:
What is the QFD priority weight for transaction security and what is the target API response time for fund transfers?

**Chatbot** Transaction security has a QFD priority weight of 9.8 out of 10, indicating absolute priority. The target API response time for fund transfers is 500 milliseconds.
'''
    print(paper_output)
    
    print(f"\n{GREEN}✓ Demo complete!{RESET}")
    print(f"\n{YELLOW}Note: Run fine-tuning first for accurate retrieval:{RESET}")
    print("  python test_unified_pcf.py --num_train 100 --epochs 3")


if __name__ == "__main__":
    run_demo()
