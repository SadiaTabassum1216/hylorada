from transformers import AutoModelForCausalLM
from hylorada import HyLoRADAConfig, HyLoRADAModel

base = AutoModelForCausalLM.from_pretrained('gpt2')
model = HyLoRADAModel(base)
counts = model.count_params()

print('=== HyLoRADA Parameter Check ===')
for k, v in counts.items():
    print(f'  {k}: {v}')
print()
print('Components:', model.config.get_component_status())
