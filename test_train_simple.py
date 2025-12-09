"""Simple training test to verify the model works"""
import torch
from pure_transformer.model import TransformerLM
from pure_transformer.configs import get_model_config

# Create tiny model
config = get_model_config('tiny')
model = TransformerLM(config).cuda()

print(f"Model: {config.model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Test training step
for step in range(5):
    # Generate random batch
    batch_size, seq_len = 4, 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Forward pass
    logits = model(x)
    
    # Simple loss (shift targets)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        x[:, 1:].reshape(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step {step+1}: loss={loss.item():.4f}")

print("\n✓ Training test successful!")
