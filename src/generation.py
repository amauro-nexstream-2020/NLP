"""
Text generation utilities.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (keep only top k tokens)
        top_p: Nucleus sampling (keep tokens with cumulative probability > p)
        device: Device to run on
    
    Returns:
        Generated text
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= model.config.n_positions else input_ids[:, -model.config.n_positions:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)
    
    # Decode
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def beam_search(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    beam_width: int = 5,
    device: Optional[torch.device] = None
) -> List[str]:
    """
    Generate text using beam search.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        beam_width: Number of beams
        device: Device to run on
    
    Returns:
        List of generated texts (top beam_width candidates)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Initialize beams: (sequence, score)
    beams = [(input_ids, 0.0)]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            all_candidates = []
            
            for seq, score in beams:
                # Forward pass
                idx_cond = seq if seq.size(1) <= model.config.n_positions else seq[:, -model.config.n_positions:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                
                # Get top-k tokens
                log_probs = F.log_softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(log_probs, beam_width)
                
                # Create new candidates
                for i in range(beam_width):
                    new_seq = torch.cat([seq, top_indices[:, i:i+1]], dim=1)
                    new_score = score + top_probs[0, i].item()
                    all_candidates.append((new_seq, new_score))
            
            # Keep top beam_width candidates
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Decode all beams
    results = []
    for seq, score in beams:
        generated_text = tokenizer.decode(seq[0].tolist())
        results.append(generated_text)
    
    return results
