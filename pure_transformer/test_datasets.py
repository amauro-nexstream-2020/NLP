#!/usr/bin/env python
"""
Test Dataset Availability and Token Counts

Verifies that all required datasets are accessible and estimates
available tokens for training.
"""

import sys
from transformers import AutoTokenizer
from datasets import load_dataset
import time

def test_fineweb():
    """Test FineWeb-Edu availability."""
    print("\n" + "="*80)
    print("Testing FineWeb-Edu")
    print("="*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Try to load a small subset first
        print("Loading FineWeb-Edu sample-10BT...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        
        # Test first few examples
        total_tokens = 0
        for i, example in enumerate(dataset):
            if i >= 10:
                break
            text = example.get("text", "")
            tokens = len(tokenizer.encode(text))
            total_tokens += tokens
            if i < 3:
                print(f"  Example {i+1}: {len(text)} chars, {tokens} tokens")
        
        avg_tokens = total_tokens / 10
        print(f"\n✓ FineWeb-Edu accessible!")
        print(f"  Average tokens/example: {avg_tokens:.0f}")
        print(f"  sample-10BT estimated: ~10 billion tokens")
        print(f"  Full dataset (CC-MAIN crawls): 15+ trillion tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ FineWeb-Edu error: {e}")
        return False


def test_finepdf():
    """Test FinePDF availability."""
    print("\n" + "="*80)
    print("Testing FinePDF/PDF datasets")
    print("="*80)
    
    # Try multiple PDF datasets
    pdf_datasets = [
        ("pixparse/pdfa-eng-wds", None),
        ("pixparse/cc-pdfs-filtered", None),
        ("alea-institute/dclm-pdf-text", None),
    ]
    
    for dataset_name, config in pdf_datasets:
        try:
            print(f"\nTrying {dataset_name}...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            dataset = load_dataset(
                dataset_name,
                name=config,
                split="train",
                streaming=True,
            )
            
            # Test first few examples
            total_tokens = 0
            count = 0
            for i, example in enumerate(dataset):
                if i >= 5:
                    break
                # PDFs might have different fields
                text = example.get("text", example.get("content", ""))
                if text:
                    tokens = len(tokenizer.encode(text))
                    total_tokens += tokens
                    count += 1
                    if i < 2:
                        print(f"  Example {i+1}: {len(text)} chars, {tokens} tokens")
            
            if count > 0:
                avg_tokens = total_tokens / count
                print(f"\n✓ {dataset_name} accessible!")
                print(f"  Average tokens/example: {avg_tokens:.0f}")
                return True
                
        except Exception as e:
            print(f"  {dataset_name} not available: {e}")
            continue
    
    print(f"\n⚠ Warning: No PDF dataset accessible")
    print(f"  You can proceed with FineWeb-Edu + USMLE only")
    return False


def test_usmle():
    """Test USMLE/MedQA availability."""
    print("\n" + "="*80)
    print("Testing USMLE/MedQA")
    print("="*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("Loading MedQA-USMLE...")
        dataset = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split="train",
            streaming=True,
        )
        
        # Test first few examples
        total_tokens = 0
        for i, example in enumerate(dataset):
            if i >= 10:
                break
            question = example.get("question", "")
            options = example.get("options", {})
            
            # Format as text
            text = f"Question: {question}\n\nOptions:\n"
            if isinstance(options, dict):
                for key, value in sorted(options.items()):
                    text += f"{key}) {value}\n"
            
            tokens = len(tokenizer.encode(text))
            total_tokens += tokens
            
            if i < 2:
                print(f"  Example {i+1}: {tokens} tokens")
        
        avg_tokens = total_tokens / 10
        print(f"\n✓ USMLE/MedQA accessible!")
        print(f"  Average tokens/question: {avg_tokens:.0f}")
        print(f"  Dataset size: ~10,000 questions")
        print(f"  Estimated total: ~{avg_tokens * 10000 / 1e6:.1f}M tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ USMLE/MedQA error: {e}")
        return False


def estimate_total_tokens():
    """Estimate total available tokens."""
    print("\n" + "="*80)
    print("Token Availability Summary")
    print("="*80)
    
    print("\nDataset Breakdown:")
    print("  FineWeb-Edu (sample-10BT): ~10B tokens")
    print("  FineWeb-Edu (CC-MAIN-2024-10): ~500B+ tokens")
    print("  FineWeb-Edu (full): 15T tokens")
    print("  FinePDF (if available): ~100B+ tokens")
    print("  USMLE QA: ~10M tokens")
    
    print("\nRecommended Configuration:")
    print("  • For 35B token training:")
    print("    - Use FineWeb-Edu sample-100BT or CC-MAIN crawl")
    print("    - Mix: 60% FineWeb, 30% PDF (if available), 10% USMLE")
    print("    - This ensures 35B+ tokens available")
    
    print("\n  • For 21B test run:")
    print("    - Use FineWeb-Edu sample-10BT")
    print("    - Mix: 90% FineWeb, 10% USMLE")
    print("    - Sufficient for 1 epoch on 21B tokens")


def main():
    print("="*80)
    print("DATASET VERIFICATION SCRIPT")
    print("="*80)
    print("Testing dataset availability and token counts...")
    
    results = []
    
    # Test each dataset
    results.append(("FineWeb-Edu", test_fineweb()))
    results.append(("FinePDF", test_finepdf()))
    results.append(("USMLE QA", test_usmle()))
    
    # Estimate tokens
    estimate_total_tokens()
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    for name, status in results:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}: {'PASS' if status else 'FAIL'}")
    
    passed = sum(1 for _, status in results if status)
    total = len(results)
    
    print(f"\nResult: {passed}/{total} datasets accessible")
    
    if results[0][1] and results[2][1]:  # FineWeb and USMLE
        print("\n✓ Ready for training!")
        print("  Minimum viable setup: FineWeb-Edu + USMLE QA")
        return 0
    else:
        print("\n✗ Critical datasets missing")
        print("  Cannot proceed with training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
