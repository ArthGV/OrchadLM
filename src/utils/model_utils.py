import torch.nn as nn

def count_parameters(model: nn.Module, fancy_print: bool = False) -> None:
    total = sum(p.numel() for p in model.parameters())
    if fancy_print:

        print(f"{'Layer':<40} {'Shape':<25} {'Params':>10}  {'%':>6}")
        print("-" * 85)
        for name, p in model.named_parameters():
            params = p.numel()
            pct    = 100 * params / total
            print(f"{name:<40} {str(list(p.shape)):<25} {params:>10,}  {pct:>5.1f}%")
        print("-" * 85)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Total':<40} {'':25} {total:>10,}  100.0%")
    print(f"{'Trainable parameters':<40} {'':25} {trainable:>10,}  {100 * trainable / total:>5.1f}%")
    print(f"{'Non-trainable':<40} {'':25} {total - trainable:>10,}  {100 * (total - trainable) / total:>5.1f}%")