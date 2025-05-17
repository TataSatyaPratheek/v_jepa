import torch
from src.data.dataset import create_loader
from src.models.vjepa import VJEPA
from src.utils.memory import empty_cache

def configure_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        fused=True  # Use fused AdamW implementation
    )

def train(model, loader, epochs=50):
    model.train()
    optimizer = configure_optimizer(model)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(epochs):
        for batch in loader:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                context = apply_masking(batch)
                targets = model.target_encoder(batch)
                preds = model.predictor(model.context_encoder(context))
                loss = torch.nn.functional.mse_loss(preds, targets)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
            empty_cache()
