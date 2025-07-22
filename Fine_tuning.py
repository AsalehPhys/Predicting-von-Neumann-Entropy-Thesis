import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Importing some functions from the training code
from Training import ExperimentalGNN, RydbergDataset, PhysicalScaleAwareLoss

# Fine-tuning configuration
FINETUNE_CONFIG = {
    'pretrained_model_path': 'best_model_rung1_6.pth',
    'processed_dir_larger': './processed_experimentalrung7-8_10k',
    'processed_file_name': 'data.pt',
    'batch_size': 128,
    'learning_rate': 0.5e-4,
    'weight_decay': 1.5e-4,
    'num_epochs': 200,
    'patience': 50,
    'finetuned_model_path': 'finetuned_model.pth',
    'dropout_p': 0.3,
    'grad_clip': 0.5,
    'random_seed': 42
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def fine_tune_model():
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pretrained model
    model = ExperimentalGNN(
        hidden_channels=512,
        dropout_p=FINETUNE_CONFIG['dropout_p']
    ).to(device)
    
    # Load pretrained weights
    pretrained_state_dict = torch.load(FINETUNE_CONFIG['pretrained_model_path'], map_location=device)
    model.load_state_dict(pretrained_state_dict)
    logging.info("Loaded pretrained model successfully")

    # Load the new dataset with larger system sizes
    dataset = RydbergDataset(root=FINETUNE_CONFIG['processed_dir_larger'])
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(FINETUNE_CONFIG['random_seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_CONFIG['batch_size'], shuffle=False)

    # Initialize loss and optimizer
    criterion = PhysicalScaleAwareLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_CONFIG['learning_rate'],
        weight_decay=FINETUNE_CONFIG['weight_decay']
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-7
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(FINETUNE_CONFIG['num_epochs']):
        model.train()
        total_train_loss = 0
        train_mae = 0
        total_train_samples = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
        
            pred_s = model(data)
            targets = data.y.squeeze().to(device)
            system_size = data.system_size.squeeze(-1).to(device)
            subsystem_size = data.nA.squeeze(-1).to(device)
            
            loss = criterion(pred_s, targets, system_size, subsystem_size)
            loss.backward()
            
            if FINETUNE_CONFIG['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), FINETUNE_CONFIG['grad_clip'])
            
            optimizer.step()
            
            mae = torch.abs(pred_s - targets).sum().item()
            train_mae += mae
            total_train_samples += data.num_graphs
            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_mae = train_mae / total_train_samples
        
        model.eval()
        total_val_loss = 0
        val_mae = 0
        total_val_samples = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_s = model(data)
                targets = data.y.squeeze().to(device) 
                system_size = data.system_size.squeeze(-1).to(device)  
                subsystem_size = data.nA.squeeze(-1).to(device) 
                
                loss = criterion(pred_s, targets, system_size, subsystem_size)
                total_val_loss += loss.item() * data.num_graphs
                
                mae = torch.abs(pred_s - targets).sum().item()
                val_mae += mae
                total_val_samples += data.num_graphs
                
                all_val_preds.extend(pred_s.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_val_mae = val_mae / total_val_samples
        scheduler.step()

        logging.info(f'Epoch {epoch+1}/{FINETUNE_CONFIG["num_epochs"]}:')
        logging.info(f'  Training Loss: {avg_train_loss:.6f}')
        logging.info(f'  Training MAE: {avg_train_mae:.6f}')
        logging.info(f'  Validation Loss: {avg_val_loss:.6f}')
        logging.info(f'  Validation MAE: {avg_val_mae:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), FINETUNE_CONFIG['finetuned_model_path'])
            logging.info(f'  Saved new best model (val_loss={best_val_loss:.6f})')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_CONFIG['patience']:
                logging.info('Early stopping triggered')
                break

    logging.info('Fine-tuning completed')
    logging.info(f'Best validation loss: {best_val_loss:.6f}')

if __name__ == "__main__":
    fine_tune_model()