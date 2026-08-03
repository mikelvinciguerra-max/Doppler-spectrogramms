import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import DopplerDataset
from model import CNN

def metric(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target) 
    return correct.sum().item() / len(target)

def train(train_loader, valid_loader, epochs, model, criterion, metric, optimizer, device):
    len_train = len(train_loader)
    len_valid = len(valid_loader)

    history = {'w': [], 'b': [], 'loss': [], 'val_loss': [], 'score': [], 'val_score': []}
    print(f"Beginning of training with {len_train} batches, validation on {len_valid} batches.")

    for e in range(epochs):
        train_loss, train_score = 0.0, 0.0
        valid_loss, valid_score = 0.0, 0.0
        
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_pred = model(x)
            loss = criterion(train_pred, y)
            score = metric(train_pred, y)
            train_loss += loss.item()
            train_score += score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                valid_pred = model(x)
                loss = criterion(valid_pred, y)
                score = metric(valid_pred, y)
                valid_loss += loss.item()
                valid_score += score

        params = list(model.parameters())
        history['w'].append(params[0].mean().item())
        history['b'].append(params[1].mean().item())
        history['loss'].append(train_loss/len_train)
        history['score'].append(train_score/len_train)
        history['val_loss'].append(valid_loss/len_valid)
        history['val_score'].append(valid_score/len_valid)
        
        print(f"Epoch {e+1:>4}/{epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f} - val_loss : {valid_loss/len_valid:>9.3f} - val_score : {valid_score/len_valid:>9.3f}")

    return history


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Training CNN on Doppler profiles")
    parser.add_argument("--train_env", type=str, default="doppler_output_a", help="Folder name of the training environment")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    ROOT_DIR     = "/media/mikel/Elements1/MikelVinciguerra/dataset_PC_ehunam/"     
    TRAIN_ENV    = args.train_env                            
    ENV_NAMES    = ["a", "b", "c", "d"] 
    NUM_CLASSES  = 5
    BATCH_SIZE   = 64
    EPOCHS       = args.epochs
    LR           = 1e-3

    env_dirs = [os.path.join(ROOT_DIR, e) for e in ENV_NAMES]
    for d in env_dirs:
        assert os.path.isdir(d), f"Folder not found: {d}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = os.path.join(ROOT_DIR, TRAIN_ENV)
    dataset = DopplerDataset(dataset_path)

    n       = len(dataset)
    n_train = int(n * 0.65)
    n_valid = int(n * 0.175)
    n_test  = n - n_train - n_valid

    train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
    
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = train(train_loader, valid_loader, EPOCHS, model, criterion, metric, optimizer, device)

    model.eval()
    test_score = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_score += metric(model(x), y)
    print(f"\nFinal test score: {test_score / len(test_loader):.4f}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_env': TRAIN_ENV,
        'env_names': ENV_NAMES,
        'num_classes': NUM_CLASSES,
        'root_dir': ROOT_DIR,
        'epochs': EPOCHS
    }
    torch.save(checkpoint, f"models/model_doppler_{TRAIN_ENV[-1]}_epochs_{EPOCHS}.pth")
    print(f"Model saved -> models/model_doppler_{TRAIN_ENV[-1]}_epochs_{EPOCHS}.pth")