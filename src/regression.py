import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

from utils.utils import (
    parse_mlp_depth,
    parse_mlp_width,
    get_optimizer,
    get_grad_norms,
    get_grad_cosine
)

import csv
import os
import wandb

from IPython import embed

def get_mlp(mlp_type, input_size, mlp_width, mlp_depth, use_ln=False, device="cuda"):
    # Choose the MLP class.
    if mlp_type == "default":
        from models.mlp import MLP as MLPClass
    elif mlp_type == "residual":
        from models.mlp import ResidualMLP as MLPClass
    elif mlp_type == "multiskip_residual":
        from models.mlp import MultiSkipResidualMLP as MLPClass
    else:
        raise NotImplementedError(f"Unknown mlp_type: {mlp_type}")
    
    trunk_hidden_size = parse_mlp_width(mlp_width)
    trunk_num_layers = parse_mlp_depth(mlp_depth, mlp_type)
    
    # Build the MLP trunk.
    mlp = MLPClass(
        input_size=int(input_size),
        hidden_size=trunk_hidden_size,
        output_size=1, # Changed to 1 for regression
        num_layers=trunk_num_layers,
        use_ln=use_ln,
        activation_fn="relu",
        device=device,
        last_act=False
    )
    return mlp

class NonStationaryDataset(Dataset):
    def __init__(self, X, y, noise_scale=0.5):
        self.X = X
        self.y = y
        self.noise_scale = noise_scale
        self.noise = np.zeros_like(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] + self.noise[idx]

    def reshuffle_noise(self):
        self.noise = np.random.normal(0, self.noise_scale, size=len(self.y))

def train_regression(run_name, mlp_type, optimizer_name, mlp_depth, non_stationary=False, epochs=25, batch_size=128, use_ln=False, device="cuda"): 
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb.
    wandb.init(
        project="regression_experiments",
        name=run_name,
        config={
            "mlp_type": mlp_type,
            "optimizer": optimizer_name,
            "mlp_depth": mlp_depth,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.00025,
            "non_stationary": non_stationary,
            "use_ln": use_ln
        }
    )
    
    # Generate synthetic regression dataset
    n_samples = 50000
    n_features = 256
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
    
    # Normalize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    if non_stationary:
        trainset = NonStationaryDataset(X_train, y_train)
        testset = NonStationaryDataset(X_test, y_test)
    else:
        trainset = TensorDataset(X_train, y_train)
        testset = TensorDataset(X_test, y_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    model = get_mlp(mlp_type, input_size=n_features, mlp_width="small", mlp_depth=mlp_depth, use_ln=use_ln, device=device).to(device)
    
    # Initialize dictionary for storing previous gradient directions
    if not hasattr(model, "prev_grad_dirs"):
        model.prev_grad_dirs = {}
    
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = get_optimizer(optimizer_name)(model.parameters(), lr=0.00025)
    
    # Lists to store epoch-level metrics
    train_loss_hist = []
    test_loss_hist = []
    
    # Define non-stationary change points
    change_points = [20, 40, 60, 80] if non_stationary else []
    
    for epoch in range(epochs):
        # Add noise at change points if non-stationary
        if non_stationary and epoch in change_points:
            trainset.reshuffle_noise()
            testset.reshuffle_noise()
            print(f"Reshuffling noise at epoch {epoch}")
            
        model.train()
        total_loss = 0
        
        # For logging aggregated gradient norms over the epoch.
        aggregated_grad_norms = {}
        # For logging aggregated gradient cosine similarities.
        aggregated_grad_cosines = {}
        
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            batch_grad_norms = get_grad_norms(model, use_ln=use_ln)
            batch_grad_cosine = get_grad_cosine(model, use_ln=use_ln)
            
            # Aggregate gradient norms.
            for key, value in batch_grad_norms.items():
                aggregated_grad_norms.setdefault(key, []).append(value)
            for key, value in batch_grad_cosine.items():
                aggregated_grad_cosines.setdefault(key, []).append(value if value is not None else 0)
            
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
        
        train_loss_hist.append(train_loss)
        
        # Evaluate on test set.
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in testloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
        
        test_loss = total_loss / len(testloader)
        print(f"Epoch {epoch+1}: Test Loss: {test_loss:.4f}")
        
        test_loss_hist.append(test_loss)
        
        # Compute average gradient norms and cosine similarities for this epoch.
        avg_grad_norms = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_grad_norms.items()}
        avg_grad_cosines = {k: sum(v_list)/len(v_list) for k, v_list in aggregated_grad_cosines.items()}
        
        # Log metrics to wandb.
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        
        for key, value in avg_grad_norms.items():
            log_dict[f"grad_norms/{key}"] = value
        for key, value in avg_grad_cosines.items():
            log_dict[f"grad_cosines/{key}"] = value
        
        wandb.log(log_dict, step=epoch)
    
    last_n = min(3, len(train_loss_hist))
    avg_train_loss = sum(train_loss_hist[-last_n:]) / last_n
    avg_test_loss = sum(test_loss_hist[-last_n:]) / last_n

    csv_filename = f"{run_name}.csv"
    csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
    
    with open(csv_filepath, "w", newline="") as csvfile:
         fieldnames = ["mlp_type", "mlp_depth", "optimizer", "non_stationary", "train_loss", "test_loss"]
         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
         writer.writeheader()
         writer.writerow({
              "mlp_type": mlp_type,
              "mlp_depth": mlp_depth,
              "optimizer": optimizer_name,
              "non_stationary": non_stationary,
              "train_loss": avg_train_loss,
              "test_loss": avg_test_loss
         })
    print(f"Logged experiment results to {csv_filepath}")
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_type", type=str, required=True)
    parser.add_argument("--mlp_depth", type=str, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--non_stationary", action="store_true", help="Make the problem non-stationary by adding random noise")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--use_ln", action="store_true", help="Use layer normalization in MLP")
    args = parser.parse_args()
    
    if args.optimizer == "kron":
        args.lr /= 3.0
    
    run_name = f"DATA:regression_MLP.TYPE:{args.mlp_type}_MLP.DEPTH:{args.mlp_depth}_OPTIM:{args.optimizer}_NS:{args.non_stationary}_LN:{args.use_ln}"
    train_regression(
        run_name,
        args.mlp_type,
        args.optimizer,
        args.mlp_depth,
        non_stationary=args.non_stationary,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_ln=args.use_ln)
