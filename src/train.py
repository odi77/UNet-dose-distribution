import torch
import matplotlib.pyplot as plt

def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, epochs, model_save_path):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        size = len(train_dataloader.dataset)

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        size = len(val_dataloader.dataset)

        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred_val = model(X_val)
                val_loss += loss_fn(pred_val, y_val).item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Validation loss: {val_loss:.4f}\n")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_save_path)

    print("Training finished.")
    return train_losses, val_losses