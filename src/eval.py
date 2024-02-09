from src.test import test_loop
from src.visualization import plot_losses
from src.train import train_loop

def eval(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, epochs, device, model_save_path):
    train_losses, val_losses = train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, epochs, model_save_path)
    plot_losses(train_losses, val_losses)
    print("\n" + "="*200 + "\n")
    test_loop(test_dataloader, model, loss_fn, device)