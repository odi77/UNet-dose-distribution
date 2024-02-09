import torch
from skimage.metrics import peak_signal_noise_ratio

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, psnr, min_psnr, max_psnr = 0, 0, float('inf'), float('-inf')

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss

            # Calculate PSNR
            batch_psnr = peak_signal_noise_ratio(y.cpu().numpy(), pred.cpu().numpy(), data_range=1.0)
            psnr += batch_psnr

            # Update min and max PSNR values
            min_psnr = min(min_psnr, batch_psnr)
            max_psnr = max(max_psnr, batch_psnr)

    test_loss /= num_batches
    psnr /= num_batches

    print(f"\nTest set: \n  PSNR: {psnr: >0.1f} dB, Avg loss: {test_loss: >8f}")
    print(f"  Min PSNR: {min_psnr: >0.1f} dB, Max PSNR: {max_psnr: >0.1f} dB\n")

    return test_loss, psnr, min_psnr, max_psnr
