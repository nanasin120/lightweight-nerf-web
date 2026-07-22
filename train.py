import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import os
from Nerf import NeRF, get_rays, render_rays
from data.dataset import Dataset
import matplotlib.pyplot as plt

N_iters = 1000
N_samples = 64
learning_rate = 5e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_interval = 50
image_interval = 10

os.makedirs('logs', exist_ok=True)

psnrs = []
iternums = []

# nerf_dataset = Dataset(r'data\tiny_nerf_data.npz')
nerf_dataset = Dataset(r'/content/drive/MyDrive/Lightweight_nerf/data/tiny_nerf_data.npz')
testimg, testpose = nerf_dataset[101]['image'].to(device), nerf_dataset[101]['pose'].to(device)

model = NeRF().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

print('Train Start')

last_time = time.time()
for i in range(N_iters + 1):
    model.train()
    epoch_start_time = time.time()

    idx = np.random.randint(len(nerf_dataset))
    sample = nerf_dataset[idx]

    image = sample['image'].to(device)
    pose = sample['pose'].to(device)
    focal = sample['focal'].item()

    H, W = image.shape[0], image.shape[1]

    rays_o, rays_d = get_rays(H, W, focal, pose)
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6, N_samples=N_samples, rand=True)
    
    loss = F.mse_loss(rgb, image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        iter_time = time.time() - last_time
        print(f"Iter [{i:4d}/{N_iters}] | Loss: {loss.item():.6f} | Time: {iter_time:.3f}s")
        last_time = time.time()

    if i % image_interval == 0:
        with torch.no_grad():
            model.eval()
            rays_o, rays_d = get_rays(H, W, focal, testpose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = F.mse_loss(rgb, testimg)
            psnr = -10. * torch.log10(loss)

            psnrs.append(psnr)
            iternums.append(i)

            plt.figure(figsize=(10,4))
            plt.subplot(121)
            plt.imshow(rgb.detach().cpu().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.savefig(f'logs/iter_{i:04d}.png')
            plt.close()

torch.save(model.state_dict(), 'logs/tiny_nerf_model.pth')
print('Done')