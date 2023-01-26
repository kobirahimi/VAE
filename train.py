import argparse
import torch, torchvision
from torchvision import transforms
import VAE
import tqdm
import matplotlib.pyplot as plt


def train(vae, trainloader, optimizer, device):
    vae.train()  # set to training mode
    total_loss = 0
    for num_batch, (inputs, _) in enumerate(trainloader, 1):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(inputs)
        loss = vae.loss(inputs, recon, mu, logvar)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss


def test(vae, testloader, filename, epoch, sample_size, device):
    vae.eval()  # set to inference mode
    with torch.no_grad():
        samples = vae.sample(sample_size)
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + '_epoch_%d.png' % epoch)
        total_loss = 0
        for num_batch, (inputs, _) in enumerate(testloader, 1):
            inputs = inputs.to(device)
            recon, mu, logvar = vae(inputs)
            loss = vae.loss(inputs, recon, mu, logvar)
            total_loss += loss.item()
        return total_loss


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    vae = VAE.Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    train_elbo = []
    test_elbo = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        train_loss = train(vae, trainloader, optimizer, device)
        train_elbo.append(train_loss / len(trainloader.dataset))
        filename = f"samples of {args.dataset}"
        test_loss = test(vae, testloader, filename, epoch + 1, args.sample_size, device)
        test_elbo.append(test_loss / len(testloader.dataset))
        print(f"Epoch {epoch + 1} finished:  train loss: {train_elbo[-1]}, test loss: {test_elbo[-1]} ")

    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.plot(train_elbo)
        ax.plot(test_elbo)
        ax.set_title("Train and Test ELBO")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ELBO")
        ax.legend(["train ELBO", "test ELBO"])
        plt.savefig("./loss/" + f"{args.dataset}_loss.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
