import os
import errno
import copy
import torch
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Dataset_Aug1(data.Dataset):
    def __init__(self, folder, image_size, exts=["jpg", "jpeg", "png"]):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay=0.995,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        epochs=100000,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=10,
        results_folder="./results",
        shuffle=True,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.epochs = epochs

        self.ds = Dataset_Aug1(folder, image_size)
        self.train_dataloader = data.DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def save(self, itrs=None):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f"model.pt"))
        else:
            torch.save(data, str(self.results_folder / f"model_{itrs}.pt"))

    def train(self):
        acc_loss = 0
        for epoch in range(self.epochs):
            u_loss = 0
            for _ in range(self.gradient_accumulate_every):
                data = next(self.train_dataloader).cuda()
                loss = torch.mean(self.model(data))  # change for DP
                if epoch % 100 == 0:
                    print(f"{epoch}: {loss.item()}")
                u_loss += loss.item()

                # loss backward
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if epoch % self.update_ema_every == 0:
                self.ema.update_model_average(self.ema_model, self.model)

            if epoch != 0 and epoch % self.save_and_sample_every == 0:
                milestone = epoch // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.train_dataloader).cuda()
                xt, direct_recons, all_images = self.ema_model.module.sample(
                    batch_size=batches, img=og_img
                )  # change for DP

                og_img = (og_img + 1) * 0.5
                utils.save_image(
                    og_img,
                    str(self.results_folder / f"sample-og-{milestone}.png"),
                    nrow=6,
                )

                all_images = (all_images + 1) * 0.5
                utils.save_image(
                    all_images,
                    str(self.results_folder / f"sample-recon-{milestone}.png"),
                    nrow=6,
                )

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(
                    direct_recons,
                    str(self.results_folder / f"sample-direct_recons-{milestone}.png"),
                    nrow=6,
                )

                xt = (xt + 1) * 0.5
                utils.save_image(
                    xt, str(self.results_folder / f"sample-xt-{milestone}.png"), nrow=6
                )

                acc_loss = acc_loss / (self.save_and_sample_every + 1)
                print(f"Mean of last {epoch}: {acc_loss}")
                acc_loss = 0

                self.save()
                if epoch % (self.save_and_sample_every * 100) == 0:
                    self.save(epoch)

        print("training completed")
