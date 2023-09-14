# trainer class
import os
import errno
import copy
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.utils import data
import torchvision
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import os
import imageio
from pytorch_msssim import ssim
from collections import OrderedDict
from PIL import Image


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace(".module", "")  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace("denoise_fn.module", "module.denoise_fn")  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]


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


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=["jpg", "jpeg", "png"]):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                transforms.CenterCrop(image_size),
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
        train_num_steps=100000,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        load_path=None,
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
        self.train_num_steps = train_num_steps

        self.ds = Dataset_Aug1(folder, image_size)
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=8,
                drop_last=True,
            )
        )

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

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

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    def add_title(self, path, title):
        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(
            img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black
        )
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            vcat,
            str(title),
            (violet.shape[1] // 2, height - 2),
            font,
            0.5,
            (0, 0, 0),
            1,
            0,
        )
        cv2.imwrite(path, vcat)

    def train(self):
        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = torch.mean(self.model(data))  # change for DP
                if self.step % 100 == 0:
                    print(f"{self.step}: {loss.item()}")
                u_loss += loss.item()

                # loss backward
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).cuda()
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
                print(f"Mean of last {self.step}: {acc_loss}")
                acc_loss = 0

                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print("training completed")

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        X_0s, X_ts = self.ema_model.module.all_sample(
            batch_size=batches, img=og_img, times=s_times
        )  # change for DP

        og_img = (og_img + 1) * 0.5
        utils.save_image(
            og_img, str(self.results_folder / f"og-{extra_path}.png"), nrow=6
        )

        frames_t = []
        frames_0 = []

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(
                x_0,
                str(self.results_folder / f"sample-{i}-{extra_path}-x0.png"),
                nrow=6,
            )
            self.add_title(
                str(self.results_folder / f"sample-{i}-{extra_path}-x0.png"), str(i)
            )
            frames_0.append(
                imageio.imread(
                    str(self.results_folder / f"sample-{i}-{extra_path}-x0.png")
                )
            )

            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(
                all_images,
                str(self.results_folder / f"sample-{i}-{extra_path}-xt.png"),
                nrow=6,
            )
            self.add_title(
                str(self.results_folder / f"sample-{i}-{extra_path}-xt.png"), str(i)
            )
            frames_t.append(
                imageio.imread(
                    str(self.results_folder / f"sample-{i}-{extra_path}-xt.png")
                )
            )

        imageio.mimsave(str(self.results_folder / f"Gif-{extra_path}-x0.gif"), frames_0)
        imageio.mimsave(str(self.results_folder / f"Gif-{extra_path}-xt.gif"), frames_t)

    def paper_showing_diffusion_images_cover_page_both_sampling(self):
        import cv2

        cnt = 0
        to_show = [2, 4, 8, 16, 32, 64, 128, 192, 256]

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            (
                Forward,
                Backward_1,
                Backward_2,
                final_all_1,
                final_all_2,
            ) = self.ema_model.module.forward_and_backward_2(
                batch_size=batches, img=og_img, noise_level=0.000
            )
            og_img = (og_img + 1) * 0.5
            final_all_1 = (final_all_1 + 1) * 0.5
            final_all_2 = (final_all_2 + 1) * 0.5

            for k in range(Forward[0].shape[0]):
                l_1 = []
                l_2 = []

                utils.save_image(
                    og_img[k], str(self.results_folder / f"og_img_{cnt}.png"), nrow=1
                )
                start = cv2.imread(f"{self.results_folder}/og_img_{cnt}.png")
                l_1.append(start)
                l_2.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(
                        x_t, str(self.results_folder / f"temp.png"), nrow=1
                    )
                    x_t = cv2.imread(f"{self.results_folder}/temp.png")
                    if j in to_show:
                        l_1.append(x_t)
                        l_2.append(x_t)

                for j in range(len(Backward_1)):
                    x_t = Backward_1[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(
                        x_t, str(self.results_folder / f"temp.png"), nrow=1
                    )
                    x_t = cv2.imread(f"{self.results_folder}/temp.png")
                    if (len(Backward_1) - j) in to_show:
                        l_1.append(x_t)

                for j in range(len(Backward_2)):
                    x_t = Backward_2[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(
                        x_t, str(self.results_folder / f"temp.png"), nrow=1
                    )
                    x_t = cv2.imread(f"{self.results_folder}/temp.png")
                    if (len(Backward_2) - j) in to_show:
                        l_2.append(x_t)

                utils.save_image(
                    final_all_1[k],
                    str(self.results_folder / f"final_1_{cnt}.png"),
                    nrow=1,
                )
                final_1 = cv2.imread(f"{self.results_folder}/final_1_{cnt}.png")
                l_1.append(final_1)

                utils.save_image(
                    final_all_2[k],
                    str(self.results_folder / f"final_2_{cnt}.png"),
                    nrow=1,
                )
                final_2 = cv2.imread(f"{self.results_folder}/final_2_{cnt}.png")
                l_2.append(final_2)

                im_h = cv2.hconcat(l_1)
                cv2.imwrite(f"{self.results_folder}/all_1_{cnt}.png", im_h)

                im_h = cv2.hconcat(l_2)
                cv2.imwrite(f"{self.results_folder}/all_2_{cnt}.png", im_h)

                cnt += 1

    def paper_showing_diffusion_images_cover_page(self):
        import cv2

        cnt = 0
        to_show = [2, 4, 8, 16, 32, 64, 128, 192, 256]

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            Forward, Backward, final_all = self.ema_model.module.forward_and_backward(
                batch_size=batches, img=og_img, noise_level=0.002
            )
            og_img = (og_img + 1) * 0.5
            final_all = (final_all + 1) * 0.5

            for k in range(Forward[0].shape[0]):
                l = []

                utils.save_image(
                    og_img[k], str(self.results_folder / f"og_img_{cnt}.png"), nrow=1
                )
                start = cv2.imread(f"{self.results_folder}/og_img_{cnt}.png")
                l.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(
                        x_t, str(self.results_folder / f"temp.png"), nrow=1
                    )
                    x_t = cv2.imread(f"{self.results_folder}/temp.png")
                    if j in to_show:
                        l.append(x_t)

                for j in range(len(Backward)):
                    x_t = Backward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(
                        x_t, str(self.results_folder / f"temp.png"), nrow=1
                    )
                    x_t = cv2.imread(f"{self.results_folder}/temp.png")
                    if (len(Backward) - j) in to_show:
                        l.append(x_t)

                utils.save_image(
                    final_all[k], str(self.results_folder / f"final_{cnt}.png"), nrow=1
                )
                final = cv2.imread(f"{self.results_folder}/final_{cnt}.png")
                l.append(final)

                im_h = cv2.hconcat(l)
                cv2.imwrite(f"{self.results_folder}/all_{cnt}.png", im_h)

                cnt += 1

    def sample_as_a_mean_blur_torch_gmm_ablation(
        self, torch_gmm, ch=3, clusters=10, noise=0
    ):
        all_samples = None
        batch_size = 100

        dl = data.DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            drop_last=True,
        )

        for i, img in enumerate(dl, 0):
            print(img.shape)
            img = torch.mean(img, [2, 3])
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        all_samples = all_samples.cuda()
        print(all_samples.shape)

        model = torch_gmm(
            num_components=clusters,
            trainer_params=dict(gpus=1),
            covariance_type="full",
            convergence_tolerance=0.001,
            batch_size=batch_size,
        )
        model.fit(all_samples)

        print(model.get_params())
        print(model)
        import pdb

        pdb.set_trace()

        num_samples = 6400
        og_x = model.sample(num_datapoints=num_samples)
        og_x = og_x.cuda()
        og_x = og_x.unsqueeze(2)
        og_x = og_x.unsqueeze(3)

        xt_folder = f"{self.results_folder}_xt"
        create_folder(xt_folder)

        out_folder = f"{self.results_folder}_out"
        create_folder(out_folder)

        direct_recons_folder = f"{self.results_folder}_dir_recons"
        create_folder(direct_recons_folder)

        cnt = 0
        bs = 64
        for j in range(100):
            og_img = og_x[j * bs : j * bs + bs]
            print(og_img.shape)
            og_img = og_img.expand(bs, ch, 128, 128)
            og_img = og_img.type(torch.cuda.FloatTensor)

            print(og_img.shape)
            xt, direct_recons, all_images = self.ema_model.module.gen_sample(
                batch_size=bs, img=og_img, noise_level=noise
            )

            for i in range(all_images.shape[0]):
                utils.save_image(
                    (all_images[i] + 1) * 0.5,
                    str(f"{out_folder}/" + f"sample-x0-{cnt}.png"),
                )

                utils.save_image(
                    (xt[i] + 1) * 0.5, str(f"{xt_folder}/" + f"sample-x0-{cnt}.png")
                )

                utils.save_image(
                    (direct_recons[i] + 1) * 0.5,
                    str(f"{direct_recons_folder}/" + f"sample-x0-{cnt}.png"),
                )

                cnt += 1

    def sample_as_a_mean_blur_torch_gmm(
        self, torch_gmm, start=0, end=1000, ch=3, clusters=10
    ):
        all_samples = None
        dataset = self.ds
        batch_size = 100

        dl = data.DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            drop_last=True,
        )

        for i, img in enumerate(dl, 0):
            print(img.shape)
            img = torch.mean(img, [2, 3])
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        all_samples = all_samples.cuda()
        print(all_samples.shape)

        model = torch_gmm(
            num_components=clusters,
            trainer_params=dict(gpus=1),
            covariance_type="full",
            convergence_tolerance=0.001,
            batch_size=batch_size,
        )
        model.fit(all_samples)

        num_samples = 48
        noise_levels = [0.001, 0.002, 0.003, 0.004]
        for i in range(1):
            og_x = model.sample(num_datapoints=num_samples)
            og_x = og_x.cuda()
            og_x = og_x.unsqueeze(2)
            og_x = og_x.unsqueeze(3)
            og_x = og_x.expand(num_samples, ch, 128, 128)
            og_x = og_x.type(torch.cuda.FloatTensor)
            # og_img = og_x

            for noise in noise_levels:
                for j in range(3):
                    print(i, noise, j)
                    og_img = og_x
                    xt, direct_recons, all_images = self.ema_model.module.gen_sample_2(
                        batch_size=num_samples, img=og_img, noise_level=noise
                    )

                    og_img = (og_img + 1) * 0.5
                    utils.save_image(
                        og_img,
                        str(self.results_folder / f"sample-og-{noise}-{i}-{j}.png"),
                        nrow=6,
                    )

                    all_images = (all_images + 1) * 0.5
                    utils.save_image(
                        all_images,
                        str(self.results_folder / f"sample-recon-{noise}-{i}-{j}.png"),
                        nrow=6,
                    )

                    direct_recons = (direct_recons + 1) * 0.5
                    utils.save_image(
                        direct_recons,
                        str(
                            self.results_folder
                            / f"sample-direct_recons-{noise}-{i}-{j}.png"
                        ),
                        nrow=6,
                    )

                    xt = (xt + 1) * 0.5
                    utils.save_image(
                        xt,
                        str(self.results_folder / f"sample-xt-{noise}-{i}-{j}.png"),
                        nrow=6,
                    )

    def sample_as_a_blur_torch_gmm(
        self, torch_gmm, siz=4, ch=3, clusters=10, sample_at=1
    ):
        all_samples = None
        flatten = nn.Flatten()

        batch_size = 100
        dl = data.DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            drop_last=True,
        )

        for i, img in enumerate(dl, 0):
            print(i)
            print(img.shape)
            img = self.ema_model.module.opt(img.cuda(), t=sample_at)
            img = F.interpolate(img, size=siz, mode="bilinear")
            img = flatten(img).cuda()

            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        print(all_samples.shape)

        model = torch_gmm(
            num_components=clusters,
            trainer_params=dict(gpus=1),
            covariance_type="full",
            convergence_tolerance=0.001,
            batch_size=batch_size,
            covariance_regularization=0.0001,
        )
        model.fit(all_samples)

        num_samples = 48
        og_x = model.sample(num_datapoints=num_samples)
        og_x = og_x.cuda()
        og_x = og_x.reshape(num_samples, ch, siz, siz)
        og_x = F.interpolate(og_x, size=128, mode="bilinear")
        og_x = og_x.type(torch.cuda.FloatTensor)

        og_img = og_x
        print(og_img.shape)
        xt, direct_recons, all_images = self.ema_model.module.sample_from_blur(
            batch_size=num_samples, img=og_img, start=sample_at
        )

        og_img = (og_img + 1) * 0.5
        utils.save_image(
            og_img,
            str(self.results_folder / f"sample-og-{sample_at}-{siz}-{clusters}.png"),
            nrow=6,
        )

        all_images = (all_images + 1) * 0.5
        utils.save_image(
            all_images,
            str(self.results_folder / f"sample-recon-{sample_at}-{siz}-{clusters}.png"),
            nrow=6,
        )

        direct_recons = (direct_recons + 1) * 0.5
        utils.save_image(
            direct_recons,
            str(
                self.results_folder
                / f"sample-direct_recons-{sample_at}-{siz}-{clusters}.png"
            ),
            nrow=6,
        )

        xt = (xt + 1) * 0.5
        utils.save_image(
            xt,
            str(self.results_folder / f"sample-xt-{sample_at}-{siz}-{clusters}.png"),
            nrow=6,
        )

    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):
        # from skimage.metrics import structural_similarity as ssim

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1

        cnt = 0
        while cnt < all_samples.shape[0]:
            og_x = all_samples[cnt : cnt + 32]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.module.all_sample(
                batch_size=og_img.shape[0], img=og_img, times=None
            )

            og_img = og_img.to("cpu")
            blurry_imgs = X_ts[0].to("cpu")
            deblurry_imgs = X_0s[-1].to("cpu")
            direct_deblurry_imgs = X_0s[0].to("cpu")

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(
                1, 3 // direct_deblurry_imgs.shape[1], 1, 1
            )

            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

                if sanity_check:
                    folder = "./sanity_check/"
                    create_folder(folder)

                    san_imgs = og_img[0:32]
                    utils.save_image(san_imgs, str(folder + f"sample-og.png"), nrow=6)

                    san_imgs = blurry_imgs[0:32]
                    utils.save_image(san_imgs, str(folder + f"sample-xt.png"), nrow=6)

                    san_imgs = deblurry_imgs[0:32]
                    utils.save_image(
                        san_imgs, str(folder + f"sample-recons.png"), nrow=6
                    )

                    san_imgs = direct_deblurry_imgs[0:32]
                    utils.save_image(
                        san_imgs, str(folder + f"sample-direct-recons.png"), nrow=6
                    )

            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)

            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)

            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)

            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat(
                    (direct_deblurred_samples, direct_deblurry_imgs), dim=0
                )

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        fid_blur = fid_func(samples=[original_sample, blurred_samples])
        rmse_blur = torch.sqrt(torch.mean((original_sample - blurred_samples) ** 2))
        ssim_blur = ssim(
            original_sample, blurred_samples, data_range=1, size_average=True
        )
        # n_og = original_sample.cpu().detach().numpy()
        # n_bs = blurred_samples.cpu().detach().numpy()
        # ssim_blur = ssim(n_og, n_bs, data_range=n_og.max() - n_og.min(), multichannel=True)
        print(f"The FID of blurry images with original image is {fid_blur}")
        print(f"The RMSE of blurry images with original image is {rmse_blur}")
        print(f"The SSIM of blurry images with original image is {ssim_blur}")

        fid_deblur = fid_func(samples=[original_sample, deblurred_samples])
        rmse_deblur = torch.sqrt(torch.mean((original_sample - deblurred_samples) ** 2))
        ssim_deblur = ssim(
            original_sample, deblurred_samples, data_range=1, size_average=True
        )
        print(f"The FID of deblurred images with original image is {fid_deblur}")
        print(f"The RMSE of deblurred images with original image is {rmse_deblur}")
        print(f"The SSIM of deblurred images with original image is {ssim_deblur}")

        print(f"Hence the improvement in FID using sampling is {fid_blur - fid_deblur}")

        fid_direct_deblur = fid_func(
            samples=[original_sample, direct_deblurred_samples]
        )
        rmse_direct_deblur = torch.sqrt(
            torch.mean((original_sample - direct_deblurred_samples) ** 2)
        )
        ssim_direct_deblur = ssim(
            original_sample, direct_deblurred_samples, data_range=1, size_average=True
        )
        print(
            f"The FID of direct deblurred images with original image is {fid_direct_deblur}"
        )
        print(
            f"The RMSE of direct deblurred images with original image is {rmse_direct_deblur}"
        )
        print(
            f"The SSIM of direct deblurred images with original image is {ssim_direct_deblur}"
        )

        print(
            f"Hence the improvement in FID using direct sampling is {fid_blur - fid_direct_deblur}"
        )

        # x0s = X_0s[-1]
        # for i in range(x0s.shape[0]):
        #     utils.save_image( (x0s[i]+1)*0.5, str(f'{self.results_folder}/' + f'sample-x0-{cnt}.png'))
        #     cnt += 1

    def save_training_data(self):
        dataset = self.ds
        create_folder(f"{self.results_folder}/")

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = (img + 1) * 0.5
            utils.save_image(img, str(f"{self.results_folder}/" + f"{idx}.png"))
            if idx % 1000 == 0:
                print(idx)
