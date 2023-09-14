import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
import torchgeometry as tgm


# helpers functions


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        device_of_kernel,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        kernel_std=0.1,
        kernel_size=3,
        blur_routine="Incremental",
        train_routine="Final",
        sampling_routine="default",
        discrete=False,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.kernel_std = kernel_std
        self.kernel_size = kernel_size
        self.blur_routine = blur_routine

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.gaussian_kernels = nn.ModuleList(self.get_kernels())
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.discrete = discrete

    def blur(self, dims, std):
        return tgm.image.get_gaussian_kernel2d(dims, std)

    def get_conv(self, dims, std, mode="circular"):
        kernel = self.blur(dims, std)
        conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=dims,
            padding=int((dims[0] - 1) / 2),
            padding_mode=mode,
            bias=False,
            groups=self.channels,
        )
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(self.channels, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)

        return conv

    def get_kernels(self):
        kernels = []
        for i in range(self.num_timesteps):
            if self.blur_routine == "Incremental":
                kernels.append(
                    self.get_conv(
                        (self.kernel_size, self.kernel_size),
                        (self.kernel_std * (i + 1), self.kernel_std * (i + 1)),
                    )
                )
            elif self.blur_routine == "Constant":
                kernels.append(
                    self.get_conv(
                        (self.kernel_size, self.kernel_size),
                        (self.kernel_std, self.kernel_std),
                    )
                )
            elif self.blur_routine == "Constant_reflect":
                kernels.append(
                    self.get_conv(
                        (self.kernel_size, self.kernel_size),
                        (self.kernel_std, self.kernel_std),
                        mode="reflect",
                    )
                )
            elif self.blur_routine == "Exponential_reflect":
                ks = self.kernel_size
                kstd = np.exp(self.kernel_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode="reflect"))
            elif self.blur_routine == "Exponential":
                ks = self.kernel_size
                kstd = np.exp(self.kernel_std * i)
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
            elif self.blur_routine == "Individual_Incremental":
                ks = 2 * i + 1
                kstd = 2 * ks
                kernels.append(self.get_conv((ks, ks), (kstd, kstd)))
            elif self.blur_routine == "Special_6_routine":
                ks = 11
                kstd = i / 100 + 0.35
                kernels.append(self.get_conv((ks, ks), (kstd, kstd), mode="reflect"))

        return kernels

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None):
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)

        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)

        orig_mean = torch.mean(img, [2, 3], keepdim=True)
        print(orig_mean.squeeze()[0])

        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == "Final":
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        x = self.gaussian_kernels[t - 2](x)
                    else:
                        for i in range(t - 1):
                            with torch.no_grad():
                                x = self.gaussian_kernels[i](x)

                elif self.sampling_routine == "x0_step_down":
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.gaussian_kernels[i](x_times)
                            if self.discrete:
                                if i == (self.num_timesteps - 1):
                                    x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                    x_times = x_times.expand(
                                        temp.shape[0],
                                        temp.shape[1],
                                        temp.shape[2],
                                        temp.shape[3],
                                    )

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1
        self.denoise_fn.train()
        return xt, direct_recons, img

    @torch.no_grad()
    def gen_sample_2(self, batch_size=16, img=None, t=None, noise_level=0):
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)

        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)

        orig_mean = torch.mean(img, [2, 3], keepdim=True)
        print(orig_mean.squeeze()[0])

        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

        noise = torch.randn_like(img) * noise_level
        img = img + noise

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == "Final":
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        x = self.gaussian_kernels[t - 2](x)
                    else:
                        for i in range(t - 1):
                            with torch.no_grad():
                                x = self.gaussian_kernels[i](x)

                elif self.sampling_routine == "x0_step_down":
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.gaussian_kernels[i](x_times)
                            if self.discrete:
                                if i == (self.num_timesteps - 1):
                                    x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                    x_times = x_times.expand(
                                        temp.shape[0],
                                        temp.shape[1],
                                        temp.shape[2],
                                        temp.shape[3],
                                    )

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1

        # img = img - noise

        return xt, direct_recons, img

    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None, noise_level=0):
        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)

        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)

        orig_mean = torch.mean(img, [2, 3], keepdim=True)
        print(orig_mean.squeeze()[0])

        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

        noise = torch.randn_like(img) * noise_level
        img = img + noise

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == "Final":
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        x = self.gaussian_kernels[t - 2](x)
                    else:
                        for i in range(t - 1):
                            with torch.no_grad():
                                x = self.gaussian_kernels[i](x)

                elif self.sampling_routine == "x0_step_down":
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.gaussian_kernels[i](x_times)
                            if self.discrete:
                                if i == (self.num_timesteps - 1):
                                    x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                    x_times = x_times.expand(
                                        temp.shape[0],
                                        temp.shape[1],
                                        temp.shape[2],
                                        temp.shape[3],
                                    )

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1

        # img = img - noise

        return xt, direct_recons, img

    @torch.no_grad()
    def opt(self, img, t=None):
        if t is None:
            t = self.num_timesteps

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)
        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)

        return img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)
        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)

        X_0s = []
        X_ts = []
        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])
            noise = torch.randn_like(img) * 0.001
            img = img + noise

        # 3(2), 2(1), 1(0)
        while times:
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            X_0s.append(x)
            X_ts.append(img)

            if self.train_routine == "Final":
                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        if times - 2 >= 0:
                            x = self.gaussian_kernels[times - 2](img)
                    else:
                        x_times_sub_1 = x
                        for i in range(times - 1):
                            with torch.no_grad():
                                x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                        x = x_times_sub_1

                elif self.sampling_routine == "x0_step_down":
                    if self.blur_routine == "Individual_Incremental":
                        if times - 2 >= 0:
                            x = self.gaussian_kernels[times - 2](img)
                    else:
                        x_times = x
                        for i in range(times):
                            with torch.no_grad():
                                x_times = self.gaussian_kernels[i](x_times)
                                if self.discrete:
                                    if i == (self.num_timesteps - 1):
                                        x_times = torch.mean(
                                            x_times, [2, 3], keepdim=True
                                        )
                                        x_times = x_times.expand(
                                            temp.shape[0],
                                            temp.shape[1],
                                            temp.shape[2],
                                            temp.shape[3],
                                        )

                        x_times_sub_1 = x
                        for i in range(times - 1):
                            with torch.no_grad():
                                x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                        x = img - x_times + x_times_sub_1
                        # x = x - (noise/self.num_timesteps)

            img = x
            times = times - 1

        if self.discrete:
            img = img - noise
        X_0s.append(img)

        self.denoise_fn.train()
        return X_0s, X_ts

    @torch.no_grad()
    def forward_and_backward(
        self, batch_size=16, img=None, noise_level=0, t=None, times=None, eval=True
    ):
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t

        Forward = []
        Forward.append(img)

        if self.blur_routine == "Individual_Incremental":
            img = self.gaussian_kernels[t - 1](img)
        else:
            for i in range(t):
                with torch.no_grad():
                    img = self.gaussian_kernels[i](img)
                    Forward.append(img)

        Backward = []
        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])
            noise = torch.randn_like(img) * noise_level
            img = img + noise

        # 3(2), 2(1), 1(0)
        while times:
            print(times)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            Backward.append(img)

            if self.train_routine == "Final":
                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        if times - 2 >= 0:
                            x = self.gaussian_kernels[times - 2](img)
                    else:
                        x_times_sub_1 = x
                        for i in range(times - 1):
                            with torch.no_grad():
                                x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                        x = x_times_sub_1

                elif self.sampling_routine == "x0_step_down":
                    if self.blur_routine == "Individual_Incremental":
                        if times - 2 >= 0:
                            x = self.gaussian_kernels[times - 2](img)
                    else:
                        x_times = x
                        for i in range(times):
                            with torch.no_grad():
                                x_times = self.gaussian_kernels[i](x_times)
                                if self.discrete:
                                    if i == (self.num_timesteps - 1):
                                        x_times = torch.mean(
                                            x_times, [2, 3], keepdim=True
                                        )
                                        x_times = x_times.expand(
                                            temp.shape[0],
                                            temp.shape[1],
                                            temp.shape[2],
                                            temp.shape[3],
                                        )

                        x_times_sub_1 = x
                        for i in range(times - 1):
                            with torch.no_grad():
                                x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                        x = img - x_times + x_times_sub_1
                        # x = x - (noise/self.num_timesteps)

            img = x
            times = times - 1

        return Forward, Backward, img

    @torch.no_grad()
    def forward_and_backward_2(self, batch_size=16, img=None, noise_level=0, eval=True):
        if eval:
            self.denoise_fn.eval()

        times = self.num_timesteps

        Forward = []
        orig_img = img
        Forward.append(img)

        for i in range(times):
            with torch.no_grad():
                img = self.gaussian_kernels[i](img)
                Forward.append(img)

        Backward_1 = []
        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])
            noise = torch.randn_like(img) * noise_level
            img = img + noise
        last_img = img

        while times:
            print(times)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            Backward_1.append(img)

            x_times = x
            for i in range(times):
                with torch.no_grad():
                    x_times = self.gaussian_kernels[i](x_times)
                    if self.discrete:
                        if i == (self.num_timesteps - 1):
                            x_times = torch.mean(x_times, [2, 3], keepdim=True)
                            x_times = x_times.expand(
                                temp.shape[0],
                                temp.shape[1],
                                temp.shape[2],
                                temp.shape[3],
                            )

            x_times_sub_1 = x
            for i in range(times - 1):
                with torch.no_grad():
                    x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

            x = img - img + x_times_sub_1

            img = x
            times = times - 1

        img_1 = img

        times = self.num_timesteps
        img = last_img
        Backward_2 = []

        while times:
            print(times)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            Backward_2.append(img)

            x_times = x
            for i in range(times):
                with torch.no_grad():
                    x_times = self.gaussian_kernels[i](x_times)
                    if self.discrete:
                        if i == (self.num_timesteps - 1):
                            x_times = torch.mean(x_times, [2, 3], keepdim=True)
                            x_times = x_times.expand(
                                temp.shape[0],
                                temp.shape[1],
                                temp.shape[2],
                                temp.shape[3],
                            )

            x_times_sub_1 = x
            for i in range(times - 1):
                with torch.no_grad():
                    x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

            x = img - x_times + x_times_sub_1

            img = x
            times = times - 1

        img_2 = img

        return Forward, Backward_1, Backward_2, img_1, img_2

    @torch.no_grad()
    def sample_from_blur(
        self, batch_size=16, img=None, t=None, times=None, eval=True, start=None
    ):
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t

        if start == None:
            start = 0

        for i in range(start, t):
            with torch.no_grad():
                img = self.gaussian_kernels[i](img)

        temp = img
        if self.discrete:
            img = torch.mean(img, [2, 3], keepdim=True)
            img = img.expand(temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3])

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == "Final":
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == "default":
                    if self.blur_routine == "Individual_Incremental":
                        x = self.gaussian_kernels[t - 2](x)
                    else:
                        for i in range(t - 1):
                            with torch.no_grad():
                                x = self.gaussian_kernels[i](x)

                elif self.sampling_routine == "x0_step_down":
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.gaussian_kernels[i](x_times)
                            if self.discrete:
                                if i == (self.num_timesteps - 1):
                                    x_times = torch.mean(x_times, [2, 3], keepdim=True)
                                    x_times = x_times.expand(
                                        temp.shape[0],
                                        temp.shape[1],
                                        temp.shape[2],
                                        temp.shape[3],
                                    )

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1

        return xt, direct_recons, img

    def q_sample(self, x_start, t):
        # So at present we will for each batch blur it till the max in t.
        # And save it. And then use t to pull what I need. It is nothing but series of convolutions anyway.
        # Remember to do convs without torch.grad
        max_iters = torch.max(t)
        all_blurs = []
        x = x_start
        for i in range(max_iters + 1):
            with torch.no_grad():
                x = self.gaussian_kernels[i](x)
                if self.discrete:
                    if i == (self.num_timesteps - 1):
                        x = torch.mean(x, [2, 3], keepdim=True)
                        x = x.expand(
                            x_start.shape[0],
                            x_start.shape[1],
                            x_start.shape[2],
                            x_start.shape[3],
                        )
                all_blurs.append(x)

        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(t.shape[0]):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])

        choose_blur = torch.stack(choose_blur)
        if self.discrete:
            choose_blur = (choose_blur + 1) * 0.5
            choose_blur = choose_blur * 255
            choose_blur = choose_blur.int().float() / 255
            choose_blur = choose_blur * 2 - 1
        # choose_blur = all_blurs
        return choose_blur

    def p_losses(self, x_start, t):
        b, c, h, w = x_start.shape
        if self.train_routine == "Final":
            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)
            if self.loss_type == "l1":
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == "l2":
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *x.shape,
            x.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
