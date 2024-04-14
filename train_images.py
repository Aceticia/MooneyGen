import click
import rp
import torch

from torchvision.transforms.functional import gaussian_blur

import source.stable_diffusion as sd

from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
from itertools import chain
from tqdm import trange

update_presets = [
    lambda weight: dict(noise_coef=.1*weight, guidance_scale=60),
    lambda weight: dict(noise_coef=.1,image_coef=-.010,guidance_scale=50),
    lambda weight: dict(noise_coef=.1*weight, image_coef=-.005*weight, guidance_scale=50),
]


@click.command()
@click.option('--prompt',
    help='The prompt to use for binarization. Use {c} for the placeholder.',
    default='A closeup black and white photo of a {c}.'
)
@click.option('--negative_prompt', help='The negative prompt to use for binarization.', default='')
@click.option('--model_name', help='The model name to use for the stable diffusion.', default="CompVis/stable-diffusion-v1-4")
@click.option('--weights', multiple=True, help='The weights for each class. Defaults to equal weight.', default=[])
@click.option('--temperature_start', help='Start temperature for binarization.', default=5.)
@click.option('--temperature_end', help='End temperature for binarization.', default=1e5)
@click.option('--temperature_target_ratio', help='The ratio of iterations to ramp up.', default=.1)
@click.option('--recon_weight', help='The weight for weighing the reconstruction loss.', default=100.)
@click.option('--num_features', help='The number of Fourier features to use for the learnable images.', default=128)
@click.option('--hidden_dim', help='The number of hiddens to use for the mlp of learnable images.', default=256)
@click.option('--image_size', help='Size of the generated image in terms of pixels.', default=128)
@click.option('--use_gpu', is_flag=True, help='Use GPU for training.')
@click.option('--use_mps', is_flag=True, help='Use MPS for training (mac).')
@click.option('--output_dir', help='The output directory to save the images.', default='./output', type=click.Path(exists=False))
@click.option('--iters', help='The number of iterations to train the images.', default=10000)
@click.option('--max_step', help='Maximum number of steps for dream target loss', default=990)
@click.option('--min_step', help='Minimum number of steps for dream target loss', default=10)
@click.option('--loss_preset', help='Preset for dream target loss.', default=0)
@click.option('--blurring_kernel_size', help='Kernel size for blurring grayscale images', default=5)
@click.option('--blurring_kernel_sigma', help='Kernel sigma for blurring grayscale images', default=1.5)
@click.argument('classes', nargs=-1)
def train_images(
        prompt, negative_prompt, classes, recon_weight, weights, model_name, image_size,
        hidden_dim, num_features, use_gpu, output_dir, max_step, min_step, 
        temperature_start, temperature_end, temperature_target_ratio,
        iters, loss_preset, blurring_kernel_size, blurring_kernel_sigma, use_mps
    ):
    if len(weights) != 0:
        assert len(weights) == len(classes), 'The number of weights must match the number of classes.'
    else:
        weights = [1] * len(classes)
    weights=rp.as_numpy_array(weights)
    weights=weights/weights.sum()
    weights=weights*len(weights)

    # First import the models
    assert not (use_gpu and use_mps), 'Cannot use both GPU and MPS.'
    device = 'cpu'
    if use_gpu:
        device = 'cuda'
    elif use_mps:
        device = 'mps'

    s = sd.StableDiffusion(device, model_name)
    device = s.device

    # Then make the prompts
    prompts = [prompt.format(c=class_name) for class_name in classes]
    negative_prompts = [negative_prompt.format(c=class_name) if len(negative_prompt) else "" for class_name in classes ]

    labels = [NegativeLabel(p1, p2) for p1, p2 in zip(prompts, negative_prompts)]

    # Make the learnable images
    images = [LearnableImageFourier(
        height=image_size, width=image_size, num_features=num_features, hidden_dim=hidden_dim, num_channels=1
    ).to(device) for _ in range(len(classes))]

    params = chain(*[image.parameters() for image in images])
    optim = torch.optim.SGD(params,lr=1e-4)

    s.max_step = max_step
    s.min_step = min_step
    preset = update_presets[loss_preset]

    temps = torch.linspace(temperature_start, temperature_end, int(temperature_target_ratio * iters))

    for iter in trange(iters):
        # Ramp up the temperature
        if iter < int(temperature_target_ratio * iters):
            curr_temp = temps[iter]
        else:
            curr_temp = temperature_end

        tensor_images = torch.stack([image() for image in images], dim=0)
        for image, label, weight in zip(tensor_images, labels, weights):
            s.train_step(
                label.embedding,
                image[None].repeat(1, 3, 1, 1),
                **preset(weight),
            )

        recon_loss = s.binarize(
            gaussian_blur(
                tensor_images,
                kernel_size=blurring_kernel_size,
                sigma=blurring_kernel_sigma
            ),
            temperature=curr_temp,
            max_intensity=1.
        ) * recon_weight
        recon_loss.backward()

        optim.step()
        optim.zero_grad()

    # Save the images
    for label, image in zip(classes, images):
        rp.save_image(rp.as_numpy_image(image().repeat(3, 1, 1)), output_dir + f'/{label}.png', False)

    # Save the mooney image too
    mooney_img = 0
    for image in images:
        mooney_img += image() / len(images)
    mooney_img = (mooney_img > 0.5).float()
    rp.save_image(rp.as_numpy_image(mooney_img.repeat(3, 1, 1)), output_dir + '/mooney.png', False)

if __name__ == '__main__':
    train_images()