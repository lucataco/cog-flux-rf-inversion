# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List
from transformers import CLIPImageProcessor
from pipeline_flux_rf_inversion import RFInversionFluxPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "FLUX.1-dev"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE,
            torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Flux txt2img Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, '.')
        self.pipe = RFInversionFluxPipeline.from_pretrained(
            MODEL_CACHE, 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("setup took: ", time.time() - start)

    @torch.amp.autocast('cuda')
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

    @staticmethod
    def make_multiple_of_16(n):
        return ((n + 15) // 16) * 16

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for generated image",
            default="Portrait of a tiger",
        ),
        image: Path = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image",
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,le=50, default=28,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80, ge=0, le=100,
        ),
        inversion_gamma: float = Input(
            description="Inversion gamma",
            default=0.5, ge=0.1, le=1.0,
        ),
        start_timestep: float = Input(
            description="Start timestep",
            default=0.0, ge=0.0, le=1.0,
        ),
        stop_timestep: float = Input(
            description="Stop timestep",
            default=0.38, ge=0.0, le=1.0,
        ),
        eta: float = Input(
            description="The controller guidance. higher eta - better faithfullness, less editability. For more significant edits, lower the value of eta",
            default=0.9, ge=0.1, le=1.0,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Load and preprocess the image while maintaining aspect ratio
        pil_image = Image.open(str(image))
        original_width, original_height = pil_image.size
        # Calculate the target size while maintaining aspect ratio
        # and ensuring dimensions are multiples of the VAE scale factor
        target_size = 1024  # Base target size
        if original_width > original_height:
            new_width = target_size
            new_height = int((target_size * original_height) / original_width)
        else:
            new_height = target_size
            new_width = int((target_size * original_width) / original_height)
        
        # Ensure dimensions are multiples of VAE scale factor1
        new_width = self.make_multiple_of_16(new_width)
        new_height = self.make_multiple_of_16(new_height)
        # Resize image maintaining aspect ratio
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        pil_image = pil_image.convert("RGB")

        print(f"Prompt: {prompt}")
        pipe = self.pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        inverted_latents, image_latents, latent_image_ids = pipe.invert(
            image=pil_image, 
            num_inversion_steps=num_inference_steps,
            gamma=inversion_gamma
        )

        output = pipe(
            prompt=prompt, 
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            start_timestep=start_timestep, 
            stop_timestep=stop_timestep,
            num_inference_steps=num_inference_steps,
            eta=eta, 
            generator=generator,
        )

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker and has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception("NSFW content detected. Try running it again, or try a different prompt.")

        return output_paths
    