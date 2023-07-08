import datetime
import os
import re
import gc
import json
import time
import base64
import io
import tempfile
import zipfile
import PIL
import subprocess
from huggingface_hub import Repository
from utils import save_to_hub, save_to_local
from dataclasses import dataclass
from io import BytesIO
def sanitize_filename(filename):
    """Sanitizes a filename by replacing special characters with underscores"""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

from typing import Optional, Literal, Union
from diffusers import (DiffusionPipeline, DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                       LMSDiscreteScheduler, EulerDiscreteScheduler, 
                       EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, 
                       DPMSolverSinglestepScheduler)

AVAILABLE_SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler,
    "PNDM": PNDMScheduler,
    "LMS Discrete": LMSDiscreteScheduler,
    "Euler Discrete": EulerDiscreteScheduler,
    "Euler Ancestral Discrete": EulerAncestralDiscreteScheduler,
    "DPM Solver Multistep": DPMSolverMultistepScheduler,
    "DPM Solver Singlestep": DPMSolverSinglestepScheduler,
}
HF_TOKEN = os.environ.get("HF_TOKEN")
import streamlit as st
st.set_page_config(layout="wide")
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from datetime import datetime
from threading import Thread

import requests

from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from loguru import logger
from PIL.PngImagePlugin import PngInfo
from st_clickable_images import clickable_images

import streamlit.components.v1 as components


prefix = 'image_generation'

def dict_to_style(d):
    return ';'.join(f'{k}:{v}' for k, v in d.items())

def clickable_images(images, titles, div_style={}, img_style={}):
    """Generates a component with clickable images"""
    img_tag = "".join(
        f'<a href="{img}" target="_blank"><img src="{img}" title="{title}" style="{dict_to_style(img_style)}"></a>'
        for img, title in zip(images, titles)
    )
    return components.html(f'<div style="{dict_to_style(div_style)}">{img_tag}</div>', scrolling=True)

def display_and_download_images(output_images, metadata):
    with st.spinner("Preparing images..."):
        # save images to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            gallery_images = []
            for i, image in enumerate(output_images):
                image.save(os.path.join(tmpdir, f"{i + 1}.png"), pnginfo=metadata)
                with open(os.path.join(tmpdir, f"{i + 1}.png"), "rb") as img:
                    encoded = base64.b64encode(img.read()).decode()
                    gallery_images.append(f"data:image/png;base64,{encoded}")

            _ = clickable_images(
                gallery_images,
                titles=[f"Image #{str(i + 1)}" for i in range(len(gallery_images))],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                img_style={"margin": "5px", "height": "200px"},
            )

            
PIPELINE_NAMES = Literal["txt2img", "inpaint", "img2img"]

DEFAULT_PROMPT = "sprinkled donut sitting on top of a purple cherry apple, colorful hyperrealism"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"



def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


@st.cache_resource(max_entries=1)
def get_pipeline(
    name: str,
    scheduler_name: str = None,
) -> Union[
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
]:
    if name in ["txt2img", "img2img"]:
        model_id = "FFusion/FFusion-BaSE"
        
        pipeline = DiffusionPipeline.from_pretrained(model_id)

        # Use specified scheduler if provided, else use DDIMScheduler
        if scheduler_name:
            SchedulerClass = AVAILABLE_SCHEDULERS[scheduler_name]
            pipeline.scheduler = SchedulerClass.from_config(
                pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
            )
        else:
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
            )
            
        pipeline = pipeline.to("cuda")
        return pipeline


def generate(
    prompt,
    pipeline_name: PIPELINE_NAMES,
    num_images=1,
    negative_prompt=None,
    steps=22,
    width=896,
    height=1024,
    guidance_scale=6,
    enable_attention_slicing=True,
    enable_xformers=True
):
    """Generates an image based on the given prompt and pipeline name"""
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)
    callback = lambda step, *_: p.progress(step / steps)

    pipe = get_pipeline(pipeline_name)
    torch.cuda.empty_cache()

    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    if enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        callback=callback,
        guidance_scale=guidance_scale,
        guidance_rescale=0.7
    )

    if pipeline_name == "txt2img":
        kwargs.update(width=width, height=height)

    elif pipeline_name in ["inpaint", "img2img"]:
        kwargs.update(image_input=image_input)

    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    # Save images to Hugging Face Hub or locally
    current_datetime = datetime.now()
    metadata = {
        "prompt": prompt,
        "timestamp": str(current_datetime),
    }

    output_images = []  # list to hold output image objects
    for _ in range(num_images):  # loop over number of images
        result = pipe(**kwargs)  # generate one image at a time
        images = result.images
        for i, image in enumerate(images):  # loop over each image
            filename = (
                "/data/"
                + sanitize_filename(re.sub(r"\s+", "_", prompt)[:50])
                + f"_{i}_{datetime.now().timestamp()}"
            )
            image.save(f"{filename}.png")
            output_images.append(image)  # add the image object to the list

            # Save image to Hugging Face Hub
            output_path = f"images/{i}.png"
            save_to_hub(image, current_datetime, metadata, output_path)

    for image in output_images:
        with open(f"{filename}.txt", "w") as f:
            f.write(prompt)

    return output_images  # return the list of image objects






def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    negative_prompt = st.text_area(
        "Negative prompt",
        value="(disfigured), bad quality, ((bad art)), ((deformed)), ((extra limbs)), (((duplicate))), ((morbid)), (((ugly)), blurry, ((bad anatomy)), (((bad proportions))), cloned face, body out of frame, out of frame, bad anatomy, gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck))), Deformed, blurry",
        key=f"{prefix}-negative-prompt",
    )
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Number of inference steps", min_value=11, max_value=69, value=14, key=f"{prefix}-inference-steps")
    with col2:
        guidance_scale = st.slider(
            "Guidance scale", min_value=0.0, max_value=20.0, value=7.5, step=0.5, key=f"{prefix}-guidance-scale"
        )
    # Add a select box for the schedulers
    scheduler_name = st.selectbox(
        "Choose a Scheduler",
        options=list(AVAILABLE_SCHEDULERS.keys()),
        index=0,  # Default index
        key=f"{prefix}-scheduler",
    )
    scheduler_class = AVAILABLE_SCHEDULERS[scheduler_name]  # Get the selected scheduler class


    pipe = get_pipeline(pipeline_name, scheduler_name=scheduler_name)        
    
   # enable_attention_slicing = st.checkbox('Enable attention slicing (enables higher resolutions but is slower)', key=f"{prefix}-attention-slicing", value=True)
   # enable_xformers = st.checkbox('Enable xformers library (better memory usage)', key=f"{prefix}-xformers", value=True)
    num_images = st.slider("Number of images to generate", min_value=1, max_value=4, value=1, key=f"{prefix}-num-images")

    images = []

    
    if st.button("Generate images", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            images = generate(
                prompt,
                pipeline_name,
                num_images=num_images,  # add this
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                enable_attention_slicing=True,  # value always set to True
                enable_xformers=True,  # value always set to True
                **kwargs,
            )

    for i, image in enumerate(images):  # loop over each image
        set_image(f"{OUTPUT_IMAGE_KEY}_{i}", image.copy())  # save each image with a unique key


    image_indices = [int(key.split('_')[-1]) for key in st.session_state.keys() if OUTPUT_IMAGE_KEY in key]
    cols = st.columns(len(image_indices) if image_indices else 1)  # create a column for each image or a single one if no images
    for i in range(max(image_indices) + 1 if image_indices else 1):  # loop over each image index
        output_image_key = f"{OUTPUT_IMAGE_KEY}_{i}"
        output_image = get_image(output_image_key)
        if output_image:
            cols[i].image(output_image)



def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=768,
            max_value=1024,
            step=128,
            value=768,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=768,
            max_value=1024,
            step=128,
            value=768,
            key=f"{prefix}-height",
        )
    return width, height
    
data_dir = "/data"  # Update with the correct path

# Get all file names in the data directory
file_names = os.listdir(data_dir)


def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    prompt_and_generate_button(prefix, "txt2img", width=width, height=height)


def inpainting_tab():
    col1, col2 = st.columns(2)

    with col1:
        image_input, mask_input = inpainting()

    with col2:
        if image_input and mask_input:
            prompt_and_generate_button(
                "inpaint", "inpaint", image_input=image_input, mask_input=mask_input
            )


def img2img_tab():
    col1, col2 = st.columns(2)

    with col1:
        image = image_uploader("img2img")
        if image:
            st.image(image)

    with col2:
        if image:
            prompt_and_generate_button("img2img", "img2img", image_input=image)

def main():
    st.title("FFusion AI -beta- Playground")

    tabs = ["FFusion BaSE 768+ (txt2img)"]
    selected_tab = st.selectbox("Choose a di.FFusion.ai model", tabs)

    if selected_tab == "FFusion BaSE 768+ (txt2img)":
        txt2img_tab()

    st.header("Citation")

    """
    ```
        @misc {ffusion_ai_2023,
    	author       = { {FFusion AI} },
    	title        = { FFusion-BaSE (Revision ba72848) },
    	year         = 2023,
    	url          = { https://huggingface.co/FFusion/FFusion-BaSE },
    	doi          = { 10.57967/hf/0851 },
    	publisher    = { Hugging Face }
        } http://doi.org/10.57967/hf/0851
    ```
    """

    """
        Please note that the demo is intended for academic and research purposes ONLY. Any use of the demo for generating inappropriate content is strictly prohibited. The responsibility for any misuse or inappropriate use of the demo lies solely with the users who generated such content, and this demo shall not be held liable for any such use. By interacting within this environment, you hereby acknowledge and agree to the terms of the CreativeML Open RAIL-M License.
    """
         
if __name__ == "__main__":
    main()
