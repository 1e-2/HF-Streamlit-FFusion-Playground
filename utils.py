import base64
import gc
import io
import os
import tempfile
import zipfile
from datetime import datetime
from threading import Thread
from huggingface_hub import Repository
import subprocess
import random
import requests
import streamlit as st
import torch
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from loguru import logger
from PIL.PngImagePlugin import PngInfo
from st_clickable_images import clickable_images


no_safety_checker = None


CODE_OF_CONDUCT = """
## Code of conduct
The app should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

Using the app to generate content that is cruel to individuals is a misuse of this app. One shall not use this app to generate content that is intended to be cruel to individuals, or to generate content that is intended to be cruel to individuals in a way that is not obvious to the viewer.
This includes, but is not limited to:
- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

By using this app, you agree to the above code of conduct.

"""


def use_auth_token():
    token_path = os.path.join(os.path.expanduser("~"), ".huggingface", "token")
    if os.path.exists(token_path):
        return True
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    return False



def download_file(file_url):
    r = requests.get(file_url, stream=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                tmp.write(chunk)
    return tmp.name


def cache_folder():
    _cache_folder = os.path.join(os.path.expanduser("~"), ".ffusion")
    os.makedirs(_cache_folder, exist_ok=True)
    return _cache_folder


def clear_memory(preserve):
    torch.cuda.empty_cache()
    gc.collect()
    to_clear = ["inpainting", "text2img", "img2text"]
    for key in to_clear:
        if key not in preserve and key in st.session_state:
            del st.session_state[key]



    import subprocess

from huggingface_hub import Repository




def save_to_hub(image, current_datetime, metadata, output_path):
    """Saves an image to Hugging Face Hub"""
    try:
        # Convert image to byte array
        byte_arr = io.BytesIO()

        # Check if the image has metadata
        if image.info:
            # Save as PNG
            image.save(byte_arr, format='PNG')
        else:
            # Save as JPG
            image.save(byte_arr, format='JPEG')

        byte_arr = byte_arr.getvalue()

        # Create a repository object
        token = os.getenv("HF_TOKEN")
        api = HfApi()
        username = "FFusion"
        repo_name = "FF"
        try:
            repo = Repository(f"{username}/{repo_name}", clone_from=f"{username}/{repo_name}", use_auth_token=token, repo_type="dataset")
        except RepositoryNotFoundError:
            repo = Repository(f"{username}/{repo_name}", clone_from=f"{username}/{repo_name}", use_auth_token=token, repo_type="dataset")

        # Pull the latest changes from the remote repository
        repo.git_pull()
        
        # Generate a random 10-digit number
        random_number = random.randint(1000000000, 9999999999)

        # Replace "0.png" in output_path with the random number
        output_path = output_path.replace("0.png", f"{random_number}.png")

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(f"{repo.local_dir}/{output_path}"), exist_ok=True)

        # Write image to repository
        with open(f"{repo.local_dir}/{output_path}", "wb") as f:
            f.write(byte_arr)

        # Set Git username and email
        subprocess.run(["git", "config", "user.name", "idle stoev"], check=True, cwd=repo.local_dir)
        subprocess.run(["git", "config", "user.email", "di@ffusion.ai"], check=True, cwd=repo.local_dir)


        # Commit and push changes
        repo.git_add(pattern=".")
        repo.git_commit(f"Add image at {current_datetime}")
        print(f"Pushing changes to {username}/{repo_name}...")
        repo.git_push()
        print(f"Image saved to {username}/{repo_name}/{output_path}")
    except Exception as e:
        print(f"Failed to save image to Hugging Face Hub: {e}")












    


def save_to_local(images, module, current_datetime, metadata, output_path):
    _metadata = PngInfo()
    _metadata.add_text("text2img", metadata)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{module}", exist_ok=True)
    os.makedirs(f"{output_path}/{module}/{current_datetime}", exist_ok=True)

    for i, img in enumerate(images):
        img.save(
            f"{output_path}/{module}/{current_datetime}/{i}.png",
            pnginfo=_metadata,
        )

    # save metadata as text file
    with open(f"{output_path}/{module}/{current_datetime}/metadata.txt", "w") as f:
        f.write(metadata)
    logger.info(f"Saved images to {output_path}/{module}/{current_datetime}")


def save_images(images, module, metadata, output_path):
    if output_path is None:
        logger.warning("No output path specified, skipping saving images")
        return

    api = HfApi()
    dset_info = None
    try:
        dset_info = api.dataset_info(output_path)
    except (HFValidationError, RepositoryNotFoundError):
        logger.warning("No valid hugging face repo. Saving locally...")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not dset_info:
        save_to_local(images, module, current_datetime, metadata, output_path)
    else:
        Thread(target=save_to_hub, args=(api, images, module, current_datetime, metadata, output_path)).start()


def display_and_download_images(output_images, metadata, download_col=None):
    # st.image(output_images, width=128, output_format="PNG")

    with st.spinner("Preparing images for download..."):
        # save images to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            gallery_images = []
            for i, image in enumerate(output_images):
                image.save(os.path.join(tmpdir, f"{i + 1}.png"), pnginfo=metadata)
                with open(os.path.join(tmpdir, f"{i + 1}.png"), "rb") as img:
                    encoded = base64.b64encode(img.read()).decode()
                    gallery_images.append(f"data:image/jpeg;base64,{encoded}")

            # zip the images
            zip_path = os.path.join(tmpdir, "images.zip")
            with zipfile.ZipFile(zip_path, "w") as zip:
                for filename in os.listdir(tmpdir):
                    if filename.endswith(".png"):
                        zip.write(os.path.join(tmpdir, filename), filename)

            # convert zip to base64
            with open(zip_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            _ = clickable_images(
                gallery_images,
                titles=[f"Image #{str(i)}" for i in range(len(gallery_images))],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                img_style={"margin": "5px", "height": "200px"},
            )

            # add download link
            st.markdown(
                f"""
                <a href="data:application/zip;base64,{encoded}" download="images.zip">
                    Download Images
                </a>
                """,
                unsafe_allow_html=True,
            )
