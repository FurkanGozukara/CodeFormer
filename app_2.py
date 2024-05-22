"""
This file is used for deploying hugging face demo:
https://huggingface.co/spaces/sczhou/CodeFormer
"""

import sys
sys.path.append('CodeFormer')
import os
import cv2
import torch
import torch.nn.functional as F
import gradio as gr
from itertools import chain

from torchvision.transforms.functional import normalize
import os
import platform
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer

from basicsr.utils.registry import ARCH_REGISTRY


def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

#os.system("pip freeze")

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
}
# download weights
if not os.path.exists('CodeFormer/weights/CodeFormer/codeformer.pth'):
    load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='CodeFormer/weights/CodeFormer', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
    load_file_from_url(url=pretrain_model_url['detection'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/parsing_parsenet.pth'):
    load_file_from_url(url=pretrain_model_url['parsing'], model_dir='CodeFormer/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir='CodeFormer/weights/realesrgan', progress=True, file_name=None)

# download images
torch.hub.download_url_to_file(
    'https://replicate.com/api/models/sczhou/codeformer/files/fa3fe3d1-76b0-4ca8-ac0d-0a925cb0ff54/06.png',
    '01.png')
torch.hub.download_url_to_file(
    'https://replicate.com/api/models/sczhou/codeformer/files/a1daba8e-af14-4b00-86a4-69cec9619b53/04.jpg',
    '02.jpg')
torch.hub.download_url_to_file(
    'https://replicate.com/api/models/sczhou/codeformer/files/542d64f9-1712-4de7-85f7-3863009a7c3d/03.jpg',
    '03.jpg')
torch.hub.download_url_to_file(
    'https://replicate.com/api/models/sczhou/codeformer/files/a11098b0-a18a-4c02-a19a-9a7045d68426/010.jpg',
    '04.jpg')
torch.hub.download_url_to_file(
    'https://replicate.com/api/models/sczhou/codeformer/files/7cf19c2c-e0cf-4712-9af8-cf5bdbb8d0ee/012.jpg',
    '05.jpg')

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

upsampler = set_realesrgan()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

os.makedirs('outputs', exist_ok=True)

def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """Run a single prediction on the model"""
    try: # global try
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"
        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)
        
        if background_enhance is None: background_enhance = True
        if face_upsample is None: face_upsample = True
        if upscale is None: upscale = 2

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        print('\timage size:', img.shape)

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )


        save_image(restored_img)

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None, None


def save_image(restored_img):
    # Set the base filename and extension
    base_filename = "img_"
    extension = ".png"
    
    # Set the output directory
    output_dir = "outputs"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find the last image file in the output directory
    image_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith(extension)]
    
    if image_files:
        # Extract the numbers from the image filenames
        numbers = [int(f[len(base_filename):-len(extension)]) for f in image_files]
        
        # Find the maximum number
        max_number = max(numbers)
        
        # Increment the number by 1 for the new filename
        new_number = max_number + 1
    else:
        # If no image files found, start with number 1
        new_number = 1
    
    # Create the new filename with padded zeros
    new_filename = f"{base_filename}{new_number:04d}{extension}"
    
    # Create the full save path
    save_path = os.path.join(output_dir, new_filename)
    
    # Save the restored image
    imwrite(restored_img, save_path)
    
    print(f"Image saved as {save_path}")

title = "CodeFormer: Robust Face Restoration and Enhancement Network - V1 - APP 2"

description = r"""Modified from sczhou/CodeFormer - Latest version on https://www.patreon.com/posts/104691847
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Input", height=512)
            background_enhance = gr.Checkbox(value=True, label="Background Enhance")
            face_enhance = gr.Checkbox(value=True, label="Face Upsample")
            upscale_factor = gr.Number(value=2, label="Rescaling Factor (up to 4)")
            codeformer_fidelity = gr.Slider(0, 1, value=0.5, step=0.01, label='Codeformer Fidelity (0 for better quality, 1 for better identity)')
            with gr.Row():
                submit_btn = gr.Button('Enhance Image')
                clear_btn = gr.Button('Clear')
        with gr.Column():
            output_img = gr.Image(type="filepath", label="Output", format="png")
            btn_open_outputs = gr.Button("Open Outputs Folder")

    def clear_inputs():
        return None, True, True, 2, 0.5

    inps = [input_img, background_enhance, face_enhance, upscale_factor, codeformer_fidelity]
    submit_btn.click(fn=inference, inputs=inps, outputs=[output_img])
    clear_btn.click(fn=clear_inputs, outputs=inps)
    btn_open_outputs.click(fn=open_folder)



demo.launch(inbrowser=True)