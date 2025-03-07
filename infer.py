#!/usr/bin/env python3
import os
import json
import atexit
import shutil
import time
import glob
import torch
from pathlib import Path
import argparse
from PIL import Image
from mmgp import offload
from Voxelization import load_block_colors, ModelViewer
from aieditor import analyze_images_and_voxel
from openai import OpenAI
from txt2sc import text_to_schematic

# Constants
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
SAVE_DIR = 'cache'  # Cache directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')  # Output directory
PROFILE = 5  # Memory optimization profile
VERBOSE = 1  # Verbosity level

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Load blockids.json globally
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

def cleanup_image_files():
    """Clean up temporary image files in the current directory"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    for filename in os.listdir(CURRENT_DIR):
        if filename.lower().endswith(image_extensions) and not filename.startswith('input_'):
            try:
                file_path = os.path.join(CURRENT_DIR, filename)
                os.remove(file_path)
                print(f"Deleted temporary image file: {filename}")
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")

# Register cleanup function to run at program exit
atexit.register(cleanup_image_files)

def save_api_key(api_key):
    """Save API key to a local file"""
    try:
        with open(API_KEY_PATH, 'w') as f:
            json.dump({'api_key': api_key}, f)
        print("API key saved")
    except Exception as e:
        print(f"Error saving API key: {e}")

def load_api_key():
    """Load API key from a local file"""
    try:
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
    except Exception as e:
        print(f"Error loading API key: {e}")
    return ''

def verify_api_key(api_key):
    """Verify if the API key is valid"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        return True
    except Exception as e:
        print(f"API key verification failed: {e}")
        return False

def gen_save_folder(max_size=60):
    """Generate a save folder"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"Removed {SAVE_DIR}/{(cur_id + 1) % max_size} successfully")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"Created {save_folder} successfully")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    """Export the mesh file"""
    if textured:
        temp_path = os.path.join(save_folder, f'textured_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'textured_mesh_{int(time.time())}.glb')
    else:
        temp_path = os.path.join(save_folder, f'white_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'white_mesh_{int(time.time())}.glb')
    
    mesh.export(temp_path, include_normals=textured)
    shutil.copy2(temp_path, output_path)
    return output_path

def setup_hunyuan_model():
    """Initialize Hunyuan model"""
    print("Loading Hunyuan model...")
    
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        has_texturegen = True
        print("Texture generator loaded successfully")
    except Exception as e:
        print(f"Failed to load texture generator: {e}")
        texgen_worker = None
        has_texturegen = False

    try:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        has_t2i = True
        print("Text-to-image model loaded successfully")
    except Exception as e:
        print(f"Failed to load text-to-image model: {e}")
        t2i_worker = None
        has_t2i = False

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)

    pipe = offload.extract_models("i23d_worker", i23d_worker)
    if has_texturegen:
        pipe.update(offload.extract_models("texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    if has_t2i:
        pipe.update(offload.extract_models("t2i_worker", t2i_worker))

    kwargs = {}
    if PROFILE < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if PROFILE != 1 and PROFILE != 3:
        kwargs["budgets"] = {"*": 2200}

    offload.profile(pipe, profile_no=PROFILE, verboseLevel=VERBOSE, **kwargs)
    
    return t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer

def generate_3d_model(prompt, seed=None, t2i_worker=None, i23d_worker=None, texgen_worker=None, 
                     rmbg_worker=None, FloaterRemover=None, DegenerateFaceRemover=None, FaceReducer=None):
    """Generate a 3D model"""
    print(f"Generating 3D model with prompt: {prompt}")
    
    if seed is None or seed == "":
        seed = int(time.time()) % 10000
        print(f"Using random seed: {seed}")
    else:
        seed = int(seed)
        print(f"Using specified seed: {seed}")
    
    save_folder = gen_save_folder()
    generator = torch.Generator().manual_seed(seed)
    
    if t2i_worker:
        print("Generating image from text...")
        try:
            image = t2i_worker(prompt)
            input_image_path = os.path.join(CURRENT_DIR, f"input_{int(time.time())}.png")
            image.save(input_image_path)
            print(f"Generated reference image saved to: {input_image_path}")
            image.save(os.path.join(save_folder, 'input.png'))
        except Exception as e:
            print(f"Failed to generate image from text: {e}")
            return None
    else:
        print("Text-to-image model not loaded, cannot generate image")
        return None
    
    print("Removing image background...")
    image = rmbg_worker(image.convert('RGB'))
    image.save(os.path.join(save_folder, 'rembg.png'))
    
    print("Generating 3D model...")
    mesh = i23d_worker(
        image=image,
        num_inference_steps=30,
        guidance_scale=5.5,
        generator=generator,
        octree_resolution=256
    )[0]
    
    print("Optimizing 3D model...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    
    if texgen_worker:
        print("Generating texture...")
        textured_mesh = texgen_worker(mesh, image)
        output_path = export_mesh(textured_mesh, save_folder, textured=True)
        print(f"Textured 3D model saved to: {output_path}")
        return output_path
    else:
        output_path = export_mesh(mesh, save_folder, textured=False)
        print(f"3D model saved to: {output_path}")
        return output_path

def process_model(glb_file):
    """Process GLB file for voxelization"""
    print(f"Processing model: {glb_file}")
    
    output_dir = os.getcwd()
    viewer.view_model(glb_file, output_dir)
    
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    
    if not os.path.exists(ply_path):
        print("Voxelization failed!")
        return None, None
    if not os.path.exists(txt_path):
        print("Voxel text generation failed!")
        return None, None
    
    print(f"Model voxelized, PLY file: {ply_path}, TXT file: {txt_path}")
    return ply_path, txt_path

def analyze_images_and_voxel_with_key(api_key):
    """Analyze images and voxel with API key"""
    if not api_key or not api_key.strip():
        print("Please provide a valid API key!")
        return None
    
    from aieditor import client
    client.api_key = api_key
    
    try:
        print("Performing AI analysis...")
        result = analyze_images_and_voxel(".")
        if isinstance(result, str) and (result.startswith("错误") or result.startswith("处理错误")):
            print(f"Error in AI analysis: {result}")
            return None
        
        working_file = os.path.join(os.getcwd(), "working.txt")
        if not os.path.exists(working_file):
            print("Configuration file generation failed!")
            return None
        
        print(f"AI analysis completed, configuration file: {working_file}")
        return working_file
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return None

def convert_to_schematic(working_file):
    """Convert working.txt to schematic format"""
    try:
        if not working_file:
            print("Please generate configuration file first!")
            return None
        
        print(f"Converting to Schematic: {working_file}")
        output_file = os.path.join(OUTPUT_DIR, f"output_{int(time.time())}.schematic")
        
        result_path = text_to_schematic(working_file, output_file)
        print(f"Conversion completed, Schematic file: {result_path}")
        return result_path
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None

def main(args):
    """Main function with command-line arguments"""
    print("==========================================")
    print("Minecraft 3D Model Generation Tool")
    print("==========================================")
    
    # Load API key
    api_key = load_api_key()
    api_valid = False
    
    while not api_valid:
        if not api_key:
            api_key = input("Please enter your OpenAI API key: ")
        
        if verify_api_key(api_key):
            api_valid = True
            save_api_key(api_key)
        else:
            print("API key is invalid, please re-enter")
            api_key = ""
    
    # Load Hunyuan model
    t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer = setup_hunyuan_model()
    
    # Get prompt and seed from command-line arguments
    prompt = args.prompt
    seed = args.seed
    
    if not prompt.strip():
        print("Prompt cannot be empty")
        return
    
    # Generate 3D model
    glb_file = generate_3d_model(
        prompt, 
        seed, 
        t2i_worker, 
        i23d_worker, 
        texgen_worker, 
        rmbg_worker, 
        FloaterRemover, 
        DegenerateFaceRemover, 
        FaceReducer
    )
    
    if not glb_file:
        print("3D model generation failed")
        return
    
    # Voxelization
    print("Starting voxelization...")
    ply_file, txt_file = process_model(glb_file)
    
    if not ply_file or not txt_file:
        print("Voxelization failed")
        return
    
    # AI color mapping
    print("Starting AI color mapping...")
    working_file = analyze_images_and_voxel_with_key(api_key)
    
    if not working_file:
        print("AI color mapping failed")
        return
    
    # Convert to Schematic
    print("Converting to Schematic...")
    schematic_file = convert_to_schematic(working_file)
    
    if not schematic_file:
        print("Schematic conversion failed")
        return
    
    print("==========================================")
    print("Processing completed!")
    print(f"Schematic file saved to: {schematic_file}")
    print("==========================================")
    
    cleanup_image_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Minecraft 3D model from a prompt.")
    parser.add_argument("-prompt", type=str, required=True, help="Prompt for generating the 3D model (e.g., 'a house')")
    parser.add_argument("-seed", type=str, default="", help="Random seed for generation (optional, default is random)")
    
    args = parser.parse_args()
    main(args)
