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
from aieditor import analyze_images_and_voxel, client  # Import client directly from aieditor
from openai import OpenAI  # Still needed for compatibility
from txt2sc import text_to_schematic

# 常量定义
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
SAVE_DIR = 'cache'  # 缓存目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__ in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output')  # 输出目录
PROFILE = 5  # 内存优化配置
VERBOSE = 1  # 详细程度

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局加载 blockids.json
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

def cleanup_image_files():
    """清理当前目录下的临时图片文件"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    for filename in os.listdir(CURRENT_DIR):
        if filename.lower().endswith(image_extensions) and not filename.startswith('input_'):
            try:
                file_path = os.path.join(CURRENT_DIR, filename)
                os.remove(file_path)
                print(f"已删除临时图片文件：{filename}")
            except Exception as e:
                print(f"删除文件 {filename} 时出错：{e}")

# 注册程序退出时的清理函数
atexit.register(cleanup_image_files)

def save_api_key(api_key):
    """保存API密钥到本地文件"""
    try:
        with open(API_KEY_PATH, 'w') as f:
            json.dump({'api_key': api_key}, f)
        print("API密钥已保存")
    except Exception as e:
        print(f"保存API密钥时出错：{e}")

def load_api_key():
    """从本地文件加载API密钥"""
    try:
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
    except Exception as e:
        print(f"加载API密钥时出错：{e}")
    return ''

def verify_api_key(api_key):
    """验证Gemini API密钥是否有效"""
    try:
        # 初始化客户端，使用Gemini API的base_url
        test_client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
        # 测试API密钥有效性，使用一个简单的chat请求
        response = test_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        print("Gemini API密钥验证成功")
        return True
    except Exception as e:
        print(f"Gemini API密钥验证失败: {e}")
        return False

def gen_save_folder(max_size=60):
    """生成保存文件夹"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"移除 {SAVE_DIR}/{(cur_id + 1) % max_size} 成功")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"创建 {save_folder} 成功")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    """导出模型文件"""
    if textured:
        temp_path = os.path.join(save_folder, 'textured_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'textured_mesh_{int(time.time())}.glb')
    else:
        temp_path = os.path.join(save_folder, 'white_mesh.glb')
        output_path = os.path.join(OUTPUT_DIR, f'white_mesh_{int(time.time())}.glb')
    
    # 导出到临时位置
    mesh.export(temp_path, include_normals=textured)
    # 复制到输出目录
    shutil.copy2(temp_path, output_path)
    return output_path

def setup_hunyuan_model():
    """初始化Hunyuan模型"""
    print("正在加载Hunyuan模型...")
    
    # 尝试加载纹理生成器
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        has_texturegen = True
        print("纹理生成器加载成功")
    except Exception as e:
        print(f"加载纹理生成器失败: {e}")
        texgen_worker = None
        has_texturegen = False

    # 尝试加载文本到图像模型
    try:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        has_t2i = True
        print("文本到图像模型加载成功")
    except Exception as e:
        print(f"加载文本到图像模型失败: {e}")
        t2i_worker = None
        has_t2i = False

    # 加载必要的模型
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)

    # 内存优化
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
    """生成3D模型"""
    print(f"正在生成3D模型，提示词: {prompt}")
    
    # 设置随机种子
    if seed is None or seed == "":
        seed = int(time.time()) % 10000
        print(f"使用随机种子: {seed}")
    else:
        seed = int(seed)
        print(f"使用指定种子: {seed}")
    
    save_folder = gen_save_folder()
    generator = torch.Generator().manual_seed(seed)
    
    # 从文本生成图像
    if t2i_worker:
        print("正在从文本生成图像...")
        try:
            image = t2i_worker(prompt)
            # 保存输入图像供AI参考
            input_image_path = os.path.join(CURRENT_DIR, f"input_{int(time.time())}.png")
            image.save(input_image_path)
            print(f"生成的参考图像已保存到: {input_image_path}")
            image.save(os.path.join(save_folder, 'input.png'))
        except Exception as e:
            print(f"文本生成图像失败: {e}")
            return None
    else:
        print("文本到图像模型未加载，无法生成图像")
        return None
    
    # 移除背景
    print("正在移除图像背景...")
    image = rmbg_worker(image.convert('RGB'))
    image.save(os.path.join(save_folder, 'rembg.png'))
    
    # 生成3D模型
    print("正在生成3D模型...")
    mesh = i23d_worker(
        image=image,
        num_inference_steps=30,
        guidance_scale=5.5,
        generator=generator,
        octree_resolution=256
    )[0]
    
    # 优化模型
    print("正在优化3D模型...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    
    # 生成纹理
    if texgen_worker:
        print("正在生成纹理...")
        textured_mesh = texgen_worker(mesh, image)
        output_path = export_mesh(textured_mesh, save_folder, textured=True)
        print(f"带纹理的3D模型已保存到: {output_path}")
        return output_path
    else:
        output_path = export_mesh(mesh, save_folder, textured=False)
        print(f"3D模型已保存到: {output_path}")
        return output_path

def process_model(glb_file):
    """处理GLB文件，进行体素化"""
    print(f"正在处理模型: {glb_file}")
    
    # 使用当前工作目录作为输出目录
    output_dir = os.getcwd()
    
    # 调用ModelViewer的view_model方法处理模型
    viewer.view_model(glb_file, output_dir)
    
    # 构造输出文件路径
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    
    # 检查文件是否生成成功
    if not os.path.exists(ply_path):
        print("体素化模型生成失败！")
        return None, None
    if not os.path.exists(txt_path):
        print("体素文本生成失败！")
        return None, None
    
    print(f"模型已体素化，PLY文件: {ply_path}, TXT文件: {txt_path}")
    return ply_path, txt_path

def analyze_images_and_voxel_with_key(api_key):
    """使用API密钥调用analyze_images_and_voxel函数"""
    if not api_key or not api_key.strip():
        print("请提供有效的Gemini API密钥！")
        return None
    
    # 设置aieditor中的全局client的API密钥
    client.api_key = api_key
    # client.base_url已在aieditor.py中设置为Gemini API的URL，无需再次设置
    
    # 调用分析函数
    try:
        print("正在进行AI分析...")
        result = analyze_images_and_voxel(".")
        if isinstance(result, str) and (result.startswith("错误") or result.startswith("处理错误")):
            print(f"AI分析过程出错：{result}")
            return None
        
        # 检查working.txt是否生成
        working_file = os.path.join(os.getcwd(), "working.txt")
        if not os.path.exists(working_file):
            print("配置文件生成失败！")
            return None
        
        print(f"AI分析完成，配置文件: {working_file}")
        return working_file
    except Exception as e:
        print(f"AI分析过程出错：{str(e)}")
        return None

def convert_to_schematic(working_file):
    """将working.txt转换为schematic格式"""
    try:
        if not working_file:
            print("请先生成配置文件！")
            return None
        
        print(f"正在转换为Schematic: {working_file}")
        output_file = os.path.join(OUTPUT_DIR, f"output_{int(time.time())}.schematic")
        
        result_path = text_to_schematic(working_file, output_file)
        print(f"转换完成，Schematic文件: {result_path}")
        return result_path
    except Exception as e:
        print(f"转换失败：{str(e)}")
        return None

def main(args):
    """主函数"""
    print("==========================================")
    print("Minecraft 3D模型生成工具")
    print("==========================================")
    
    # 从命令行参数获取值
    prompt = args.prompt
    seed = args.seed
    api_key = args.key if args.key else load_api_key()

    # 验证提示词
    if not prompt or not prompt.strip():
        print("错误：提示词不能为空")
        return

    # 验证API密钥
    api_valid = False
    while not api_valid:
        if not api_key:
            print("错误：未提供Gemini API密钥，且本地未找到有效密钥")
            return
        
        if verify_api_key(api_key):
            api_valid = True
            save_api_key(api_key)
        else:
            print("错误：Gemini API密钥无效")
            return
    
    # 加载Hunyuan模型
    t2i_worker, i23d_worker, texgen_worker, rmbg_worker, FloaterRemover, DegenerateFaceRemover, FaceReducer = setup_hunyuan_model()
    
    # 生成3D模型
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
        print("3D模型生成失败")
        return
    
    # 体素化处理
    print("开始体素化处理...")
    ply_file, txt_file = process_model(glb_file)
    
    if not ply_file or not txt_file:
        print("体素化失败")
        return
    
    # AI颜色映射
    print("开始AI颜色映射...")
    working_file = analyze_images_and_voxel_with_key(api_key)
    
    if not working_file:
        print("AI颜色映射失败")
        return
    
    # 转换为Schematic
    print("转换为Schematic...")
    schematic_file = convert_to_schematic(working_file)
    
    if not schematic_file:
        print("Schematic转换失败")
        return
    
    print("==========================================")
    print("处理完成！")
    print(f"Schematic文件已保存到: {schematic_file}")
    print("==========================================")
    
    # 清理临时图片文件
    cleanup_image_files()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Minecraft 3D模型生成工具")
    parser.add_argument("-prompt", type=str, required=True, help="生成3D模型的提示词")
    parser.add_argument("-seed", type=str, default="", help="随机种子（可选，留空则随机生成）")
    parser.add_argument("-key", type=str, default="", help="Gemini API密钥（可选，若未提供则尝试加载本地密钥）")
    
    # 解析参数并运行主函数
    args = parser.parse_args()
    main(args)
