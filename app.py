# gradio_interface.py
import gradio as gr
import os
import json
import atexit
import shutil
import time
from Voxelization import load_block_colors, ModelViewer
from aieditor import analyze_images_and_voxel
from openai import OpenAI
from txt2sc import text_to_schematic
from hunyuan import generate_shape_and_texture, generate_shape_only, HAS_TEXTUREGEN

# 全局加载 blockids.json，避免重复加载
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

def cleanup_image_files():
    """清理当前目录下的临时图片文件"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    current_dir = os.getcwd()
    for filename in os.listdir(current_dir):
        if filename.lower().endswith(image_extensions):
            try:
                file_path = os.path.join(current_dir, filename)
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

def process_hunyuan(text_prompt):
    """处理混元生成3D模型的流程"""
    try:
        if not text_prompt or not text_prompt.strip():
            raise gr.Error("请输入文本提示词！")

        # 生成3D模型
        if HAS_TEXTUREGEN:
            path, path_textured, _, _ = generate_shape_and_texture(
                caption=text_prompt,
                steps=30,
                guidance_scale=5.5,
                seed=1234,
                octree_resolution=256,
                remove_background=True
            )
            # 复制生成的图片到当前目录
            current_dir = os.getcwd()
            for file in os.listdir('gradio_cache'):
                if file.endswith('.png'):
                    src_path = os.path.join('gradio_cache', file)
                    dst_path = os.path.join(current_dir, file)
                    shutil.copy2(src_path, dst_path)

            # 自动处理体素化
            ply_path, txt_path = process_model(path_textured)
            
            # 自动进行AI颜色映射
            working_file = analyze_images_and_voxel_with_key(None, load_api_key())
            
            # 转换为schematic
            schematic_path = convert_to_schematic(working_file)
            
            return path_textured, ply_path, txt_path, working_file, schematic_path
        else:
            path, _ = generate_shape_only(
                caption=text_prompt,
                steps=30,
                guidance_scale=5.5,
                seed=1234,
                octree_resolution=256,
                remove_background=True
            )
            # 自动处理体素化
            ply_path, txt_path = process_model(path)
            
            # 自动进行AI颜色映射
            working_file = analyze_images_and_voxel_with_key(None, load_api_key())
            
            # 转换为schematic
            schematic_path = convert_to_schematic(working_file)
            
            return path, ply_path, txt_path, working_file, schematic_path
    except Exception as e:
        raise gr.Error(f"生成过程出错：{str(e)}")

# 定义处理API密钥的函数
def analyze_images_and_voxel_with_key(img, api_key):
    """使用API密钥调用analyze_images_and_voxel函数"""
    if not api_key or not api_key.strip():
        raise gr.Error("请提供有效的API密钥！")
    
    # 保存新的API密钥
    save_api_key(api_key)
    
    # 设置OpenAI客户端的API密钥
    from aieditor import client
    client.api_key = api_key
    
    # 确定图片目录
    img_dir = "."  # 默认当前目录
    if img is not None and hasattr(img, 'name'):
        img_dir = os.path.dirname(img.name)
    
    # 调用分析函数
    try:
        result = analyze_images_and_voxel(img_dir)
        if isinstance(result, str) and (result.startswith("错误") or result.startswith("处理错误")):
            raise gr.Error(f"AI分析过程出错：{result}")
        
        # 检查working.txt是否生成
        working_file = os.path.join(os.getcwd(), "working.txt")
        if not os.path.exists(working_file):
            raise gr.Error("配置文件生成失败！")
        
        return working_file
    except gr.Error as e:
        raise e
    except Exception as e:
        error_msg = str(e)
        raise gr.Error(f"AI分析过程出错：{error_msg}")

def convert_to_schematic(working_file):
    """将working.txt转换为schematic格式"""
    try:
        if not working_file:
            raise gr.Error("请先生成配置文件！")
        
        input_file = working_file.name if hasattr(working_file, 'name') else working_file
        output_file = os.path.join(os.path.dirname(input_file), "output.schematic")
        
        result_path = text_to_schematic(input_file, output_file)
        # 转换完成后清理图片文件
        cleanup_image_files()
        return result_path
    except Exception as e:
        raise gr.Error(f"转换失败：{str(e)}")

def process_model(glb_file):
    """处理上传的 .glb 文件，进行体素化，并返回生成的 .ply 和 .txt 文件路径。"""
    if not glb_file:
        raise gr.Error("请上传一个 .glb 文件！")
    
    # 获取上传文件的路径
    glb_path = glb_file.name if hasattr(glb_file, 'name') else glb_file
    
    # 使用当前工作目录作为输出目录
    output_dir = os.getcwd()
    
    # 调用 ModelViewer 的 view_model 方法处理模型
    viewer.view_model(glb_path, output_dir)
    
    # 构造输出文件路径
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    
    # 检查文件是否生成成功
    if not os.path.exists(ply_path):
        raise gr.Error("体素化模型生成失败！")
    if not os.path.exists(txt_path):
        raise gr.Error("体素文本生成失败！")
    
    return ply_path, txt_path

# 定义 Gradio 接口
with gr.Blocks() as app:
    gr.Markdown("## Minecraft建模工具套件")
    
    # 加载保存的API密钥
    saved_api_key = load_api_key()
    
    with gr.Tabs():
        with gr.TabItem("混元生成3D模型"):
            with gr.Row():
                text_input = gr.Textbox(label="输入文本提示词", placeholder="请输入描述性文本，例如：一个可爱的猫咪")
            with gr.Row():
                glb_output = gr.File(label="生成的GLB模型")
                ply_output_auto = gr.File(label="体素模型输出")
                txt_output_auto = gr.File(label="体素文本输出")
            with gr.Row():
                working_file_output_auto = gr.File(label="生成配置文件")
                schematic_output_auto = gr.File(label="Schematic输出")
            gr.Button("开始生成").click(
                fn=process_hunyuan,
                inputs=[text_input],
                outputs=[glb_output, ply_output_auto, txt_output_auto, working_file_output_auto, schematic_output_auto]
            )

        with gr.TabItem("模型体素化工具"):
            with gr.Row():
                glb_input = gr.File(label="上传.glb模型文件", file_types=[".glb"])
                ply_output = gr.File(label="体素模型输出")
                txt_output = gr.File(label="体素文本输出")
            gr.Button("开始处理").click(
                fn=process_model,
                inputs=[glb_input],
                outputs=[ply_output, txt_output]
            )

        with gr.TabItem("AI颜色映射工具"):
            with gr.Row():
                image_input = gr.File(label="上传参考图片（可选）", file_types=["image"])
                api_key_input = gr.Textbox(label="API密钥", type="password", value=saved_api_key)
                working_file_output = gr.File(label="生成配置文件")
            with gr.Row():
                analyze_btn = gr.Button("执行分析")
                convert_btn = gr.Button("转换为Schematic")
                schematic_output = gr.File(label="Schematic输出")
            
            analyze_btn.click(
                fn=lambda img, api_key: analyze_images_and_voxel_with_key(img, api_key),
                inputs=[image_input, api_key_input],
                outputs=[working_file_output]
            )
            
            convert_btn.click(
                fn=convert_to_schematic,
                inputs=[working_file_output],
                outputs=[schematic_output]
            )

if __name__ == "__main__":
    app.launch()