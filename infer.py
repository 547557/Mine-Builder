import os
import json
import atexit
import random
import datetime
from Voxelization import load_block_colors, ModelViewer
from aieditor import analyze_images_and_voxel
from txt2sc import text_to_schematic
from hunyuan import generate_model_and_image  # 假设 hunyuan.py 中有此函数

# 全局变量
BLOCK_COLORS_PATH = 'blockids.json'
API_KEY_PATH = 'api_key.json'
block_colors = load_block_colors(BLOCK_COLORS_PATH)
viewer = ModelViewer(block_colors)

# 清理临时图片文件
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

# 保存和加载 API key
def save_api_key(api_key):
    """保存 API 密钥到本地文件"""
    try:
        with open(API_KEY_PATH, 'w') as f:
            json.dump({'api_key': api_key}, f)
    except Exception as e:
        print(f"保存 API 密钥时出错：{e}")

def load_api_key():
    """从本地文件加载 API 密钥"""
    try:
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, 'r') as f:
                data = json.load(f)
                return data.get('api_key', '')
    except Exception as e:
        print(f"加载 API 密钥时出错：{e}")
    return ''

# 处理模型体素化
def process_model(glb_path):
    """处理 .glb 文件，进行体素化，返回生成的 .ply 和 .txt 文件路径"""
    if not glb_path:
        raise Exception("请提供 .glb 文件路径！")
    output_dir = os.getcwd()
    viewer.view_model(glb_path, output_dir)
    ply_path = os.path.join(output_dir, "voxel_model_32x32x32.ply")
    txt_path = os.path.join(output_dir, "voxel_model.txt")
    if not os.path.exists(ply_path):
        raise Exception("体素化模型生成失败！")
    if not os.path.exists(txt_path):
        raise Exception("体素文本生成失败！")
    return ply_path, txt_path

# AI 分析和颜色映射
def analyze_images_and_voxel_with_key(img_dir, api_key):
    """使用 API 密钥调用 analyze_images_and_voxel 函数"""
    if not api_key or not api_key.strip():
        raise Exception("请提供有效的 API 密钥！")
    save_api_key(api_key)
    from aieditor import client
    client.api_key = api_key
    try:
        result = analyze_images_and_voxel(img_dir)
        if isinstance(result, str) and (result.startswith("错误") or result.startswith("处理错误")):
            raise Exception(f"AI 分析过程出错：{result}")
        working_file = os.path.join(os.getcwd(), "working.txt")
        if not os.path.exists(working_file):
            raise Exception("配置文件生成失败！")
        return working_file
    except Exception as e:
        raise Exception(f"AI 分析过程出错：{str(e)}")

# 转换为 schematic
def convert_to_schematic(working_file, output_file):
    """将 working.txt 转换为 schematic 格式"""
    try:
        if not working_file:
            raise Exception("请先生成配置文件！")
        input_file = working_file
        result_path = text_to_schematic(input_file, output_file)
        return result_path
    except Exception as e:
        raise Exception(f"转换失败：{str(e)}")

# 主函数
def main():
    # 加载或获取 API key
    api_key = load_api_key()
    if not api_key:
        api_key = input("请输入您的 API 密钥：").strip()
        save_api_key(api_key)

    # 获取提示词
    while True:
        prompt = input("请输入提示词：").strip()
        if prompt:
            break
        print("提示词不能为空，请重新输入。")

    # 获取种子
    seed_input = input("请输入种子（可选，直接按回车随机）：").strip()
    if seed_input:
        try:
            seed = int(seed_input)
        except ValueError:
            print("种子必须是整数，将随机生成种子。")
            seed = random.randint(0, 1000000)
    else:
        seed = random.randint(0, 1000000)

    # 生成模型和图片
    print("正在生成模型和图片...")
    glb_path, image_path = generate_model_and_image(prompt, seed)
    print("模型和图片生成完毕。")

    # 体素化
    print("正在体素化...")
    ply_path, txt_path = process_model(glb_path)
    print("体素化完毕。")

    # AI 映射，处理 API key 无效的情况
    while True:
        try:
            print("正在进行 AI 映射...")
            working_file = analyze_images_and_voxel_with_key(".", api_key)
            print("AI 映射完毕。")
            break
        except Exception as e:
            if "API" in str(e) or "密钥" in str(e) or "authentication" in str(e).lower():
                print("API 密钥无效，请重新输入。")
                api_key = input("请输入您的 API 密钥：").strip()
                save_api_key(api_key)
            else:
                print(f"发生错误：{e}")
                return

    # 转换为 schematic
    print("正在转换为 schematic...")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"output_{timestamp}.schematic")
    try:
        result_path = convert_to_schematic(working_file, output_file)
        print(f"转换完毕，schematic 文件已保存到：{result_path}")
    except Exception as e:
        print(f"转换失败：{e}")

if __name__ == "__main__":
    main()
