
import os
import json
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import time
import traceback

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/"  # Gemini API的基本URL
)

# 文件读取函数
def read_file(filename: str):
    """读取指定文件内容，如果文件不存在则报错"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found!")
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# 函数调用工具：回答颜色对应的方块ID
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_block_id",
            "description": "根据给定的颜色，返回对应的Minecraft方块ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string", "description": "体素文件中的颜色名称"},
                    "block_id": {"type": "string", "description": "对应的Minecraft方块ID (如 '1' 或 '251:1')"}
                },
                "required": ["color", "block_id"]
            }
        }
    }
]

def extract_colors_from_voxel_text(file_content):
    """从体素文本内容中提取所有不同的颜色名称"""
    colors = set()
    lines = file_content.splitlines()

    for line_number, line in enumerate(lines):
        mod = line_number % 34  # 每34行一个循环：1行头，32行数据，1空行
        if 1 <= mod <= 32:  # 数据行
            components = line.strip().split()
            for comp in components:
                if comp != ".":
                    colors.add(comp)

    return sorted(colors)  # 返回排序后的颜色列表

def preserve_voxel_structure(file_content):
    """保留voxel_model.txt的原始结构，返回行列表"""
    return file_content.splitlines()

def replace_colors_with_ids(lines, color_to_id):
    """将原始行中的颜色替换为对应的block_id"""
    new_lines = []
    for line_number, line in enumerate(lines):
        mod = line_number % 34
        if 1 <= mod <= 32:  # 数据行
            components = line.strip().split()
            new_components = [color_to_id.get(comp, comp) if comp != "." else "." for comp in components]
            new_lines.append(" ".join(new_components))
        else:  # 头部或空行
            new_lines.append(line)
    return new_lines

def write_working_file(lines, output_filename="working.txt"):
    """将替换后的内容写入working.txt"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ 已生成输出文件: {output_filename}")

def analyze_images_and_voxel(image_dir=".", max_retries=3):
    # 读取 blockids.json 内容
    try:
        with open("blockids.json", 'r', encoding='utf-8-sig') as f:
            blockids_data = json.load(f)
        print("\n✅ blockids.json 文件读取成功.")
    except FileNotFoundError:
        error_msg = "blockids.json 文件未找到！请确保该文件存在于当前目录。"
        print(f"\n❌ {error_msg}")
        return f"错误：{error_msg}"
    except json.JSONDecodeError as e:
        error_msg = f"blockids.json 文件格式错误：{str(e)}"
        print(f"\n❌ {error_msg}")
        return f"错误：{error_msg}"

    # 读取 voxel_model.txt 内容并提取颜色和结构
    try:
        voxel_file_path = os.path.join(image_dir, "voxel_model.txt")
        voxel_content = read_file(voxel_file_path)
        print(f"\n✅ {voxel_file_path} 文件读取成功.")
        print(f"{voxel_file_path} 文件内容 (前 100 字):\n", voxel_content[:100] + "...")
        colors = extract_colors_from_voxel_text(voxel_content)
        original_lines = preserve_voxel_structure(voxel_content)
        print(f"\n发现以下颜色需要映射：{', '.join(colors)}")
    except FileNotFoundError as e:
        error_msg = f"{voxel_file_path} 文件未找到！请确保已完成模型体素化步骤。"
        print(f"\n❌ 错误：{error_msg}")
        return f"错误：{error_msg}"

    # 准备图片数据
    image_filenames = [filename for filename in os.listdir(image_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_data = []
    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        retries = 0
        while retries <= max_retries:
            try:
                img = Image.open(image_path)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data.append({
                    "filename": filename,
                    "image_url": f"data:image/jpeg;base64,{img_str}"
                })
                print(f"\n✅ 图片 {filename} 加载成功.")
                break
            except Exception as e:
                error_message = str(e)
                print(f"❌ 图片 {filename} 加载失败 (重试 {retries}/{max_retries})：{error_message}")
                if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                    retries += 1
                    time.sleep(retries * 2)
                else:
                    print(f"非可重试错误，停止重试。详细错误信息:\n{traceback.format_exc()}")
                    break
        if retries > max_retries:
            print(f"❌ 图片 {filename} 加载彻底失败，超出最大重试次数。")

    # ------------------- 合并回合：分析图片并映射颜色 -------------------
    print("\n------------------- 合并回合：分析图片并映射颜色 -------------------")
    color_to_id = {}
    try:
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        请根据以下图片和文件内容完成以下任务：
                        1. 首先，分析每张图片中的建筑结构、材料和可能的用途，并生成一个详细的描述。
                        2. 然后，为 voxel_model.txt 中的颜色列表分配方块ID。

                        参考以下 blockids.json 内容作为颜色到方块ID的映射依据：
                        {json.dumps(blockids_data, indent=2)}

                        voxel_model.txt 中的颜色需要映射：
                        {', '.join(colors)}

                        请先提供图片的描述，然后为每个颜色调用函数 'get_block_id' 返回结果，并在参数中明确指定 'color' 和 'block_id'。
                        'block_id' 应为字符串，如 '1' 或 '251:1'，支持方块ID变体。
                        优先从 blockids.json 中提取颜色到方块ID的映射（基于描述中的颜色名称）。
                        如果 blockids.json 中有多个方块对应同一颜色，根据图片内容选择最合适的ID。
                        如果没有明确映射，再根据图片内容推测一个合理的ID。
                        """.strip()
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": img["image_url"]}
                    } for img in image_data
                ]
            }
        ]

        try:
            print("\n🔄 正在调用AI进行分析，请稍候...")
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.01,
                top_p=0.95,
                max_tokens=8192
            )
            print("\n✅ AI调用成功")
        except Exception as api_error:
            error_detail = str(api_error)
            print(f"\n❌ AI调用失败：{error_detail}")
            if "api_key" in error_detail.lower() or "apikey" in error_detail.lower() or "key" in error_detail.lower():
                raise Exception(f"API密钥无效或未提供。详细信息：{error_detail}")
            elif "timeout" in error_detail.lower() or "timed out" in error_detail.lower():
                raise Exception(f"API调用超时。请稍后重试。详细信息：{error_detail}")
            elif "connection error" in error_detail.lower():
                raise Exception(f"连接错误。请检查网络连接。详细信息：{error_detail}")
            else:
                raise Exception(f"AI服务调用失败。详细信息：{error_detail}")

        # 记录AI的完整响应
        print(f"\n🔍 AI完整响应:\n{response}")

        # 提取图片描述
        description = response.choices[0].message.content
        print(f"\n📝 图片描述:\n{description}")

        # 处理函数调用
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🔧 函数调用详情 - 函数名: {func_name}, 参数: {func_args}")

                if func_name == "get_block_id" and "color" in func_args and "block_id" in func_args:
                    color = func_args["color"]
                    block_id = func_args["block_id"]
                    if block_id in blockids_data:
                        color_to_id[color] = block_id
                        print(f"✅ 颜色 '{color}' 成功映射为方块ID: {block_id}")
                    else:
                        print(f"❌ 无效的block_id: {block_id}，不在blockids.json中，跳过颜色 '{color}'")
                else:
                    print(f"❌ 函数调用参数不完整或错误，跳过")
        else:
            print("⚠️ AI未调用函数。")

        # 检查是否所有颜色都被映射
        missing_colors = [color for color in colors if color not in color_to_id]
        if missing_colors:
            print(f"⚠️ 以下颜色未被映射: {', '.join(missing_colors)}")
            # 对于未映射的颜色，保留原始值
            # 这里不做额外处理，确保格式不变

        # 输出映射结果
        print("\n🎉 颜色到方块ID的映射结果:")
        for color, block_id in color_to_id.items():
            print(f"颜色 '{color}' -> 方块ID: {block_id}")

        # 生成working.txt
        new_lines = replace_colors_with_ids(original_lines, color_to_id)
        write_working_file(new_lines)

        return color_to_id

    except Exception as e:
        print(f"\n❌ 处理错误：{str(e)}\n详细错误信息: \n{traceback.format_exc()}")
        return f"处理错误：{str(e)}"

# 使用示例
if __name__ == "__main__":
    result = analyze_images_and_voxel()
    print(f"\n\n最终结果:\n{result}")
