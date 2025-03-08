import os
import json
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import time
import traceback

# 初始化 OpenAI 客户端（假设已正确配置）
client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# 文件读取函数
def read_file(filename: str):
    """读取指定文件内容，如果文件不存在则报错"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found!")
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# 函数调用工具
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
        mod = line_number % 34
        if 1 <= mod <= 32:
            components = line.strip().split()
            for comp in components:
                if comp != ".":
                    colors.add(comp)
    return sorted(colors)

def preserve_voxel_structure(file_content):
    """保留voxel_model.txt的原始结构，返回行列表"""
    return file_content.splitlines()

def replace_colors_with_ids(lines, color_to_id):
    """将原始行中的颜色替换为对应的block_id"""
    new_lines = []
    for line_number, line in enumerate(lines):
        mod = line_number % 34
        if 1 <= mod <= 32:
            components = line.strip().split()
            new_components = [color_to_id.get(comp, comp) if comp != "." else "." for comp in components]
            new_lines.append(" ".join(new_components))
        else:
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
        colors = extract_colors_from_voxel_text(voxel_content)
        original_lines = preserve_voxel_structure(voxel_content)
        print(f"\n发现以下颜色需要映射：{', '.join(colors)}")
    except FileNotFoundError as e:
        error_msg = f"{voxel_file_path} 文件未找到！请确保已完成模型体素化步骤。"
        print(f"\n❌ 错误：{error_msg}")
        return f"错误：{error_msg}"

    # 准备图片数据，排除以'extracted'开头的文件
    image_filenames = [
        filename for filename in os.listdir(image_dir)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        and not filename.lower().startswith('extracted')
    ]
    image_data = []
    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
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
        except Exception as e:
            print(f"❌ 图片 {filename} 加载失败：{str(e)}")

    # 初始化颜色映射字典和图片描述
    color_to_id = {}
    image_description = None
    retry_count = 0

    # 主循环：处理颜色映射
    while retry_count <= max_retries:
        if retry_count == 0:
            # 第一次请求：分析图片和体素文件
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            请根据以下图片和文件内容完成以下任务：
                            1. 分析每张图片中的建筑结构、材料和可能的用途，并生成一个详细的描述。
                            2. 为 voxel_model.txt 中的颜色列表分配方块ID。

                            参考以下 blockids.json 内容：
                            {json.dumps(blockids_data, indent=2)}

                            需要映射的颜色：
                            {', '.join(colors)}

                            请先提供图片描述，然后为每个颜色调用 'get_block_id' 函数返回结果，
                            参数中明确指定 'color' 和 'block_id'。
                            'block_id' 为字符串，如 '1' 或 '251:1'。
                            优先从 blockids.json 中提取映射，若有多个选项，根据图片描述选择最合适的值，
                            若无明确映射，则根据描述推测合理ID。
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
        else:
            # 后续请求：为未映射的颜色请求映射
            missing_colors = [color for color in colors if color not in color_to_id]
            if not missing_colors:
                break
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            以下是图片的描述：
                            {image_description}

                            以下颜色尚未映射：
                            {', '.join(missing_colors)}

                            请为这些颜色提供Minecraft方块ID。
                            参考 blockids.json：
                            {json.dumps(blockids_data, indent=2)}

                            为每个颜色调用 'get_block_id' 函数返回结果，
                            参数中明确指定 'color' 和 'block_id'。
                            'block_id' 为字符串，如 '1' 或 '251:1'。
                            优先从 blockids.json 中提取映射，若有多个选项，根据描述选择最合适的值，
                            若无明确映射，则根据描述推测合理ID。
                            """.strip()
                        }
                    ]
                }
            ]

        # 发送AI请求
        try:
            print(f"\n🔄 第 {retry_count + 1} 次调用AI进行分析...")
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
        except Exception as e:
            print(f"\n❌ AI调用失败：{str(e)}")
            retry_count += 1
            continue

        # 提取图片描述（仅第一次）
        if retry_count == 0:
            image_description = response.choices[0].message.content
            print(f"\n📝 图片描述:\n{image_description}")

        # 处理函数调用
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                if func_name == "get_block_id" and "color" in func_args and "block_id" in func_args:
                    color = func_args["color"]
                    block_id = func_args["block_id"]
                    if block_id in blockids_data:
                        color_to_id[color] = block_id
                        print(f"✅ 颜色 '{color}' 成功映射为方块ID: {block_id}")
                    else:
                        print(f"❌ 无效的block_id: {block_id}，跳过颜色 '{color}'")

        # 检查未映射的颜色
        missing_colors = [color for color in colors if color not in color_to_id]
        if not missing_colors:
            break
        print(f"⚠️ 以下颜色未被映射: {', '.join(missing_colors)}")
        retry_count += 1

    # 检查最终结果
    if missing_colors:
        print(f"\n⚠️ 经过 {max_retries} 次重试，仍有以下颜色未映射: {', '.join(missing_colors)}")
        print("这些颜色将保留原始值。")

    # 输出映射结果
    print("\n🎉 颜色到方块ID的映射结果:")
    for color, block_id in color_to_id.items():
        print(f"颜色 '{color}' -> 方块ID: {block_id}")

    # 生成working.txt
    new_lines = replace_colors_with_ids(original_lines, color_to_id)
    write_working_file(new_lines)

    return color_to_id

if __name__ == "__main__":
    result = analyze_images_and_voxel()
    print(f"\n\n最终结果:\n{result}")
