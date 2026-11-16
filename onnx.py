import torch
from ultralytics import YOLO
import traceback

# 1. 加载模型
print("[*] 正在加载模型...")
PT_PATH = ""
model = YOLO(PT_PATH)
pytorch_model = model.model
pytorch_model.eval()

# 强制使用 CPU 进行 Debug，因为 CPU 上的张量报错信息最准确
device = torch.device("cpu")
pytorch_model.to(device)

# 2. 核心：构建模块执行栈追踪器
module_stack = []

def pre_hook(module, input, name):
    """进入模块时压入栈"""
    module_stack.append(name)

def post_hook(module, input, output, name):
    """模块执行成功离开时弹出栈"""
    if module_stack and module_stack[-1] == name:
        module_stack.pop()

# 遍历模型中的所有子模块，注册雷达钩子
for name, module in pytorch_model.named_modules():
    # 过滤掉空的根节点名称
    if name == "":
        continue
    module.register_forward_pre_hook(lambda m, i, n=name: pre_hook(m, i, n))
    module.register_forward_hook(lambda m, i, o, n=name: post_hook(m, i, o, n))

# 3. 伪造输入张量
dummy_input = torch.randn(1, 3, 512, 512).to(device)

print("==================================================")
print("[*] 雷达已部署！开始追踪 ONNX 导出过程...")
print("==================================================")

try:
    # 开始导出
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        "error_test.onnx",
        opset_version=16,
        do_constant_folding=True
    )
    print("✅ 居然导出成功了？（这说明问题可能不是静态结构引起的）")

except Exception as e:
    print("\n🚨 捕获到崩溃！")
    print("==================================================")
    print("🎯 【精准定位】：崩溃瞬间，模型正在执行的模块层级路径如下：")
    print("--------------------------------------------------")
    
    # 打印堆栈层级
    indent = ""
    for idx, mod_name in enumerate(module_stack):
        if idx == len(module_stack) - 1:
            print(f"{indent}└── 💥 [崩溃点在这里] -> {mod_name}")
        else:
            print(f"{indent}├── {mod_name}")
        indent += "    "
        
    print("--------------------------------------------------")
    print(f"具体的原生错误信息: {str(e)}")
    print("==================================================")
    print("\n👉 请把上面打印出来的【崩溃点路径】复制发给我，我们就能精准击破它！")