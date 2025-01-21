import torch
print(torch.cuda.is_available())  # 如果返回 True，说明支持 GPU
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称
