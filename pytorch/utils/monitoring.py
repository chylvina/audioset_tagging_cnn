def gpu_monitor():
    """监控GPU使用情况"""
    print('GPU Memory Usage:')
    print(f'Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
    print(f'Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
    print(f'Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')

# 在关键位置调用
if iteration % 100 == 0:
    gpu_monitor() 