import re
import matplotlib.pyplot as plt



def visual_loss(log_dir:str):
    # 读取txt文件并提取loss数据
    with open('./static/log/PretrainMXRDM.txt', 'r') as file:
        lines = file.readlines()

    # 初始化存储loss数据的列表
    loss_data = []

    for line in lines:
        match = re.search(r'iter(\d+)_train, avg_loss= ([0-9.]+)', line)
        if match:
            iter_num = int(match.group(1))
            loss = float(match.group(2))
            loss_data.append(loss)
            current_iter = iter_num

    # 创建迭代次数的列表，叠加iter
    iterations = list(range(0, len(loss_data) * 50, 50))

    # 创建图形
    fig = plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_data, marker='o', linestyle='-', color='cornflowerblue', markersize=5, label='Loss')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Loss vs. Iteration with Iteration Accumulation', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('lightgray')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig

def robot_out(log_dir:str):
    with open(log_dir, 'r') as file:
        lines = file.readlines()
    messages = []
    for line in lines:
        if line.startswith('INFO:root:user:'):
            message = line.replace('INFO:root:user:','').strip()
            messages.append(message)
        elif line.startswith('INFO:root:assistant: '):
            message = line.replace('INFO:root:assistant:','').strip()
            messages.append(message)
        elif line.startswith('INFO:root:'):
            message = line.replace('INFO:root:','').strip()
            messages.append(message)

    return messages



# robot_out('./static/log/PretrainMXRDM.txt')