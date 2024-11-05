#!/bin/bash

# 自动连接服务器并执行命令
ssh jetflow_74_12238 << EOF

# 激活 Anaconda 环境
source ~/anaconda3/bin/activate pytorch  # 请根据你的Anaconda路径调整

# 可选：确认环境激活成功
echo "当前环境：\$(conda info --envs | grep '*' | awk '{print \$1}')"

# 在环境中运行指定命令
# 示例：python your_script.py

cd /data1/zhn/macdata/code/github/python/font_ocr_rec
python appRecog.py

EOF
