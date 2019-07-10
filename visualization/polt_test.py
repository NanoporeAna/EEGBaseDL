from visualization.brain_map import ax_scalp

values=[-0.2,0.4,-0.150,-0.344,0.25,0.6,-0.47,0.80,-0.191]
value=[0.2,-0.4,0.150,0.344,-0.25,0.6,0.47,-0.80,0.191]#这里是得到相关性的值
channel = ['FVL','FVR','BP','F3-C3','T3-P3','P3-O1','F4-C3','T4-P4','P4-O2']
a = ax_scalp(value,channel)
"""
现在只弄了通道值在电极上的值对应的框架，通道位置可能有点不对
"""