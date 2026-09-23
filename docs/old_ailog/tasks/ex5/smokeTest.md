# 主任务概述

SmokeTest Phase 1/2/3

# 要求

- 简单构造Phase 0
    - 数据集先借用../FwdLLMForVarianceTest中的AGNews数据集
- 模型在~/projects/smell/JengaForMemoryTest/Jenga/checkpoints，即Jenga的LLAMA2和附属的预测器权重
- 你可以实际阅读~/projects/smell中的全部项目，可以和我要权限
- 你要考虑3个总客户，3个客户每轮，3轮，三个Phase, -np 2, bsz = 1, epoch = 1
- 为了防止连锁问题难以查bug，你需要写三个RUNME脚本
- conda环境使用smell-v1
    - 如若需要其他依赖自行pip
- 考虑在/script设计RUNME-SmokeTest-Phase*.sh脚本，连接到对应脚本
- 考虑将输出日志重定向到/log以便你查bug
- /tmp下放临时文件
- 你仅可以写/home/yangyongbo内的文件，你不能越界
- kill程序时不允许kill任何非yangyongbo发起的程序
- 任何对源代码的修改需要参照standard.md