{
    "version": "0.2.0",
    "configurations": [
        
        
    
        
        
        
        
        
        
        {
            "name": "Python: TorchRun DDP Debug",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/gjy/anaconda3/envs/yjh/bin/torchrun",  // 直接调用 torchrun 可执行文件
            "args": [
                "--nnodes=1",
                "--nproc_per_node=4",
                "--rdzv_id=12345",        // 分布式训练的随机 ID
                "--rdzv_backend=c10d",    // 使用 c10d 后端
                "--rdzv_endpoint=localhost:29500", // 主节点地址
                "/home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/main_mask2former.py",  // 你的主脚本路径
                // 添加你的脚本参数（例如配置文件路径）

            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "NCCL_DEBUG": "INFO",      // 启用 NCCL 调试信息
                "NCCL_SOCKET_IFNAME": "en",  // 指定网络接口（根据实际情况修改）
                "CUDA_VISIBLE_DEVICES": "0,1,2,3", // 可见的 GPU 设备
                "PYTHONPATH": "${workspaceFolder}"  // 确保 Python 路径包含项目根目录
            }
        }
    ]
}