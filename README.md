# HUST-Introduction-to-AI
### **本实验为基于强化学习的PPO算法分别在离散动作和连续动作的两种仿真环境中对机械控制模型进行训练**

附件包括：

1. 实验报告：**人工智能导论报告_宁子健_U202315633 .pdf**
2. 仿真环境1：**Acrobot**
   - **img文件夹**：包括训练过程可视化和模型测试结果的图片
   - **模型参数**：Actorbot_actor.pth，Actorbot_critic.pth
   - **模型训练文件**：ppo_Actorbot.py
   - **模型测试文件**：Actorbot_test.py
3. 仿真环境2：**RobotArm(机械臂避障抓取目标物)**
   - **img文件夹**：包括训练过程和测试结果的图片
   - **环境搭建相关**：fr5_description文件夹、env.py
   - **模型参数**：RobotArm.zip
   - **模型训练文件**：ppo_RobotArm.py
   - **模型测试文件**：RobotArm_test.py

4. 模型测试录屏：模型测试录屏.mp4
   - 模型测试时Acrobot环境每回合完成所需时间过长所以仅在最后一回合进行页面可视化，RobotArm测试时全程可视化。

环境配置：gymnasium，pybullet，stable_baselines3，swanlab，pytorch等库(具体参考代码)，环境配置完成后测试文件是可以直接运行的。
