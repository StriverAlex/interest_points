# import os
# import hydra
# import datetime
# import wandb
# import torch
# import numpy as np
# from omegaconf import DictConfig, OmegaConf
# from omni.isaac.kit import SimulationApp
# from caric_ppo import CARICPPO
# from omni_drones.controllers import LeePositionController
# from omni_drones.utils.torchrl.transforms import VelController
# from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
# from torchrl.envs.transforms import TransformedEnv, Compose
# from utils import evaluate
# from torchrl.envs.utils import ExplorationType
# import gc
# import psutil
# import traceback

# FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

# @hydra.main(config_path=FILE_PATH, config_name="caric_train", version_base=None)
# def main(cfg):
#     """CARIC-RL混合训练主函数"""
#     # 设备配置
#     if hasattr(cfg, 'device') and 'cuda' in cfg.device:
#         if not torch.cuda.is_available():
#             print("[WARNING] CUDA不可用，回退到CPU")
#             cfg.device = 'cpu'
#         else:
#             device_id = cfg.device.split(':')[1] if ':' in cfg.device else '0'
#             if int(device_id) >= torch.cuda.device_count():
#                 print(f"[WARNING] CUDA设备 {device_id} 不可用，使用 CUDA:0")
#                 cfg.device = 'cuda:0'
#             torch.cuda.set_device(cfg.device)
    
#     print(f"[CARIC-RL]: 使用设备: {cfg.device}")
    
#     # 设置随机种子
#     torch.manual_seed(cfg.seed)
#     np.random.seed(cfg.seed)
    
#     # 创建仿真应用
#     sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

#     # 初始化Wandb
#     run = wandb.init(
#         project=cfg.wandb.project,
#         name=f"CARIC-RL/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
#         entity=cfg.wandb.entity,
#         config=cfg,
#         mode=cfg.wandb.mode,
#         tags=["CARIC", "building_coverage", "single_agent"]
#     )

#     try:
#         print("\n=== 初始化CARIC-RL混合环境 ===")
#         # 环境初始化
#         from caric_rl_env import CARICRLEnvironment
#         env = CARICRLEnvironment(cfg)
        
#         # 控制器设置
#         controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
#         print(f"使用Lee位置控制器")
        
#         vel_transform = VelController(controller, yaw_control=True)
#         transformed_env = TransformedEnv(env, vel_transform).train()
#         transformed_env.set_seed(cfg.seed)   

#         print("\n=== 环境观察空间 ===")
#         print(f"状态维度: {transformed_env.observation_spec}")
        
#         # 策略初始化
#         print("\n=== 初始化CARIC-PPO策略 ===")
#         policy = CARICPPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
#         print("策略初始化成功")
        
#         # 集成CARIC传统组件
#         caric_components = initialize_caric_components(env, cfg)
        
#         # 集采统计
#         episode_stats_keys = [
#             k for k in transformed_env.observation_spec.keys(True, True) 
#             if isinstance(k, tuple) and k[0]=="stats"
#         ]
#         episode_stats = EpisodeStats(episode_stats_keys)

#         # 数据收集器
#         collector = SyncDataCollector(
#             transformed_env,
#             policy=policy, 
#             frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
#             total_frames=cfg.max_frame_num,
#             device=cfg.device,
#             return_same_td=True,
#             exploration_type=ExplorationType.RANDOM
#         )

#         # 训练指标
#         best_mission_completion = 0.0
#         best_exploration_coverage = 0.0
#         no_improvement_count = 0
#         patience = cfg.get('early_stopping_patience', 50)
        
#         # CARIC特定指标
#         caric_metrics = {
#             'total_interest_points_found': 0,
#             'average_mapping_quality': 0.0,
#             'scanning_efficiency': 0.0,
#             'exploration_completeness': 0.0
#         }

#         def print_memory_usage():
#             """打印内存使用情况"""
#             if torch.cuda.is_available():
#                 print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
#             process = psutil.Process(os.getpid())
#             print(f"CPU内存: {process.memory_info().rss/1024**3:.2f}GB")

#         # 训练循环
#         print("\n=== 开始CARIC-RL训练 ===")
#         for i, data in enumerate(collector):
#             try:
#                 # 内存清理
#                 if i % 100 == 0:
#                     gc.collect()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
                
#                 # 训练信息
#                 info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

#                 # 策略训练
#                 train_loss_stats = policy.train(data)
#                 info.update(train_loss_stats)
                
#                 # 集成CARIC组件更新
#                 caric_info = update_caric_components(caric_components, data, env)
#                 info.update(caric_info)

#                 # 集采统计处理
#                 episode_stats.add(data)
                
#                 if len(episode_stats) > 0:
#                     stats = {
#                         "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
#                         for k, v in episode_stats.pop().items(True, True)
#                     }
#                     info.update(stats)

#                 # 进度打印
#                 if i % 20 == 0:
#                     exploration_coverage = info.get("train/stats.exploration_coverage", 0.0)
#                     interest_points_found = info.get("train/stats.interest_points_found", 0.0)
#                     mission_completion = info.get("train/stats.mission_completion", 0.0)
#                     scanning_efficiency = info.get("train/stats.scanning_efficiency", 0.0)
#                     actor_loss = info.get("actor_loss", 0.0)
#                     critic_loss = info.get("critic_loss", 0.0)
                    
#                     print(f"步骤 {i:4d} | 帧数: {collector._frames:8d} | "
#                           f"探索覆盖: {exploration_coverage:.3f} | 兴趣点: {interest_points_found:.0f} | "
#                           f"任务完成: {mission_completion:.3f} | 扫描效率: {scanning_efficiency:.3f} | "
#                           f"A损失: {actor_loss:.4f} | C损失: {critic_loss:.4f}")
                    
#                     if i % 100 == 0:
#                         print_memory_usage()

#                 # 评估
#                 if i % cfg.eval_interval == 0 and i > 0:
#                     print(f"\n[步骤 {i} 评估]")
                    
#                     try:
#                         env.eval()
#                         eval_info = evaluate_caric_performance(
#                             env=transformed_env, 
#                             policy=policy,
#                             caric_components=caric_components,
#                             seed=cfg.seed, 
#                             cfg=cfg,
#                             exploration_type=ExplorationType.MEAN
#                         )
#                         env.train()
                        
#                         info.update(eval_info)
#                         eval_exploration = eval_info.get("eval/stats.exploration_coverage", 0.0)
#                         eval_mission = eval_info.get("eval/stats.mission_completion", 0.0)
#                         eval_interest_points = eval_info.get("eval/stats.interest_points_found", 0.0)
#                         eval_scanning = eval_info.get("eval/stats.scanning_efficiency", 0.0)
                        
#                         print(f"评估结果:")
#                         print(f"  探索覆盖: {eval_exploration:.4f}")
#                         print(f"  任务完成: {eval_mission:.4f}")
#                         print(f"  发现兴趣点: {eval_interest_points:.0f}")
#                         print(f"  扫描效率: {eval_scanning:.4f}")
                        
#                         # 最佳模型保存
#                         combined_score = eval_exploration * 0.3 + eval_mission * 0.7
#                         best_combined = best_exploration_coverage * 0.3 + best_mission_completion * 0.7
                        
#                         if combined_score > best_combined:
#                             best_exploration_coverage = eval_exploration
#                             best_mission_completion = eval_mission
#                             no_improvement_count = 0
                            
#                             best_ckpt_path = os.path.join(run.dir, "caric_best_model.pt")
#                             torch.save({
#                                 'policy_state_dict': policy.state_dict(),
#                                 'training_step': policy.training_step,
#                                 'best_exploration_coverage': best_exploration_coverage,
#                                 'best_mission_completion': best_mission_completion,
#                                 'caric_metrics': caric_metrics,
#                                 'config': cfg
#                             }, best_ckpt_path)
                            
#                             print(f"新的最佳模型已保存! 综合分数: {combined_score:.4f}")
#                             wandb.run.summary["best_exploration_coverage"] = best_exploration_coverage
#                             wandb.run.summary["best_mission_completion"] = best_mission_completion
#                             wandb.run.summary["best_combined_score"] = combined_score
#                         else:
#                             no_improvement_count += 1
#                             print(f"连续 {no_improvement_count} 次评估无改进")
                        
#                         # 早停检查
#                         if no_improvement_count >= patience:
#                             print(f"连续 {no_improvement_count} 次评估无改进，触发早停")
#                             break
                            
#                     except Exception as e:
#                         print(f"评估失败: {e}")
#                         traceback.print_exc()
#                         env.train()

#                 # 定期模型保存
#                 if i % cfg.save_interval == 0 and i > 0:
#                     ckpt_path = os.path.join(run.dir, f"caric_checkpoint_{i}.pt")
#                     torch.save({
#                         'policy_state_dict': policy.state_dict(),
#                         'training_step': policy.training_step,
#                         'caric_metrics': caric_metrics,
#                         'config': cfg
#                     }, ckpt_path)
#                     print(f"步骤 {i} 检查点已保存")

#                 # 记录到wandb
#                 run.log(info)
                
#                 # 更新CARIC指标
#                 update_caric_metrics(caric_metrics, info)
                
#             except Exception as e:
#                 print(f"训练步骤 {i} 失败: {e}")
#                 traceback.print_exc()
#                 # 尝试恢复
#                 gc.collect()
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#                 continue

#         # 保存最终模型
#         final_ckpt_path = os.path.join(run.dir, "caric_final_model.pt")
#         torch.save({
#             'policy_state_dict': policy.state_dict(),
#             'training_step': policy.training_step,
#             'best_exploration_coverage': best_exploration_coverage,
#             'best_mission_completion': best_mission_completion,
#             'final_caric_metrics': caric_metrics,
#             'config': cfg
#         }, final_ckpt_path)
#         print("最终CARIC模型已保存")
        
#         # 打印最终训练总结
#         print("\n" + "="*60)
#         print("CARIC-RL训练完成")
#         print("="*60)
#         print(f"最佳探索覆盖率: {best_exploration_coverage:.4f}")
#         print(f"最佳任务完成率: {best_mission_completion:.4f}")
#         print(f"总训练步骤: {i}")
#         print(f"总处理帧数: {collector._frames}")
#         print("\nCARIC特定指标:")
#         for key, value in caric_metrics.items():
#             print(f"  {key}: {value}")
#         print("="*60)

#     except Exception as e:
#         print(f"训练失败，错误: {e}")
#         traceback.print_exc()
#     finally:
#         print("清理资源...")
#         wandb.finish()
#         sim_app.close()
#         print("CARIC-RL训练脚本完成")


# def initialize_caric_components(env, cfg):
#     """初始化CARIC传统组件"""
#     components = {
#         'mapping_quality_tracker': MappingQualityTracker(),
#         'interest_point_analyzer': InterestPointAnalyzer(),
#         'scanning_efficiency_monitor': ScanningEfficiencyMonitor(),
#         'exploration_completeness_tracker': ExplorationCompletenessTracker()
#     }
#     return components


# def update_caric_components(components, data, env):
#     """更新CARIC组件并返回相关指标"""
#     info = {}
    
#     # 更新建图质量
#     mapping_info = components['mapping_quality_tracker'].update(env)
#     info.update(mapping_info)
    
#     # 更新兴趣点分析
#     interest_info = components['interest_point_analyzer'].update(env)
#     info.update(interest_info)
    
#     # 更新扫描效率
#     scanning_info = components['scanning_efficiency_monitor'].update(env)
#     info.update(scanning_info)
    
#     # 更新探索完整性
#     exploration_info = components['exploration_completeness_tracker'].update(env)
#     info.update(exploration_info)
    
#     return info


# def evaluate_caric_performance(env, policy, caric_components, cfg, seed=0, exploration_type=ExplorationType.MEAN):
#     """评估CARIC性能"""
#     from omni_drones.utils.torchrl import RenderCallback
#     from torchrl.envs.utils import set_exploration_type
    
#     env.enable_render(True)
#     env.eval()
#     env.set_seed(seed)

#     render_callback = RenderCallback(interval=2)
    
#     with set_exploration_type(exploration_type):
#         trajs = env.rollout(
#             max_steps=env.max_episode_length,
#             policy=policy,
#             callback=render_callback,
#             auto_reset=True,
#             break_when_any_done=False,
#             return_contiguous=False,
#         )
    
#     env.enable_render(not cfg.headless)
#     env.reset()
    
#     # 处理轨迹数据
#     done = trajs.get(("next", "done")) 
#     first_done = torch.argmax(done.long(), dim=1).cpu()

#     def take_first_episode(tensor: torch.Tensor):
#         indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
#         return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

#     traj_stats = {
#         k: take_first_episode(v)
#         for k, v in trajs[("next", "stats")].cpu().items()
#     }

#     info = {
#         "eval/stats." + k: torch.mean(v.float()).item() 
#         for k, v in traj_stats.items()
#     }

#     # 添加CARIC特定评估
#     caric_eval_info = evaluate_caric_specific_metrics(env, trajs, caric_components)
#     info.update(caric_eval_info)

#     # 记录视频
#     info["recording"] = wandb.Video(
#         render_callback.get_video_array(axes="t c h w"), 
#         fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
#         format="mp4"
#     )
    
#     env.train()
#     return info


# def evaluate_caric_specific_metrics(env, trajs, components):
#     """评估CARIC特定指标"""
#     caric_info = {}
    
#     # 3D建图质量分析
#     mapping_quality = analyze_3d_mapping_quality(env)
#     caric_info["eval/caric_mapping_quality"] = mapping_quality
    
#     # 兴趣点检测效果
#     interest_detection_score = analyze_interest_point_detection(env)
#     caric_info["eval/caric_interest_detection"] = interest_detection_score
    
#     # 扫描路径效率
#     path_efficiency = analyze_scanning_path_efficiency(trajs)
#     caric_info["eval/caric_path_efficiency"] = path_efficiency
    
#     # 覆盖完整性
#     coverage_completeness = analyze_coverage_completeness(env)
#     caric_info["eval/caric_coverage_completeness"] = coverage_completeness
    
#     return caric_info


# def update_caric_metrics(metrics, info):
#     """更新CARIC指标"""
#     # 更新累积指标
#     if "train/stats.interest_points_found" in info:
#         metrics['total_interest_points_found'] = max(
#             metrics['total_interest_points_found'], 
#             info["train/stats.interest_points_found"]
#         )
    
#     if "caric_mapping_quality" in info:
#         # 滑动平均
#         alpha = 0.9
#         metrics['average_mapping_quality'] = (
#             alpha * metrics['average_mapping_quality'] + 
#             (1 - alpha) * info["caric_mapping_quality"]
#         )
    
#     if "train/stats.scanning_efficiency" in info:
#         metrics['scanning_efficiency'] = info["train/stats.scanning_efficiency"]
    
#     if "train/stats.exploration_coverage" in info:
#         metrics['exploration_completeness'] = info["train/stats.exploration_coverage"]


# # CARIC组件类定义
# class MappingQualityTracker:
#     """建图质量跟踪器"""
    
#     def __init__(self):
#         self.quality_history = []
#         self.confidence_threshold = 0.5
    
#     def update(self, env):
#         """更新建图质量指标"""
#         total_quality = 0.0
        
#         for env_idx in range(env.num_envs):
#             # 计算高置信度体素比例
#             high_conf_voxels = torch.sum(env.confidence_grid[env_idx] > self.confidence_threshold)
#             total_voxels = torch.sum(env.occupancy_grid[env_idx] > 0)
            
#             if total_voxels > 0:
#                 quality = high_conf_voxels.float() / total_voxels.float()
#                 total_quality += quality.item()
        
#         avg_quality = total_quality / env.num_envs
#         self.quality_history.append(avg_quality)
        
#         return {"caric_mapping_quality": avg_quality}


# class InterestPointAnalyzer:
#     """兴趣点分析器"""
    
#     def __init__(self):
#         self.detection_history = []
#         self.confidence_history = []
    
#     def update(self, env):
#         """更新兴趣点检测分析"""
#         total_detection_score = 0.0
#         total_confidence = 0.0
        
#         for env_idx in range(env.num_envs):
#             # 活跃兴趣点数量
#             active_points = torch.sum(env.interest_point_active[env_idx])
            
#             # 平均置信度
#             if active_points > 0:
#                 active_mask = env.interest_point_active[env_idx]
#                 avg_confidence = torch.mean(env.interest_point_confidence[env_idx][active_mask])
#                 total_confidence += avg_confidence.item()
            
#             # 检测分数（基于点的密度和分布）
#             detection_score = self._calculate_detection_score(env, env_idx)
#             total_detection_score += detection_score
        
#         avg_detection = total_detection_score / env.num_envs
#         avg_confidence = total_confidence / env.num_envs
        
#         return {
#             "caric_interest_detection_score": avg_detection,
#             "caric_interest_confidence": avg_confidence
#         }
    
#     def _calculate_detection_score(self, env, env_idx):
#         """计算兴趣点检测分数"""
#         active_points = env.dynamic_interest_points[env_idx][env.interest_point_active[env_idx]]
        
#         if len(active_points) == 0:
#             return 0.0
        
#         # 基于点的空间分布评分
#         # 这里可以实现更复杂的评分逻辑
#         return min(len(active_points) / 50.0, 1.0)  # 简化版本


# class ScanningEfficiencyMonitor:
#     """扫描效率监控器"""
    
#     def __init__(self):
#         self.scan_history = []
#         self.path_length_history = []
    
#     def update(self, env):
#         """更新扫描效率指标"""
#         total_efficiency = 0.0
        
#         for env_idx in range(env.num_envs):
#             # 扫描完成的点数
#             scanned_points = torch.sum(env.interest_point_scanned[env_idx])
            
#             # 总活跃点数
#             active_points = torch.sum(env.interest_point_active[env_idx])
            
#             if active_points > 0:
#                 efficiency = scanned_points.float() / active_points.float()
#                 total_efficiency += efficiency.item()
        
#         avg_efficiency = total_efficiency / env.num_envs
        
#         return {"caric_scanning_efficiency": avg_efficiency}


# class ExplorationCompletenessTracker:
#     """探索完整性跟踪器"""
    
#     def __init__(self):
#         self.completeness_history = []
    
#     def update(self, env):
#         """更新探索完整性"""
#         total_completeness = 0.0
        
#         for env_idx in range(env.num_envs):
#             # 已知体素数量
#             known_voxels = torch.sum(env.occupancy_grid[env_idx] > 0)
#             total_voxels = env.occupancy_grid[env_idx].numel()
            
#             completeness = known_voxels.float() / total_voxels
#             total_completeness += completeness.item()
        
#         avg_completeness = total_completeness / env.num_envs
        
#         return {"caric_exploration_completeness": avg_completeness}


# # 分析函数
# def analyze_3d_mapping_quality(env):
#     """分析3D建图质量"""
#     quality_scores = []
    
#     for env_idx in range(env.num_envs):
#         # 计算地图完整性
#         occupied_voxels = torch.sum(env.occupancy_grid[env_idx] == 2)
#         free_voxels = torch.sum(env.occupancy_grid[env_idx] == 1)
#         total_mapped = occupied_voxels + free_voxels
        
#         # 计算置信度分布
#         avg_confidence = torch.mean(env.confidence_grid[env_idx][env.confidence_grid[env_idx] > 0])
        
#         # 综合质量分数
#         if total_mapped > 0:
#             quality = (total_mapped.float() / env.occupancy_grid[env_idx].numel()) * avg_confidence.item()
#             quality_scores.append(quality)
    
#     return np.mean(quality_scores) if quality_scores else 0.0


# def analyze_interest_point_detection(env):
#     """分析兴趣点检测效果"""
#     detection_scores = []
    
#     for env_idx in range(env.num_envs):
#         active_points = torch.sum(env.interest_point_active[env_idx])
        
#         # 基于类型多样性的分数
#         if active_points > 0:
#             types = env.interest_point_type[env_idx][env.interest_point_active[env_idx]]
#             unique_types = torch.unique(types).numel()
#             type_diversity = unique_types / 4.0  # 假设有4种类型
            
#             # 基于空间分布的分数
#             positions = env.dynamic_interest_points[env_idx][env.interest_point_active[env_idx]]
#             spatial_spread = torch.std(positions).item() if len(positions) > 1 else 0.0
            
#             score = (type_diversity + min(spatial_spread / 10.0, 1.0)) / 2.0
#             detection_scores.append(score)
    
#     return np.mean(detection_scores) if detection_scores else 0.0


# def analyze_scanning_path_efficiency(trajs):
#     """分析扫描路径效率"""
#     # 这里可以分析轨迹的效率，比如路径长度vs扫描收益
#     # 简化实现
#     return 0.8  # 占位符


# def analyze_coverage_completeness(env):
#     """分析覆盖完整性"""
#     completeness_scores = []
    
#     for env_idx in range(env.num_envs):
#         # 兴趣点覆盖率
#         active_points = torch.sum(env.interest_point_active[env_idx])
#         scanned_points = torch.sum(env.interest_point_scanned[env_idx])
        
#         if active_points > 0:
#             point_coverage = scanned_points.float() / active_points.float()
#         else:
#             point_coverage = 0.0
        
#         # 空间覆盖率
#         visited_voxels = torch.sum(env.visited_grid[env_idx])
#         total_voxels = env.visited_grid[env_idx].numel()
#         spatial_coverage = visited_voxels.float() / total_voxels
        
#         # 综合完整性
#         completeness = (point_coverage + spatial_coverage) / 2.0
#         completeness_scores.append(completeness.item())
    
#     return np.mean(completeness_scores) if completeness_scores else 0.0


# if __name__ == "__main__":
#     main()





import os
import hydra
import datetime
import wandb
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import CARICAwarePPO  # 使用CARIC感知的PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import gc
import psutil
import traceback

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    """CARIC-RL训练主函数"""
    
    # Device configuration
    if hasattr(cfg, 'device') and 'cuda' in cfg.device:
        if not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            cfg.device = 'cpu'
        else:
            device_id = cfg.device.split(':')[1] if ':' in cfg.device else '0'
            if int(device_id) >= torch.cuda.device_count():
                print(f"[WARNING] CUDA device {device_id} not available. Using CUDA:0")
                cfg.device = 'cuda:0'
            torch.cuda.set_device(cfg.device)
    
    print(f"[CARIC-RL]: Training using device: {cfg.device}")
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create simulation app
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Initialize Wandb for CARIC-RL
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"CARIC-RL/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
            tags=["CARIC-RL", "Hybrid", "Scanning", "RL"]
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"CARIC-RL/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    try:
        # CARIC-RL Environment initialization
        print("\n=== Initializing CARIC-RL Environment ===")
        from env import CARICRLEnvironment
        env = CARICRLEnvironment(cfg)
        print("CARIC-RL Environment initialized successfully")
        
        # Controller setup (simplified for CARIC-RL)
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        print(f"=== Using Lee Position Controller (No Gimbal) ===")
        
        vel_transform = VelController(controller, yaw_control=True)
        transformed_env = TransformedEnv(env, vel_transform).train()
        transformed_env.set_seed(cfg.seed)   

        print("\n=== CARIC-RL Observation Space ===")
        print("State dimension:", cfg.env.get('state_dim', 12))
        print("LiDAR resolution:", f"{env.lidar_hbeams}x{env.lidar_vbeams}")
        print("Local map size:", "64x64x3")
        print("Max visible interest points:", 20)
        print("Scan state dimension:", 8)
        print(transformed_env.observation_spec) 
        
        # CARIC-Aware PPO Policy initialization
        print("\n=== Initializing CARIC-Aware PPO Policy ===")
        policy = CARICAwarePPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
        print("=== CARIC-Aware PPO Policy initialized successfully ===\n")
        
        # Episode statistics collector (CARIC-specific metrics)
        episode_stats_keys = [
            k for k in transformed_env.observation_spec.keys(True, True) 
            if isinstance(k, tuple) and k[0]=="stats"
        ]
        episode_stats = EpisodeStats(episode_stats_keys)

        # Data collector
        collector = SyncDataCollector(
            transformed_env,
            policy=policy, 
            frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
            total_frames=cfg.max_frame_num,
            device=cfg.device,
            return_same_td=True,
            exploration_type=ExplorationType.RANDOM
        )

        # CARIC-specific training metrics
        best_coverage = 0.0
        best_scanning_efficiency = 0.0
        best_interest_points_scanned = 0.0
        no_improvement_count = 0
        patience = cfg.get('early_stopping_patience', 50)
        
        def print_memory_usage():
            """Print current memory usage"""
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            process = psutil.Process(os.getpid())
            print(f"CPU Memory: {process.memory_info().rss/1024**3:.2f}GB")

        # CARIC-RL Training loop
        print("Starting CARIC-RL training loop...")
        print("="*60)
        print("Training Metrics:")
        print("- Total Coverage: Interest points scanned / Interest points found")
        print("- Scanning Efficiency: Points scanned per timestep")
        print("- Exploration Progress: Visited voxels / Total voxels")
        print("- Path Efficiency: Scanned points / Visited voxels")
        print("="*60)
        
        for i, data in enumerate(collector):
            try:
                # Periodic memory cleanup
                if i % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Training information
                info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

                # CARIC-Aware PPO training
                train_loss_stats = policy.train(data)
                info.update(train_loss_stats)

                # Episode statistics processing
                episode_stats.add(data)
                
                if len(episode_stats) > 0:
                    stats = {
                        "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                        for k, v in episode_stats.pop().items(True, True)
                    }
                    info.update(stats)

                # CARIC-specific progress printing
                if i % 20 == 0:
                    total_coverage = info.get("train/stats.total_coverage", 0.0)
                    interest_points_found = info.get("train/stats.interest_points_found", 0.0)
                    interest_points_scanned = info.get("train/stats.interest_points_scanned", 0.0)
                    scanning_efficiency = info.get("train/stats.scanning_efficiency", 0.0)
                    exploration_progress = info.get("train/stats.exploration_progress", 0.0)
                    actor_loss = info.get("actor_loss", 0.0)
                    critic_loss = info.get("critic_loss", 0.0)
                    
                    print(f"Step {i:4d} | Frames: {collector._frames:8d}")
                    print(f"  Coverage: {total_coverage:.3f} | Found: {interest_points_found:.0f} | Scanned: {interest_points_scanned:.0f}")
                    print(f"  Scan Eff: {scanning_efficiency:.4f} | Explore: {exploration_progress:.3f}")
                    print(f"  A_Loss: {actor_loss:.4f} | C_Loss: {critic_loss:.4f}")
                    
                    if i % 100 == 0:
                        print_memory_usage()

                # CARIC-RL Evaluation
                if i % cfg.eval_interval == 0 and i > 0:
                    print(f"\n[CARIC-RL Evaluation at step {i}]")
                    
                    try:
                        # Safe evaluation process
                        env.eval()
                        eval_info = evaluate(
                            env=transformed_env, 
                            policy=policy,
                            seed=cfg.seed, 
                            cfg=cfg,
                            exploration_type=ExplorationType.MEAN
                        )
                        env.train()
                        
                        info.update(eval_info)
                        eval_coverage = eval_info.get("eval/stats.total_coverage", 0.0)
                        eval_scanning_efficiency = eval_info.get("eval/stats.scanning_efficiency", 0.0)
                        eval_interest_points_scanned = eval_info.get("eval/stats.interest_points_scanned", 0.0)
                        eval_exploration_progress = eval_info.get("eval/stats.exploration_progress", 0.0)
                        
                        print(f"CARIC-RL Evaluation Results:")
                        print(f"  Coverage: {eval_coverage:.4f}")
                        print(f"  Scanning Efficiency: {eval_scanning_efficiency:.4f}")
                        print(f"  Interest Points Scanned: {eval_interest_points_scanned:.0f}")
                        print(f"  Exploration Progress: {eval_exploration_progress:.3f}")
                        
                        # Best model saving based on scanning efficiency (CARIC's key metric)
                        improvement = False
                        if eval_scanning_efficiency > best_scanning_efficiency:
                            best_scanning_efficiency = eval_scanning_efficiency
                            improvement = True
                        if eval_coverage > best_coverage:
                            best_coverage = eval_coverage
                            improvement = True
                        if eval_interest_points_scanned > best_interest_points_scanned:
                            best_interest_points_scanned = eval_interest_points_scanned
                            improvement = True
                        
                        if improvement:
                            no_improvement_count = 0
                            
                            best_ckpt_path = os.path.join(run.dir, "checkpoint_best_caric_rl.pt")
                            torch.save({
                                'policy_state_dict': policy.state_dict(),
                                'training_step': policy.training_step,
                                'best_coverage': best_coverage,
                                'best_scanning_efficiency': best_scanning_efficiency,
                                'best_interest_points_scanned': best_interest_points_scanned,
                                'config': cfg
                            }, best_ckpt_path)
                            
                            print(f"New best CARIC-RL model saved!")
                            print(f"  Best Coverage: {best_coverage:.4f}")
                            print(f"  Best Scanning Efficiency: {best_scanning_efficiency:.4f}")
                            
                            # Update wandb summary
                            wandb.run.summary["best_coverage"] = best_coverage
                            wandb.run.summary["best_scanning_efficiency"] = best_scanning_efficiency
                            wandb.run.summary["best_interest_points_scanned"] = best_interest_points_scanned
                        else:
                            no_improvement_count += 1
                            print(f"No improvement for {no_improvement_count} evaluations")
                        
                        # Early stopping check
                        if no_improvement_count >= patience:
                            print(f"Early stopping triggered after {no_improvement_count} evaluations without improvement")
                            break
                            
                    except Exception as e:
                        print(f"Evaluation failed: {e}")
                        traceback.print_exc()
                        env.train()

                # Periodic model saving
                if i % cfg.save_interval == 0 and i > 0:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_caric_rl_{i}.pt")
                    torch.save({
                        'policy_state_dict': policy.state_dict(),
                        'training_step': policy.training_step,
                        'config': cfg
                    }, ckpt_path)
                    print(f"CARIC-RL checkpoint saved at step {i}")

                # Log to wandb
                run.log(info)
                
            except Exception as e:
                print(f"Training step {i} failed: {e}")
                traceback.print_exc()
                # Attempt recovery
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # Save final CARIC-RL model
        final_ckpt_path = os.path.join(run.dir, "checkpoint_caric_rl_final.pt")
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'training_step': policy.training_step,
            'best_coverage': best_coverage,
            'best_scanning_efficiency': best_scanning_efficiency,
            'best_interest_points_scanned': best_interest_points_scanned,
            'config': cfg
        }, final_ckpt_path)
        print("Final CARIC-RL model saved")
        
        # Print final training summary
        print("\n" + "="*60)
        print("CARIC-RL TRAINING COMPLETE")
        print("="*60)
        print(f"Best Coverage Achieved: {best_coverage:.4f}")
        print(f"Best Scanning Efficiency: {best_scanning_efficiency:.4f}")
        print(f"Best Interest Points Scanned: {best_interest_points_scanned:.0f}")
        print(f"Total Training Steps: {i}")
        print(f"Total Frames Processed: {collector._frames}")
        print("\nCARI-RL vs Original CARIC Comparison:")
        print("- Original CARIC uses A* for path planning")
        print("- CARIC-RL uses learned policy for path planning")
        print("- Both use identical interest point detection")
        print("- Both use identical 3D mapping system")
        print("="*60)

    except Exception as e:
        print(f"CARIC-RL training failed with error: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up CARIC-RL training...")
        wandb.finish()
        sim_app.close()
        print("CARIC-RL training script finished.")

if __name__ == "__main__":
    main()