# import torch
# import einops
# import numpy as np
# import math
# from tensordict.tensordict import TensorDict, TensorDictBase
# from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
# from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
# import omni.isaac.orbit.sim as sim_utils
# from omni_drones.robots.drone import MultirotorBase
# from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns, Camera, CameraCfg
# from omni_drones.utils.torch import euler_to_quaternion, quat_axis
# from omni.isaac.orbit.assets import AssetBaseCfg
# from omni.isaac.orbit.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
# from omni.isaac.orbit.terrains.config.rough import HfDiscreteObstaclesTerrainCfg
# import time

# class CARICRLEnvironment(IsaacEnv):
#     """
#     CARIC core functions retained:

#     - Dynamic interest point detection (based on sensor data)

#     - 3D occupancy mapping

#     - Scan state management

#     RL replacement part:

#     - A* path planning → RL policy network

#     - Fixed rule exploration → learning path selection
#     """

#     def __init__(self, cfg):
#         print("[CARIC-RL]: Initializing hybrid environment...")
        
#         self.lidar_range = cfg.sensor.lidar_range
#         self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
#         self.lidar_vbeams = cfg.sensor.lidar_vbeams
#         self.lidar_hres = cfg.sensor.lidar_hres
#         self.lidar_hbeams = int(360/self.lidar_hres)
        
#         # CARIC
#         self.scan_distance = cfg.env.scan_distance  #scan distence
#         self.voxel_size = 0.5  # voxel size
#         self.exploration_threshold = 0.8  # exploration threshold
        
#         super().__init__(cfg, cfg.headless)
        
#         self.drone.initialize()
#         self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
#         self._setup_lidar()
#         self._setup_camera()
        
#         self._setup_caric_mapping()
#         self._setup_caric_interest_detection()
#         self._setup_caric_scanning_logic()
        
#         self._setup_rl_state_tracking()

#     def _design_scene(self):

#         drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
#         cfg = drone_model.cfg_cls(force_sensor=False)
#         self.drone = drone_model(cfg=cfg)
#         drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

#         self._setup_basic_scene()

#         self._create_unknown_buildings()

#     def _setup_basic_scene(self):

#         light = AssetBaseCfg(
#             prim_path="/World/light",
#             spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
#         )
#         light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)

#         cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(100., 100.))
#         cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

#         self.map_range = [40.0, 40.0, 20.0]

#     def _create_unknown_buildings(self):

        
#         terrain_cfg = TerrainImporterCfg(
#             num_envs=self.num_envs,
#             env_spacing=0.0,
#             prim_path="/World/unknown_buildings",
#             terrain_type="generator",
#             terrain_generator=TerrainGeneratorCfg(
#                 seed=np.random.randint(0, 10000),  # random seed for terrain generation
#                 size=(self.map_range[0]*2, self.map_range[1]*2),
#                 border_width=5.0,
#                 num_rows=1,
#                 num_cols=1,
#                 horizontal_scale=0.5,
#                 vertical_scale=0.5,
#                 sub_terrains={
#                     "complex_buildings": HfDiscreteObstaclesTerrainCfg(
#                         horizontal_scale=0.5,
#                         vertical_scale=0.5,
#                         border_width=2.0,
#                         num_obstacles=np.random.randint(8, 15),  # random number of obstacles
#                         obstacle_height_mode="range",
#                         obstacle_width_range=(2.0, 12.0),
#                         obstacle_height_range=[3.0, 6.0, 10.0, 15.0, 20.0],
#                         obstacle_height_probability=[0.3, 0.3, 0.2, 0.1, 0.1],
#                         platform_width=0.0,
#                     ),
#                 },
#             ),
#             debug_vis=False,
#         )
#         self.terrain_importer = TerrainImporter(terrain_cfg)

#     def _setup_lidar(self):

#         ray_caster_cfg = RayCasterCfg(
#             prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
#             offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
#             attach_yaw_only=True,
#             pattern_cfg=patterns.BpearlPatternCfg(
#                 horizontal_res=self.lidar_hres,
#                 vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams)
#             ),
#             debug_vis=False,
#             mesh_prim_paths=["/World/unknown_buildings"],
#         )
#         self.lidar = RayCaster(ray_caster_cfg)
#         self.lidar._initialize_impl()
#         self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)

#     def _setup_camera(self):

#         camera_cfg = CameraCfg(
#             prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
#             offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, -0.05), rot=(1.0, 0.0, 0.0, 0.0)),
#             spawn=sim_utils.PinholeCameraCfg(
#                 focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
#             ),
#             width=640, height=480,
#             data_types=["rgb", "distance_to_camera"],
#         )
#         self.camera = Camera(camera_cfg)
#         self.camera._initialize_impl()
        

#         self.camera_fov = 60.0  
#         self.camera_range = 15.0  # maximum range of the camera

#     def _setup_caric_mapping(self):

#         # 3D occupancy grid 
#         map_size = [int(2 * self.map_range[i] / self.voxel_size) for i in range(3)]
        
#         with torch.device(self.device):
#             # 占用栅格 (0: unknown, 1: free, 2: occupied)
#             self.occupancy_grid = torch.zeros(self.num_envs, *map_size, dtype=torch.long)
            

#             self.confidence_grid = torch.zeros(self.num_envs, *map_size)
            
#             # CARIC的visited_map)
#             self.visited_grid = torch.zeros(self.num_envs, *map_size, dtype=torch.bool)
            
#             #CARIC的interest_map)
#             self.interest_grid = torch.zeros(self.num_envs, *map_size)
            
#             # CARIC的scan_map)
#             self.map_center_idx = torch.tensor([s//2 for s in map_size], device=self.device)

#     def _setup_caric_interest_detection(self):

#         with torch.device(self.device):
#             # dynamic_interest_points）
#             self.max_interest_points = 100
#             self.interest_points = torch.zeros(self.num_envs, self.max_interest_points, 3)
#             self.interest_point_confidence = torch.zeros(self.num_envs, self.max_interest_points)
#             self.interest_point_type = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.long)
#             self.interest_point_scanned = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
#             self.interest_point_active = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
#             self.interest_point_visible = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
            
#             self.edge_detection_threshold = 0.3
#             self.corner_detection_threshold = 0.5
#             self.scan_completion_distance = 3.0  

#     def _setup_caric_scanning_logic(self):

#         with torch.device(self.device):
#             # current_target）
#             self.current_scan_target = torch.zeros(self.num_envs, 3)
#             self.has_scan_target = torch.zeros(self.num_envs, dtype=torch.bool)
            
#             # 
#             self.scan_progress = torch.zeros(self.num_envs, 1)
#             self.total_interest_points_found = torch.zeros(self.num_envs, 1)
#             self.total_points_scanned = torch.zeros(self.num_envs, 1)
            
#             # recent_positions
#             self.recent_positions = torch.zeros(self.num_envs, 10, 3)  # recent 10 positions
#             self.recent_position_idx = torch.zeros(self.num_envs, dtype=torch.long)

#     def _setup_rl_state_tracking(self):

#         with torch.device(self.device):
#             # exploration statistics
#             self.exploration_progress = torch.zeros(self.num_envs, 1)
#             self.coverage_efficiency = torch.zeros(self.num_envs, 1)
            
#             self.scanning_efficiency = torch.zeros(self.num_envs, 1)
#             self.path_efficiency = torch.zeros(self.num_envs, 1)

#     def _set_specs(self):

#         state_dim = 12  # position (x, y, z), velocity (vx, vy, vz), orientation (roll, pitch, yaw), battery level
#         lidar_channels = 1
#         map_channels = 3  # occupancy, confidence, interest
#         map_size = 64  #  64 grid cells
#         interest_points_dim = 8  #  position (x, y, z), confidence, type, scanned, active, visible
#         max_visible_points = 20
        
#         # 观察规范
#         self.observation_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "observation": CompositeSpec({

#                     "state": UnboundedContinuousTensorSpec((state_dim,), device=self.device),
#                     # LiDAR
#                     "lidar": UnboundedContinuousTensorSpec((lidar_channels, self.lidar_hbeams, self.lidar_vbeams), device=self.device),

#                     "local_map": UnboundedContinuousTensorSpec((map_channels, map_size, map_size), device=self.device),

#                     "visible_interest_points": UnboundedContinuousTensorSpec((max_visible_points, interest_points_dim), device=self.device),

#                     "scan_state": UnboundedContinuousTensorSpec((8,), device=self.device),
#                 })
#             }).expand(self.num_envs)
#         }, shape=[self.num_envs], device=self.device)
        
#         self.action_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "action": UnboundedContinuousTensorSpec((4,), device=self.device)  # [vx, vy, vz, vyaw]
#             })
#         }).expand(self.num_envs).to(self.device)
        
#         self.reward_spec = CompositeSpec({
#             "agents": CompositeSpec({
#                 "reward": UnboundedContinuousTensorSpec((1,))
#             })
#         }).expand(self.num_envs).to(self.device)

#         self.done_spec = CompositeSpec({
#             "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
#             "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
#             "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
#         }).expand(self.num_envs).to(self.device)
        
#         stats_spec = CompositeSpec({
#             "total_coverage": UnboundedContinuousTensorSpec(1),
#             "interest_points_found": UnboundedContinuousTensorSpec(1),
#             "interest_points_scanned": UnboundedContinuousTensorSpec(1),
#             "scanning_efficiency": UnboundedContinuousTensorSpec(1),
#             "path_efficiency": UnboundedContinuousTensorSpec(1),
#             "exploration_progress": UnboundedContinuousTensorSpec(1),
#         }).expand(self.num_envs).to(self.device)
        
#         self.observation_spec["stats"] = stats_spec
#         self.stats = stats_spec.zero()

#     def _reset_idx(self, env_ids: torch.Tensor):
#         self.drone._reset_idx(env_ids, self.training)
        
#         pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
#         pos[:, 0, 0] = torch.rand(len(env_ids), device=self.device) * 20 - 10  # -10 to 10
#         pos[:, 0, 1] = torch.rand(len(env_ids), device=self.device) * 20 - 10
#         pos[:, 0, 2] = torch.rand(len(env_ids), device=self.device) * 5 + 3   # 3 to 8
#         # pos[:, 0, 2] = 0.2 # 保持在0.2高度

#         rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
#         rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * math.pi
#         # rpy[..., 2] = torch.atan2(-pos[:, 0, 1], -pos[:, 0, 0]).unsqueeze(-1) # 面向原点
#         rot = euler_to_quaternion(rpy)
        
#         self.drone.set_world_poses(pos, rot, env_ids)
#         self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
#         self._reset_caric_state(env_ids)
        
#         self.stats[env_ids] = 0

#     def _reset_caric_state(self, env_ids):
#         # 清空地图
#         self.occupancy_grid[env_ids] = 0
#         self.confidence_grid[env_ids] = 0
#         self.visited_grid[env_ids] = False
#         self.interest_grid[env_ids] = 0
        
#         # 清空兴趣点
#         self.interest_points[env_ids] = 0
#         self.interest_point_confidence[env_ids] = 0
#         self.interest_point_scanned[env_ids] = False
#         self.interest_point_active[env_ids] = False
#         self.interest_point_visible[env_ids] = False
        
#         # 重置扫描状态
#         self.current_scan_target[env_ids] = 0
#         self.has_scan_target[env_ids] = False
        
#         # 重置性能指标
#         self.exploration_progress[env_ids] = 0
#         self.total_interest_points_found[env_ids] = 0
#         self.total_points_scanned[env_ids] = 0

#     def _pre_sim_step(self, tensordict: TensorDictBase):

#         actions = tensordict[("agents", "action")]
        
#         self.drone.apply_action(actions)

#     def _post_sim_step(self, tensordict: TensorDictBase):

#         self.lidar.update(self.dt)
#         self.camera.update(self.dt)
        
#         # CARIC核心更新
#         self._update_caric_mapping()
#         self._update_caric_interest_detection()
#         self._update_caric_scanning_logic()

#     def _update_caric_mapping(self):

#         drone_pos = self.lidar.data.pos_w
#         hit_points = self.lidar.data.ray_hits_w
        
#         # 计算距离
#         distances = torch.norm(hit_points - drone_pos.unsqueeze(1), dim=-1)
#         distances = distances.clamp_max(self.lidar_range)
        
#         # 更新occupancy grid
#         self._update_occupancy_grid_caric_style(drone_pos, hit_points, distances)
        
#         # 更新访问状态
#         self._update_visited_status(drone_pos)

#     def _update_occupancy_grid_caric_style(self, drone_positions, hit_points, distances):

#         for env_idx in range(self.num_envs):
#             drone_pos = drone_positions[env_idx]
#             env_hit_points = hit_points[env_idx]
#             env_distances = distances[env_idx]
            
#             # 转换到体素坐标
#             drone_voxel = self._world_to_voxel(drone_pos)
            
#             for ray_idx in range(env_hit_points.shape[0]):
#                 hit_point = env_hit_points[ray_idx]
#                 distance = env_distances[ray_idx]
                
#                 if distance >= self.lidar_range - 0.1:
#                     continue
                
#                 # 标记占用点（类似CARIC的insert_point）
#                 hit_voxel = self._world_to_voxel(hit_point)
#                 if self._is_valid_voxel(hit_voxel):
#                     # 设置为占用
#                     self.occupancy_grid[env_idx, hit_voxel[0], hit_voxel[1], hit_voxel[2]] = 2
#                     self.confidence_grid[env_idx, hit_voxel[0], hit_voxel[1], hit_voxel[2]] += 0.1
                    
#                     # 更新周围的兴趣度（完全类似CARIC的logic）
#                     self._update_interest_around_obstacle(env_idx, hit_voxel)
                
#                 # 标记射线路径为自由空间
#                 self._mark_ray_as_free(env_idx, drone_pos, hit_point)

#     def _update_interest_around_obstacle(self, env_idx, obstacle_voxel):

#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 for dz in [-1, 0, 1]:
#                     if abs(dx) + abs(dy) + abs(dz) == 1:  # 6-连通邻域
#                         neighbor = obstacle_voxel + torch.tensor([dx, dy, dz], device=self.device)
#                         if self._is_valid_voxel(neighbor):
#                             # 如果邻居是未知区域且未访问过，设为兴趣点
#                             if (self.occupancy_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]] == 0 and
#                                 not self.visited_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]]):
#                                 self.interest_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]] = 1.0

#     def _update_caric_interest_detection(self):
#         """更新CARIC动态兴趣点检测"""
#         for env_idx in range(self.num_envs):
#             # 从interest_grid检测新的兴趣点
#             self._detect_and_add_interest_points(env_idx)
            
#             # 更新兴趣点可见性（基于相机FOV）
#             self._update_interest_point_visibility(env_idx)

#     def _detect_and_add_interest_points(self, env_idx):
#         """从兴趣度网格检测并添加新的兴趣点"""
#         # 找到所有兴趣点位置
#         interest_voxels = torch.where(self.interest_grid[env_idx] > 0.5)
        
#         if len(interest_voxels[0]) == 0:
#             return
        
#         # 转换为世界坐标
#         for i in range(len(interest_voxels[0])):
#             voxel_idx = torch.stack([interest_voxels[0][i], interest_voxels[1][i], interest_voxels[2][i]])
#             world_pos = self._voxel_to_world(voxel_idx)
#             confidence = self.interest_grid[env_idx, voxel_idx[0], voxel_idx[1], voxel_idx[2]]
            
#             # 检查是否已存在此兴趣点
#             if self._is_new_interest_point(env_idx, world_pos):
#                 self._add_interest_point(env_idx, world_pos, confidence, point_type=0)

#     def _is_new_interest_point(self, env_idx, world_pos, threshold=2.0):
#         """检查是否为新的兴趣点"""
#         active_mask = self.interest_point_active[env_idx]
#         if not active_mask.any():
#             return True
        
#         active_points = self.interest_points[env_idx, active_mask]
#         distances = torch.norm(active_points - world_pos, dim=-1)
#         return torch.all(distances > threshold)

#     def _add_interest_point(self, env_idx, position, confidence, point_type=0):

#         for i in range(self.max_interest_points):
#             if not self.interest_point_active[env_idx, i]:
#                 self.interest_points[env_idx, i] = position
#                 self.interest_point_confidence[env_idx, i] = confidence
#                 self.interest_point_type[env_idx, i] = point_type
#                 self.interest_point_active[env_idx, i] = True
#                 self.interest_point_scanned[env_idx, i] = False
                
#                 self.total_interest_points_found[env_idx] += 1
#                 break

#     def _update_interest_point_visibility(self, env_idx):
#         """更新兴趣点可见性（基于相机FOV和距离）"""
#         drone_pos = self.root_state[env_idx, 0, :3]
#         drone_quat = self.root_state[env_idx, 0, 3:7]

#         camera_forward = quat_axis(drone_quat.unsqueeze(0), axis=0).squeeze(0)  # x轴
        
#         active_mask = self.interest_point_active[env_idx]
#         active_points = self.interest_points[env_idx, active_mask]
        
#         if active_points.shape[0] == 0:
#             return
        
#         to_points = active_points - drone_pos
#         distances = torch.norm(to_points, dim=-1)
        
#         distance_mask = distances < self.camera_range
        
#         to_points_norm = to_points / distances.unsqueeze(-1).clamp_min(1e-6)
#         angles = torch.acos(torch.clamp(torch.dot(camera_forward, to_points_norm.T), -1, 1))
#         fov_mask = angles < (self.camera_fov * math.pi / 180 / 2)
        
#         visibility_mask = distance_mask & fov_mask

#         active_indices = torch.where(active_mask)[0]
#         self.interest_point_visible[env_idx] = False
#         self.interest_point_visible[env_idx, active_indices] = visibility_mask

#     def _update_caric_scanning_logic(self):
#         """更新CARIC扫描逻辑"""
#         for env_idx in range(self.num_envs):
#             drone_pos = self.root_state[env_idx, 0, :3]
            

#             if self.has_scan_target[env_idx]:
#                 target_distance = torch.norm(drone_pos - self.current_scan_target[env_idx])
#                 if target_distance < self.scan_completion_distance:

#                     self._complete_scanning(env_idx, drone_pos)
#                     self.has_scan_target[env_idx] = False
            

#             if not self.has_scan_target[env_idx]:
#                 self._select_next_scan_target(env_idx, drone_pos)

#     def _complete_scanning(self, env_idx, drone_pos):
#         """完成扫描，标记附近的兴趣点为已扫描"""
#         active_mask = self.interest_point_active[env_idx] & self.interest_point_visible[env_idx]
#         if not active_mask.any():
#             return
        
#         active_points = self.interest_points[env_idx, active_mask]
#         distances = torch.norm(active_points - drone_pos, dim=-1)
        
#         scanned_mask = distances < self.scan_completion_distance
        
#         if scanned_mask.any():

#             active_indices = torch.where(active_mask)[0]
#             scanned_indices = active_indices[scanned_mask]

#             self.interest_point_scanned[env_idx, scanned_indices] = True

#             self.total_points_scanned[env_idx] += torch.sum(scanned_mask.float())
            
#             for idx in scanned_indices:
#                 point_pos = self.interest_points[env_idx, idx]
#                 point_voxel = self._world_to_voxel(point_pos)
#                 if self._is_valid_voxel(point_voxel):
#                     self.interest_grid[env_idx, point_voxel[0], point_voxel[1], point_voxel[2]] = 0

#     def _select_next_scan_target(self, env_idx, drone_pos):

#         available_mask = (self.interest_point_active[env_idx] & 
#                          self.interest_point_visible[env_idx] & 
#                          ~self.interest_point_scanned[env_idx])
        
#         if not available_mask.any():

#             unscanned_mask = (self.interest_point_active[env_idx] & 
#                             ~self.interest_point_scanned[env_idx])
#             if unscanned_mask.any():
#                 unscanned_points = self.interest_points[env_idx, unscanned_mask]
#                 distances = torch.norm(unscanned_points - drone_pos, dim=-1)
#                 closest_idx = torch.argmin(distances)
                
#                 unscanned_indices = torch.where(unscanned_mask)[0]
#                 target_idx = unscanned_indices[closest_idx]
#                 self.current_scan_target[env_idx] = self.interest_points[env_idx, target_idx]
#                 self.has_scan_target[env_idx] = True
#             return

#         available_points = self.interest_points[env_idx, available_mask]
#         available_confidence = self.interest_point_confidence[env_idx, available_mask]
        
#         distances = torch.norm(available_points - drone_pos, dim=-1)

#         scores = available_confidence / (distances + 1.0)  # 避免除零
        
#         best_idx = torch.argmax(scores)
#         available_indices = torch.where(available_mask)[0]
#         target_idx = available_indices[best_idx]
        
#         self.current_scan_target[env_idx] = self.interest_points[env_idx, target_idx]
#         self.has_scan_target[env_idx] = True

#     def _compute_state_and_obs(self):
#         """计算状态和观察"""

#         self.root_state = self.drone.get_state(env_frame=False)

#         self._update_lidar_scan()

#         obs = self._compute_caric_observations()

#         self._compute_caric_rewards()

#         self._update_caric_stats()
        
#         return TensorDict({
#             "agents": TensorDict({
#                 "observation": obs,
#             }, [self.num_envs]),
#             "stats": self.stats.clone(),
#         }, self.batch_size)

#     def _update_lidar_scan(self):
#         """更新LiDAR扫描数据"""
#         drone_pos = self.lidar.data.pos_w
#         hit_points = self.lidar.data.ray_hits_w
        
#         distances = torch.norm(hit_points - drone_pos.unsqueeze(1), dim=-1)
#         distances = distances.clamp_max(self.lidar_range)
        
#         self.lidar_scan = (self.lidar_range - distances).reshape(
#             self.num_envs, 1, *self.lidar_resolution
#         )

#     def _compute_caric_observations(self):
#         """计算CARIC风格的观察"""
#         drone_pos = self.root_state[..., :3].squeeze(1)
#         drone_quat = self.root_state[..., 3:7].squeeze(1)
#         drone_vel = self.root_state[..., 7:10].squeeze(1)
        
#         state_obs = self._compute_state_observation(drone_pos, drone_quat, drone_vel)
        
#         local_map_obs = self._compute_local_map_observation(drone_pos)

#         visible_interest_obs = self._compute_visible_interest_points_observation(drone_pos)

#         scan_state_obs = self._compute_scan_state_observation(drone_pos)
        
#         return {
#             "state": state_obs,
#             "lidar": self.lidar_scan,
#             "local_map": local_map_obs,
#             "visible_interest_points": visible_interest_obs,
#             "scan_state": scan_state_obs,
#         }

#     def _compute_state_observation(self, drone_pos, drone_quat, drone_vel):
#         """计算基础状态观察"""

#         rel_scan_target = torch.zeros_like(drone_pos)
#         rel_scan_dist = torch.zeros(self.num_envs, 1, device=self.device)
        
#         for env_idx in range(self.num_envs):
#             if self.has_scan_target[env_idx]:
#                 rel_scan_target[env_idx] = self.current_scan_target[env_idx] - drone_pos[env_idx]
#                 rel_scan_dist[env_idx, 0] = torch.norm(rel_scan_target[env_idx])
#                 if rel_scan_dist[env_idx, 0] > 0:
#                     rel_scan_target[env_idx] = rel_scan_target[env_idx] / rel_scan_dist[env_idx, 0]
        
#         state_obs = torch.cat([
#             drone_pos,  # 3
#             drone_quat,  # 4
#             drone_vel,  # 3
#             rel_scan_target,  # 3 (归一化的目标方向)
#             rel_scan_dist.clamp_max(50),  # 1 (限制距离)
#         ], dim=-1)
        
#         return state_obs

#     def _compute_local_map_observation(self, drone_pos):
#         """计算局部地图观察"""
#         map_size = 64
#         local_maps = torch.zeros(self.num_envs, 3, map_size, map_size, device=self.device)
        
#         for env_idx in range(self.num_envs):
#             # 获取drone周围的地图区域
#             drone_voxel = self._world_to_voxel(drone_pos[env_idx])
            
#             # 提取局部区域
#             half_size = map_size // 2
#             x_start = max(0, drone_voxel[0] - half_size)
#             x_end = min(self.occupancy_grid.shape[1], drone_voxel[0] + half_size)
#             y_start = max(0, drone_voxel[1] - half_size)
#             y_end = min(self.occupancy_grid.shape[2], drone_voxel[1] + half_size)
            
#             # 选择当前高度层
#             z_idx = max(0, min(self.occupancy_grid.shape[3] - 1, drone_voxel[2]))
            
#             # 填充局部地图
#             actual_x_size = min(map_size, x_end - x_start)
#             actual_y_size = min(map_size, y_end - y_start)
            
#             if actual_x_size > 0 and actual_y_size > 0:
#                 # Occupancy channel
#                 local_maps[env_idx, 0, :actual_x_size, :actual_y_size] = \
#                     self.occupancy_grid[env_idx, x_start:x_start+actual_x_size, 
#                                       y_start:y_start+actual_y_size, z_idx].float()
                
#                 # Confidence channel
#                 local_maps[env_idx, 1, :actual_x_size, :actual_y_size] = \
#                     self.confidence_grid[env_idx, x_start:x_start+actual_x_size, 
#                                        y_start:y_start+actual_y_size, z_idx]
                
#                 # Interest channel
#                 local_maps[env_idx, 2, :actual_x_size, :actual_y_size] = \
#                     self.interest_grid[env_idx, x_start:x_start+actual_x_size, 
#                                      y_start:y_start+actual_y_size, z_idx]
        
#         return local_maps

#     def _compute_visible_interest_points_observation(self, drone_pos):
#         """计算可见兴趣点观察"""
#         max_points = 20
#         interest_obs = torch.zeros(self.num_envs, max_points, 8, device=self.device)
        
#         for env_idx in range(self.num_envs):

#             visible_mask = self.interest_point_visible[env_idx] & self.interest_point_active[env_idx]
#             visible_indices = torch.where(visible_mask)[0]
            
#             if len(visible_indices) > 0:
#                 visible_points = self.interest_points[env_idx, visible_indices]
#                 distances = torch.norm(visible_points - drone_pos[env_idx], dim=-1)
                
#                 # 按距离排序
#                 sorted_indices = torch.argsort(distances)
                
#                 # 取最近的点
#                 num_to_take = min(len(sorted_indices), max_points)
#                 for i in range(num_to_take):
#                     idx = visible_indices[sorted_indices[i]]
#                     rel_pos = self.interest_points[env_idx, idx] - drone_pos[env_idx]
                    
#                     interest_obs[env_idx, i] = torch.cat([
#                         rel_pos,  # 3: 相对位置
#                         torch.tensor([self.interest_point_type[env_idx, idx]], device=self.device, dtype=torch.float),  # 1: 类型
#                         torch.tensor([self.interest_point_confidence[env_idx, idx]], device=self.device),  # 1: 置信度
#                         torch.tensor([distances[sorted_indices[i]]], device=self.device),  # 1: 距离
#                         torch.tensor([1.0 if self.interest_point_scanned[env_idx, idx] else 0.0], device=self.device),  # 1: 是否已扫描
#                         torch.tensor([1.0], device=self.device),  # 1: 有效标志
#                     ])
        
#         return interest_obs

#     def _compute_scan_state_observation(self, drone_pos):
#         """计算CARIC扫描状态观察"""
#         scan_state = torch.zeros(self.num_envs, 8, device=self.device)
        
#         for env_idx in range(self.num_envs):
#             # 扫描进度信息
#             total_active = torch.sum(self.interest_point_active[env_idx].float())
#             total_scanned = torch.sum(self.interest_point_scanned[env_idx].float())
#             total_visible = torch.sum(self.interest_point_visible[env_idx].float())
            
#             scan_progress = total_scanned / total_active.clamp_min(1)
#             visibility_ratio = total_visible / total_active.clamp_min(1)
            
#             # 距离最近未扫描点的距离
#             unscanned_mask = (self.interest_point_active[env_idx] & 
#                             ~self.interest_point_scanned[env_idx])
#             nearest_unscanned_dist = 0.0
#             if unscanned_mask.any():
#                 unscanned_points = self.interest_points[env_idx, unscanned_mask]
#                 distances = torch.norm(unscanned_points - drone_pos[env_idx], dim=-1)
#                 nearest_unscanned_dist = torch.min(distances).item()
            
#             scan_state[env_idx] = torch.tensor([
#                 scan_progress,  # 扫描进度
#                 visibility_ratio,  # 可见性比例
#                 total_active / 100.0,  # 总活跃点数（归一化）
#                 total_scanned / 100.0,  # 总扫描数（归一化）
#                 nearest_unscanned_dist / 50.0,  # 最近未扫描点距离（归一化）
#                 1.0 if self.has_scan_target[env_idx] else 0.0,  # 是否有扫描目标
#                 0.0,  
#                 0.0,  
#             ], device=self.device)
        
#         return scan_state

#     def _compute_caric_rewards(self):
#         """计算CARIC风格的奖励"""
#         drone_pos = self.root_state[..., :3].squeeze(1)
        
#         # find rward
#         discovery_reward = self._compute_discovery_reward()
        
#         # finished scanning reward
#         scanning_reward = self._compute_scanning_reward()
        
#         # exploration reward (new area exploration)
#         exploration_reward = self._compute_exploration_reward()
        
#         # efficiency reward (based on scanning efficiency)
#         efficiency_reward = self._compute_efficiency_reward()
        
#         # collision penalty
#         collision_penalty = self._compute_collision_penalty()
        
#         # target reward (based on distance to scan target)
#         target_reward = self._compute_target_reward(drone_pos)
        
#         # total reward
#         self.reward = (
#             discovery_reward * 2.0 +      
#             scanning_reward * 5.0 +      
#             exploration_reward * 1.0 +    
#             efficiency_reward * 1.5 +     
#             collision_penalty +           
#             target_reward * 3.0           
#         )

#     def _compute_discovery_reward(self):
#         """计算兴趣点发现奖励"""

#         current_found = torch.sum(self.interest_point_active.float(), dim=1, keepdim=True)
#         prev_found = getattr(self, '_prev_found_points', torch.zeros_like(current_found))
        
#         discovery_reward = (current_found - prev_found).clamp_min(0)
#         self._prev_found_points = current_found.clone()
        
#         return discovery_reward

#     def _compute_scanning_reward(self):
#         """计算扫描完成奖励"""
#         current_scanned = torch.sum(self.interest_point_scanned.float(), dim=1, keepdim=True)
#         prev_scanned = getattr(self, '_prev_scanned_points', torch.zeros_like(current_scanned))
        
#         scanning_reward = (current_scanned - prev_scanned).clamp_min(0)
#         self._prev_scanned_points = current_scanned.clone()
        
#         return scanning_reward

#     def _compute_exploration_reward(self):
#         """计算探索奖励"""
#         exploration_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
#         for env_idx in range(self.num_envs):

#             total_voxels = self.visited_grid[env_idx].numel()
#             visited_voxels = torch.sum(self.visited_grid[env_idx])
#             current_exploration = visited_voxels.float() / total_voxels
            
#             prev_exploration = getattr(self, '_prev_exploration', torch.zeros(self.num_envs, 1, device=self.device))
#             exploration_reward[env_idx, 0] = (current_exploration - prev_exploration[env_idx, 0]).clamp_min(0)
#             prev_exploration[env_idx, 0] = current_exploration
        
#         if not hasattr(self, '_prev_exploration'):
#             self._prev_exploration = torch.zeros(self.num_envs, 1, device=self.device)
        
#         return exploration_reward

#     def _compute_efficiency_reward(self):
#         """计算效率奖励"""
#         efficiency_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
#         for env_idx in range(self.num_envs):
#             # scan efficiency
#             scanned_count = torch.sum(self.interest_point_scanned[env_idx])
#             time_steps = self.progress_buf[env_idx] + 1
            
#             efficiency = scanned_count.float() / time_steps.float()
#             efficiency_reward[env_idx, 0] = efficiency * 0.1  # 小的持续奖励
        
#         return efficiency_reward

#     def _compute_collision_penalty(self):
#         """计算碰撞惩罚"""
#         collision_penalty = torch.zeros(self.num_envs, 1, device=self.device)
        
#         # 检查LiDAR最近距离
#         min_distances = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
#         collision_mask = min_distances < 1.0  # 1米安全距离
        
#         collision_penalty[collision_mask] = -10.0
        
#         return collision_penalty

#     def _compute_target_reward(self, drone_pos):
#         """计算目标导向奖励"""
#         target_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
#         for env_idx in range(self.num_envs):
#             if self.has_scan_target[env_idx]:
#                 # 距离目标的距离奖励
#                 distance = torch.norm(drone_pos[env_idx] - self.current_scan_target[env_idx])
#                 prev_distance = getattr(self, '_prev_target_distance', {}).get(env_idx, distance)
                
#                 # 接近目标时给予奖励
#                 distance_improvement = prev_distance - distance
#                 target_reward[env_idx, 0] = distance_improvement.clamp_min(0) * 0.5
                
#                 # 更新之前的距离
#                 if not hasattr(self, '_prev_target_distance'):
#                     self._prev_target_distance = {}
#                 self._prev_target_distance[env_idx] = distance
        
#         return target_reward

#     def _update_caric_stats(self):
#         """更新CARIC统计信息"""
#         # 总覆盖率
#         total_coverage = torch.zeros(self.num_envs, 1, device=self.device)
#         for env_idx in range(self.num_envs):
#             total_active = torch.sum(self.interest_point_active[env_idx])
#             total_scanned = torch.sum(self.interest_point_scanned[env_idx])
#             if total_active > 0:
#                 total_coverage[env_idx, 0] = total_scanned.float() / total_active.float()
        
#         # 扫描效率
#         scanning_efficiency = torch.zeros(self.num_envs, 1, device=self.device)
#         for env_idx in range(self.num_envs):
#             time_steps = self.progress_buf[env_idx] + 1
#             scanned_count = torch.sum(self.interest_point_scanned[env_idx])
#             scanning_efficiency[env_idx, 0] = scanned_count.float() / time_steps.float()
        
#         # 路径效率
#         path_efficiency = torch.zeros(self.num_envs, 1, device=self.device)
#         for env_idx in range(self.num_envs):
#             # 简化的路径效率计算
#             visited_count = torch.sum(self.visited_grid[env_idx])
#             scanned_count = torch.sum(self.interest_point_scanned[env_idx])
#             if visited_count > 0:
#                 path_efficiency[env_idx, 0] = scanned_count.float() / visited_count.float()
        
#         # 探索进度
#         exploration_progress = torch.zeros(self.num_envs, 1, device=self.device)
#         for env_idx in range(self.num_envs):
#             total_voxels = self.visited_grid[env_idx].numel()
#             visited_voxels = torch.sum(self.visited_grid[env_idx])
#             exploration_progress[env_idx, 0] = visited_voxels.float() / total_voxels
        
#         # 更新统计
#         self.stats["total_coverage"] = total_coverage
#         self.stats["interest_points_found"] = torch.sum(self.interest_point_active.float(), dim=1, keepdim=True)
#         self.stats["interest_points_scanned"] = torch.sum(self.interest_point_scanned.float(), dim=1, keepdim=True)
#         self.stats["scanning_efficiency"] = scanning_efficiency
#         self.stats["path_efficiency"] = path_efficiency
#         self.stats["exploration_progress"] = exploration_progress

#     def _world_to_voxel(self, world_pos):
#         """世界坐标转体素坐标"""
#         relative_pos = world_pos + torch.tensor([self.map_range[0], self.map_range[1], 0], device=self.device)
#         voxel_indices = (relative_pos / self.voxel_size).long()
#         return voxel_indices

#     def _voxel_to_world(self, voxel_indices):
#         """体素坐标转世界坐标"""
#         relative_pos = voxel_indices.float() * self.voxel_size
#         world_pos = relative_pos - torch.tensor([self.map_range[0], self.map_range[1], 0], device=self.device)
#         return world_pos

#     def _is_valid_voxel(self, voxel_indices):
#         """检查体素索引是否有效"""
#         return (voxel_indices >= 0).all() and (voxel_indices < torch.tensor(self.occupancy_grid.shape[1:], device=self.device)).all()

#     def _mark_ray_as_free(self, env_idx, start_pos, end_pos):
#         """标记射线路径为自由空间"""
#         direction = end_pos - start_pos
#         ray_length = torch.norm(direction)
#         if ray_length < 1e-6:
#             return
            
#         direction = direction / ray_length
        
#         num_samples = int((ray_length / (self.voxel_size * 0.5)).item()) + 1
#         for i in range(1, num_samples):
#             t = i * self.voxel_size * 0.5
#             if t >= ray_length:
#                 break
#             sample_pos = start_pos + direction * t
#             sample_voxel = self._world_to_voxel(sample_pos)
            
#             if self._is_valid_voxel(sample_voxel):
#                 # 只有在不是障碍物的情况下才标记为自由
#                 if self.occupancy_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] != 2:
#                     self.occupancy_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] = 1
#                     self.confidence_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] += 0.02

#     def _update_visited_status(self, drone_positions):
#         """更新访问状态"""
#         for env_idx in range(self.num_envs):
#             drone_voxel = self._world_to_voxel(drone_positions[env_idx])
#             if self._is_valid_voxel(drone_voxel):
#                 self.visited_grid[env_idx, drone_voxel[0], drone_voxel[1], drone_voxel[2]] = True

#     def _compute_reward_and_done(self):
#         """计算奖励和完成状态"""

#         drone_pos = self.root_state[..., :3].squeeze(1)
        
#         # collision detection
#         min_distances = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
#         collision = min_distances < 0.5
        
#         # out of bounds
#         out_of_bounds = (
#             (torch.abs(drone_pos[:, 0:1]) > self.map_range[0]) |
#             (torch.abs(drone_pos[:, 1:2]) > self.map_range[1]) |
#             (drone_pos[:, 2:3] < 0.5) |
#             (drone_pos[:, 2:3] > self.map_range[2])
#         )
        
#         # 任务完成检测（扫描覆盖率达到阈值）
#         mission_complete = self.stats["total_coverage"] > 0.90
        
#         self.terminated = collision | out_of_bounds | mission_complete
#         self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
#         return TensorDict(
#             {
#                 "agents": {
#                     "reward": self.reward
#                 },
#                 "done": self.terminated | self.truncated,
#                 "terminated": self.terminated,
#                 "truncated": self.truncated,
#             },
#             self.batch_size,
#         )


# BuildingCoverageEnv = CARICRLEnvironment
import torch
import einops
import numpy as np
import math
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omegaconf import OmegaConf
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns, Camera, CameraCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import (
    TerrainImporterCfg, 
    TerrainImporter, 
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
import time

class CARICRLEnvironment(IsaacEnv):
    """
    CARIC core functions retained:

    - Dynamic interest point detection (based on sensor data)

    - 3D occupancy mapping

    - Scan state management

    RL replacement part:

    - A* path planning → RL policy network

    - Fixed rule exploration → learning path selection
    """

    def __init__(self, cfg):
        print("[CARIC-RL]: Initializing hybrid environment...")
        
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        
        # CARIC parameters
        self.scan_distance = getattr(cfg.env, 'scan_distance', 3.0)  # default scan distance
        self.voxel_size = getattr(cfg.env, 'voxel_size', 0.5)  # default voxel size  
        self.exploration_threshold = getattr(cfg.env, 'exploration_threshold', 0.8)  # default exploration threshold

        
        super().__init__(cfg, cfg.headless)
        
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        
        self._setup_lidar()
        self._setup_camera()
        
        self._setup_caric_mapping()
        self._setup_caric_interest_detection()
        self._setup_caric_scanning_logic()
        
        self._setup_rl_state_tracking()

    def _design_scene(self):

        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        self._setup_basic_scene()

        self._create_unknown_buildings()

    def _setup_basic_scene(self):

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)

        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(100., 100.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [40.0, 40.0, 20.0]

    def _create_unknown_buildings(self):
        """Create unknown buildings with proper configuration"""
        try:
            terrain_cfg = TerrainImporterCfg(
                num_envs=self.num_envs,
                env_spacing=0.0,
                prim_path="/World/unknown_buildings",
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    seed=np.random.randint(0, 10000),
                    size=(self.map_range[0]*2, self.map_range[1]*2),
                    border_width=0.0,
                    num_rows=1,
                    num_cols=1,
                    horizontal_scale=0.5,
                    vertical_scale=0.5,
                    sub_terrains={
                        "complex_buildings": HfDiscreteObstaclesTerrainCfg(
                            horizontal_scale=0.5,
                            vertical_scale=0.5,
                            border_width=2.0,
                            num_obstacles=1,#np.random.randint(8, 15),

                            obstacle_height_mode="choice", 
                            obstacle_width_range=(2.0, 12.0),
                            obstacle_height_range=[5.0, 8.0, 12.0, 16.0, 20.0],  
                            obstacle_height_probability=[0.3, 0.3, 0.2, 0.1, 0.1],  
                            platform_width=0.0,
                        ),
                    },
                ),
                debug_vis=False,
            )
            self.terrain_importer = TerrainImporter(terrain_cfg)
            print("[INFO] Terrain created successfully")
            
        except Exception as e:
            print(f"[WARNING] Failed to create terrain: {e}")
            print("[INFO] Continuing without terrain...")

            self.terrain_importer = None
    
    def _setup_lidar(self):

        ray_caster_cfg = RayCasterCfg(

            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),  
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams)
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/unknown_buildings"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)

    def _setup_camera(self):
        """设置相机传感器 - 作为base_link的子节点"""
        camera_cfg = CameraCfg(

            prim_path="/World/envs/env_.*/Hummingbird_0/base_link/CameraSensor",
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, -0.05), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955
            ),
            width=640, 
            height=480,
            data_types=["rgb", "distance_to_camera"],
        )
        self.camera = Camera(camera_cfg)
        self.camera._initialize_impl()
        
        self.camera_fov = 60.0  
        self.camera_range = 15.0

    def _setup_caric_mapping(self):

        # 3D occupancy grid 
        map_size = [int(2 * self.map_range[i] / self.voxel_size) for i in range(3)]
        
        with torch.device(self.device):
            # occupancy grid (0: unknown, 1: free, 2: occupied)
            self.occupancy_grid = torch.zeros(self.num_envs, *map_size, dtype=torch.long)
            

            self.confidence_grid = torch.zeros(self.num_envs, *map_size)
            
            # CARIC visited_map
            self.visited_grid = torch.zeros(self.num_envs, *map_size, dtype=torch.bool)
            
            # CARIC interest_map
            self.interest_grid = torch.zeros(self.num_envs, *map_size)
            
            # CARIC scan_map
            self.map_center_idx = torch.tensor([s//2 for s in map_size], device=self.device)

    def _setup_caric_interest_detection(self):

        with torch.device(self.device):
            # dynamic interest points
            self.max_interest_points = 100
            self.interest_points = torch.zeros(self.num_envs, self.max_interest_points, 3)
            self.interest_point_confidence = torch.zeros(self.num_envs, self.max_interest_points)
            self.interest_point_type = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.long)
            self.interest_point_scanned = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
            self.interest_point_active = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
            self.interest_point_visible = torch.zeros(self.num_envs, self.max_interest_points, dtype=torch.bool)
            
            self.edge_detection_threshold = 0.3
            self.corner_detection_threshold = 0.5
            self.scan_completion_distance = 3.0  

    def _setup_caric_scanning_logic(self):

        with torch.device(self.device):
            # current target
            self.current_scan_target = torch.zeros(self.num_envs, 3)
            self.has_scan_target = torch.zeros(self.num_envs, dtype=torch.bool)
            
            # progress tracking
            self.scan_progress = torch.zeros(self.num_envs, 1)
            self.total_interest_points_found = torch.zeros(self.num_envs, 1)
            self.total_points_scanned = torch.zeros(self.num_envs, 1)
            
            # recent positions
            self.recent_positions = torch.zeros(self.num_envs, 10, 3)  # recent 10 positions
            self.recent_position_idx = torch.zeros(self.num_envs, dtype=torch.long)

    def _setup_rl_state_tracking(self):

        with torch.device(self.device):
            # exploration statistics
            self.exploration_progress = torch.zeros(self.num_envs, 1)
            self.coverage_efficiency = torch.zeros(self.num_envs, 1)
            
            self.scanning_efficiency = torch.zeros(self.num_envs, 1)
            self.path_efficiency = torch.zeros(self.num_envs, 1)

    def _set_specs(self):

        state_dim = 12  # position (x, y, z), velocity (vx, vy, vz), orientation (roll, pitch, yaw), battery level
        lidar_channels = 1
        map_channels = 3  # occupancy, confidence, interest
        map_size = 64  # 64x64 grid cells
        interest_points_dim = 8  # position (x, y, z), confidence, type, scanned, active, visible
        max_visible_points = 20
        
        # observation specs
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({

                    "state": UnboundedContinuousTensorSpec((state_dim,), device=self.device),
                    # LiDAR
                    "lidar": UnboundedContinuousTensorSpec((lidar_channels, self.lidar_hbeams, self.lidar_vbeams), device=self.device),

                    "local_map": UnboundedContinuousTensorSpec((map_channels, map_size, map_size), device=self.device),

                    "visible_interest_points": UnboundedContinuousTensorSpec((max_visible_points, interest_points_dim), device=self.device),

                    "scan_state": UnboundedContinuousTensorSpec((8,), device=self.device),
                })
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((4,), device=self.device)  # [vx, vy, vz, vyaw]
            })
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        
        stats_spec = CompositeSpec({
            "total_coverage": UnboundedContinuousTensorSpec(1),
            "interest_points_found": UnboundedContinuousTensorSpec(1),
            "interest_points_scanned": UnboundedContinuousTensorSpec(1),
            "scanning_efficiency": UnboundedContinuousTensorSpec(1),
            "path_efficiency": UnboundedContinuousTensorSpec(1),
            "exploration_progress": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset with action history cleanup"""
        self.drone._reset_idx(env_ids, self.training)
        
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = torch.rand(len(env_ids), device=self.device) * 20 - 10
        pos[:, 0, 1] = torch.rand(len(env_ids), device=self.device) * 20 - 10
        pos[:, 0, 2] = torch.rand(len(env_ids), device=self.device) * 5 + 3

        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        rpy[..., 2] = torch.rand(len(env_ids), 1, device=self.device) * 2 * math.pi
        rot = euler_to_quaternion(rpy)
        
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        self._reset_caric_state(env_ids)
        
        # Reset action history for specified environments
        if hasattr(self, 'prev_action'):
            self.prev_action[env_ids] = 0.0
        
        # Reset reward computation history
        if hasattr(self, '_prev_velocity'):
            self._prev_velocity[env_ids] = 0.0
        
        self.stats[env_ids] = self.stats[env_ids].zero_()

    def _reset_caric_state(self, env_ids):
        # clear maps
        self.occupancy_grid[env_ids] = 0
        self.confidence_grid[env_ids] = 0.0
        self.visited_grid[env_ids] = False
        self.interest_grid[env_ids] = 0.0
        
        # clear interest points
        self.interest_points[env_ids] = 0.0
        self.interest_point_confidence[env_ids] = 0.0
        self.interest_point_scanned[env_ids] = False
        self.interest_point_active[env_ids] = False
        self.interest_point_visible[env_ids] = False
        
        # reset scan state
        self.current_scan_target[env_ids] = 0.0
        self.has_scan_target[env_ids] = False
        
        # reset performance metrics
        self.exploration_progress[env_ids] = 0.0
        self.total_interest_points_found[env_ids] = 0.0
        self.total_points_scanned[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Pre-simulation step with action smoothing"""
        raw_actions = tensordict[("agents", "action")]
        
        # Action smoothing and limiting
        smoothed_actions = self._smooth_and_limit_actions(raw_actions)
        
        # Update tensordict with smoothed actions
        tensordict[("agents", "action")] = smoothed_actions
        
        self.drone.apply_action(smoothed_actions)

    def _smooth_and_limit_actions(self, actions):
        """Smooth and limit action changes for stability"""
        if not hasattr(self, 'prev_action'):
            self.prev_action = torch.zeros_like(actions)
            self.action_smoothing = 0.7  # Smoothing coefficient
            self.max_action_change = 1.0  # Maximum action change per step
        
        # Limit action change magnitude
        action_diff = actions - self.prev_action
        action_diff_clipped = torch.clamp(
            action_diff, 
            -self.max_action_change, 
            self.max_action_change
        )
        
        # Apply smoothing
        smoothed_actions = (
            self.action_smoothing * self.prev_action + 
            (1 - self.action_smoothing) * (self.prev_action + action_diff_clipped)
        )
        
        # Ensure final actions are within reasonable bounds
        # [vx, vy, vz, vyaw] limits
        action_limits = torch.tensor([3.0, 3.0, 3.0, 1.5], device=actions.device)
        smoothed_actions = torch.clamp(smoothed_actions, -action_limits, action_limits)
        
        # Update action history
        self.prev_action = smoothed_actions.clone()
        
        return smoothed_actions
    
    def _post_sim_step(self, tensordict: TensorDictBase):

        self.lidar.update(self.dt)
        self.camera.update(self.dt)
        
        # CARIC core updates
        self._update_caric_mapping()
        self._update_caric_interest_detection()
        self._update_caric_scanning_logic()

    def _update_caric_mapping(self):

        drone_pos = self.lidar.data.pos_w
        hit_points = self.lidar.data.ray_hits_w
        
        # calculate distances
        distances = torch.norm(hit_points - drone_pos.unsqueeze(1), dim=-1)
        distances = distances.clamp_max(self.lidar_range)
        
        # update occupancy grid
        self._update_occupancy_grid_caric_style(drone_pos, hit_points, distances)
        
        # update visited status
        self._update_visited_status(drone_pos)

    def _update_occupancy_grid_caric_style(self, drone_positions, hit_points, distances):

        for env_idx in range(self.num_envs):
            drone_pos = drone_positions[env_idx]
            env_hit_points = hit_points[env_idx]
            env_distances = distances[env_idx]
            
            # convert to voxel coordinates
            drone_voxel = self._world_to_voxel(drone_pos)
            
            for ray_idx in range(env_hit_points.shape[0]):
                hit_point = env_hit_points[ray_idx]
                distance = env_distances[ray_idx]
                
                if distance >= self.lidar_range - 0.1:
                    continue
                
                # mark occupied point (similar to CARIC's insert_point)
                hit_voxel = self._world_to_voxel(hit_point)
                if self._is_valid_voxel(hit_voxel):
                    # set as occupied
                    self.occupancy_grid[env_idx, hit_voxel[0], hit_voxel[1], hit_voxel[2]] = 2
                    self.confidence_grid[env_idx, hit_voxel[0], hit_voxel[1], hit_voxel[2]] += 0.1
                    
                    # update interest around obstacle (exactly like CARIC logic)
                    self._update_interest_around_obstacle(env_idx, hit_voxel)
                
                # mark ray path as free space
                self._mark_ray_as_free(env_idx, drone_pos, hit_point)

    def _update_interest_around_obstacle(self, env_idx, obstacle_voxel):

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if abs(dx) + abs(dy) + abs(dz) == 1:  # 6-connectivity
                        neighbor = obstacle_voxel + torch.tensor([dx, dy, dz], device=self.device)
                        if self._is_valid_voxel(neighbor):
                            # if neighbor is unknown and unvisited, set as interest point
                            if (self.occupancy_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]] == 0 and
                                not self.visited_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]]):
                                self.interest_grid[env_idx, neighbor[0], neighbor[1], neighbor[2]] = 1.0

    def _update_caric_interest_detection(self):
        """Update CARIC dynamic interest point detection"""
        for env_idx in range(self.num_envs):
            # detect new interest points from interest_grid
            self._detect_and_add_interest_points(env_idx)
            
            # update interest point visibility (based on camera FOV)
            self._update_interest_point_visibility(env_idx)

    def _detect_and_add_interest_points(self, env_idx):
        """Detect and add new interest points from interest grid"""
        # find all interest point locations
        interest_voxels = torch.where(self.interest_grid[env_idx] > 0.5)
        
        if len(interest_voxels[0]) == 0:
            return
        
        # convert to world coordinates
        for i in range(len(interest_voxels[0])):
            voxel_idx = torch.stack([interest_voxels[0][i], interest_voxels[1][i], interest_voxels[2][i]])
            world_pos = self._voxel_to_world(voxel_idx)
            confidence = self.interest_grid[env_idx, voxel_idx[0], voxel_idx[1], voxel_idx[2]]
            
            # check if this interest point already exists
            if self._is_new_interest_point(env_idx, world_pos):
                self._add_interest_point(env_idx, world_pos, confidence, point_type=0)

    def _is_new_interest_point(self, env_idx, world_pos, threshold=2.0):
        """Check if this is a new interest point"""
        active_mask = self.interest_point_active[env_idx]
        if not active_mask.any():
            return True
        
        active_points = self.interest_points[env_idx, active_mask]
        distances = torch.norm(active_points - world_pos, dim=-1)
        return torch.all(distances > threshold)

    def _add_interest_point(self, env_idx, position, confidence, point_type=0):

        for i in range(self.max_interest_points):
            if not self.interest_point_active[env_idx, i]:
                self.interest_points[env_idx, i] = position
                self.interest_point_confidence[env_idx, i] = confidence
                self.interest_point_type[env_idx, i] = point_type
                self.interest_point_active[env_idx, i] = True
                self.interest_point_scanned[env_idx, i] = False
                
                self.total_interest_points_found[env_idx] += 1
                break

    def _update_interest_point_visibility(self, env_idx):
            """Update interest point visibility (based on camera FOV and distance)"""
            drone_pos = self.root_state[env_idx, 0, :3]
            drone_quat = self.root_state[env_idx, 0, 3:7]

            camera_forward = quat_axis(drone_quat.unsqueeze(0), axis=0).squeeze(0)  # x-axis
            
            active_mask = self.interest_point_active[env_idx]
            active_points = self.interest_points[env_idx, active_mask]
            
            if active_points.shape[0] == 0:
                return
            
            to_points = active_points - drone_pos
            distances = torch.norm(to_points, dim=-1)
            
            distance_mask = distances < self.camera_range
            
            to_points_norm = to_points / distances.unsqueeze(-1).clamp_min(1e-6)

            dot_products = torch.einsum('i,ji->j', camera_forward, to_points_norm)
            angles = torch.acos(torch.clamp(dot_products, -1, 1))
            
            fov_mask = angles < (self.camera_fov * math.pi / 180 / 2)
            
            visibility_mask = distance_mask & fov_mask

            active_indices = torch.where(active_mask)[0]
            self.interest_point_visible[env_idx] = False
            self.interest_point_visible[env_idx, active_indices] = visibility_mask

    def _update_caric_scanning_logic(self):
        """Update CARIC scanning logic"""
        for env_idx in range(self.num_envs):
            drone_pos = self.root_state[env_idx, 0, :3]
            

            if self.has_scan_target[env_idx]:
                target_distance = torch.norm(drone_pos - self.current_scan_target[env_idx])
                if target_distance < self.scan_completion_distance:

                    self._complete_scanning(env_idx, drone_pos)
                    self.has_scan_target[env_idx] = False
            

            if not self.has_scan_target[env_idx]:
                self._select_next_scan_target(env_idx, drone_pos)

    def _complete_scanning(self, env_idx, drone_pos):
        """Complete scanning, mark nearby interest points as scanned"""
        active_mask = self.interest_point_active[env_idx] & self.interest_point_visible[env_idx]
        if not active_mask.any():
            return
        
        active_points = self.interest_points[env_idx, active_mask]
        distances = torch.norm(active_points - drone_pos, dim=-1)
        
        scanned_mask = distances < self.scan_completion_distance
        
        if scanned_mask.any():

            active_indices = torch.where(active_mask)[0]
            scanned_indices = active_indices[scanned_mask]

            self.interest_point_scanned[env_idx, scanned_indices] = True

            self.total_points_scanned[env_idx] += torch.sum(scanned_mask.float())
            
            for idx in scanned_indices:
                point_pos = self.interest_points[env_idx, idx]
                point_voxel = self._world_to_voxel(point_pos)
                if self._is_valid_voxel(point_voxel):
                    self.interest_grid[env_idx, point_voxel[0], point_voxel[1], point_voxel[2]] = 0

    def _select_next_scan_target(self, env_idx, drone_pos):

        available_mask = (self.interest_point_active[env_idx] & 
                         self.interest_point_visible[env_idx] & 
                         ~self.interest_point_scanned[env_idx])
        
        if not available_mask.any():

            unscanned_mask = (self.interest_point_active[env_idx] & 
                            ~self.interest_point_scanned[env_idx])
            if unscanned_mask.any():
                unscanned_points = self.interest_points[env_idx, unscanned_mask]
                distances = torch.norm(unscanned_points - drone_pos, dim=-1)
                closest_idx = torch.argmin(distances)
                
                unscanned_indices = torch.where(unscanned_mask)[0]
                target_idx = unscanned_indices[closest_idx]
                self.current_scan_target[env_idx] = self.interest_points[env_idx, target_idx]
                self.has_scan_target[env_idx] = True
            return

        available_points = self.interest_points[env_idx, available_mask]
        available_confidence = self.interest_point_confidence[env_idx, available_mask]
        
        distances = torch.norm(available_points - drone_pos, dim=-1)

        scores = available_confidence / (distances + 1.0)  # avoid division by zero
        
        best_idx = torch.argmax(scores)
        available_indices = torch.where(available_mask)[0]
        target_idx = available_indices[best_idx]
        
        self.current_scan_target[env_idx] = self.interest_points[env_idx, target_idx]
        self.has_scan_target[env_idx] = True


    def _compute_state_and_obs(self):

            self.root_state = self.drone.get_state(env_frame=False)

            self._update_lidar_scan()

            obs = self._compute_caric_observations()

            self._compute_caric_rewards()

            self._update_caric_stats()

            drone_state = torch.cat([
                self.root_state[..., :3],   # 位置 (3)
                self.root_state[..., 3:7],  # 四元数 (4) 
                self.root_state[..., 7:10], # 线速度 (3)
                self.root_state[..., 10:13] # 角速度 (3)
            ], dim=-1)  # 总共13维
            
            return TensorDict({
                "agents": TensorDict({
                    "observation": obs,
                }, [self.num_envs]),
                "stats": self.stats.clone(),
                "info": TensorDict({
                    "drone_state": drone_state 
                }, [self.num_envs])
            }, self.batch_size)

    def _update_lidar_scan(self):
        """Update LiDAR scan data"""
        drone_pos = self.lidar.data.pos_w
        hit_points = self.lidar.data.ray_hits_w
        
        distances = torch.norm(hit_points - drone_pos.unsqueeze(1), dim=-1)
        distances = distances.clamp_max(self.lidar_range)
        
        self.lidar_scan = (self.lidar_range - distances).reshape(
            self.num_envs, 1, *self.lidar_resolution
        )

    def _compute_caric_observations(self):
        """Compute CARIC-style observations"""
        drone_pos = self.root_state[..., :3].squeeze(1)
        drone_quat = self.root_state[..., 3:7].squeeze(1)
        drone_vel = self.root_state[..., 7:10].squeeze(1)
        
        state_obs = self._compute_state_observation(drone_pos, drone_quat, drone_vel)
        
        local_map_obs = self._compute_local_map_observation(drone_pos)

        visible_interest_obs = self._compute_visible_interest_points_observation(drone_pos)

        scan_state_obs = self._compute_scan_state_observation(drone_pos)
        
        return {
            "state": state_obs,
            "lidar": self.lidar_scan,
            "local_map": local_map_obs,
            "visible_interest_points": visible_interest_obs,
            "scan_state": scan_state_obs,
        }

    def _compute_state_observation(self, drone_pos, drone_quat, drone_vel):
        """Compute basic state observation"""

        pos_obs = drone_pos / 10.0 # [x, y, z]

        yaw = torch.atan2(
            2 * (drone_quat[..., 0] * drone_quat[..., 3] + drone_quat[..., 1] * drone_quat[..., 2]),
            1 - 2 * (drone_quat[..., 2]**2 + drone_quat[..., 3]**2)
        ).unsqueeze(-1)
        
        pitch = torch.asin(
            torch.clamp(2 * (drone_quat[..., 0] * drone_quat[..., 2] - drone_quat[..., 3] * drone_quat[..., 1]), -1, 1)
        ).unsqueeze(-1)
        
        attitude_obs = torch.cat([yaw, pitch], dim=-1)  # [yaw, pitch]

        vel_obs = drone_vel / 3.0 # [vx, vy, vz]

        rel_scan_target = torch.zeros_like(drone_pos)
        for env_idx in range(self.num_envs):
            if self.has_scan_target[env_idx]:
                target_vec = self.current_scan_target[env_idx] - drone_pos[env_idx]
                target_dist = torch.norm(target_vec)
                if target_dist > 1e-6:
                    rel_scan_target[env_idx] = target_vec / target_dist  # 归一化方向向量
        
        has_target_obs = self.has_scan_target.float().unsqueeze(-1)  # [0 or 1]

        state_obs = torch.cat([
            pos_obs,           # 3维: 位置
            attitude_obs,      # 2维: yaw, pitch  
            vel_obs,           # 3维: 速度
            rel_scan_target,   # 3维: 目标方向
            has_target_obs,    # 1维: 是否有目标
        ], dim=-1)  # 总计: 12维
        
        return state_obs

    def _compute_local_map_observation(self, drone_pos):
        """Compute local map observation"""
        map_size = 64
        local_maps = torch.zeros(self.num_envs, 3, map_size, map_size, device=self.device)
        
        for env_idx in range(self.num_envs):
            # get map area around drone
            drone_voxel = self._world_to_voxel(drone_pos[env_idx])
            
            # extract local region
            half_size = map_size // 2
            x_start = max(0, drone_voxel[0] - half_size)
            x_end = min(self.occupancy_grid.shape[1], drone_voxel[0] + half_size)
            y_start = max(0, drone_voxel[1] - half_size)
            y_end = min(self.occupancy_grid.shape[2], drone_voxel[1] + half_size)
            
            # select current height layer
            z_idx = max(0, min(self.occupancy_grid.shape[3] - 1, drone_voxel[2]))
            
            # fill local map
            actual_x_size = min(map_size, x_end - x_start)
            actual_y_size = min(map_size, y_end - y_start)
            
            if actual_x_size > 0 and actual_y_size > 0:
                # Occupancy channel
                local_maps[env_idx, 0, :actual_x_size, :actual_y_size] = \
                    self.occupancy_grid[env_idx, x_start:x_start+actual_x_size, 
                                      y_start:y_start+actual_y_size, z_idx].float()
                
                # Confidence channel
                local_maps[env_idx, 1, :actual_x_size, :actual_y_size] = \
                    self.confidence_grid[env_idx, x_start:x_start+actual_x_size, 
                                       y_start:y_start+actual_y_size, z_idx]
                
                # Interest channel
                local_maps[env_idx, 2, :actual_x_size, :actual_y_size] = \
                    self.interest_grid[env_idx, x_start:x_start+actual_x_size, 
                                     y_start:y_start+actual_y_size, z_idx]
        
        return local_maps

    def _compute_visible_interest_points_observation(self, drone_pos):
        """Compute visible interest points observation"""
        max_points = 20
        interest_obs = torch.zeros(self.num_envs, max_points, 8, device=self.device)
        
        for env_idx in range(self.num_envs):

            visible_mask = self.interest_point_visible[env_idx] & self.interest_point_active[env_idx]
            visible_indices = torch.where(visible_mask)[0]
            
            if len(visible_indices) > 0:
                visible_points = self.interest_points[env_idx, visible_indices]
                distances = torch.norm(visible_points - drone_pos[env_idx], dim=-1)
                
                # sort by distance
                sorted_indices = torch.argsort(distances)
                
                # take closest points
                num_to_take = min(len(sorted_indices), max_points)
                for i in range(num_to_take):
                    idx = visible_indices[sorted_indices[i]]
                    rel_pos = self.interest_points[env_idx, idx] - drone_pos[env_idx]
                    
                    interest_obs[env_idx, i] = torch.cat([
                        rel_pos,  # 3: relative position
                        torch.tensor([self.interest_point_type[env_idx, idx]], device=self.device, dtype=torch.float),  # 1: type
                        torch.tensor([self.interest_point_confidence[env_idx, idx]], device=self.device),  # 1: confidence
                        torch.tensor([distances[sorted_indices[i]]], device=self.device),  # 1: distance
                        torch.tensor([1.0 if self.interest_point_scanned[env_idx, idx] else 0.0], device=self.device),  # 1: scanned
                        torch.tensor([1.0], device=self.device),  # 1: valid flag
                    ])
        
        return interest_obs

    def _compute_scan_state_observation(self, drone_pos):
        """Compute CARIC scan state observation"""
        scan_state = torch.zeros(self.num_envs, 8, device=self.device)
        
        for env_idx in range(self.num_envs):
            # scan progress information
            total_active = torch.sum(self.interest_point_active[env_idx].float())
            total_scanned = torch.sum(self.interest_point_scanned[env_idx].float())
            total_visible = torch.sum(self.interest_point_visible[env_idx].float())
            
            scan_progress = total_scanned / total_active.clamp_min(1)
            visibility_ratio = total_visible / total_active.clamp_min(1)
            
            # distance to nearest unscanned point
            unscanned_mask = (self.interest_point_active[env_idx] & 
                            ~self.interest_point_scanned[env_idx])
            nearest_unscanned_dist = 0.0
            if unscanned_mask.any():
                unscanned_points = self.interest_points[env_idx, unscanned_mask]
                distances = torch.norm(unscanned_points - drone_pos[env_idx], dim=-1)
                nearest_unscanned_dist = torch.min(distances).item()
            
            scan_state[env_idx] = torch.tensor([
                scan_progress,  # scan progress
                visibility_ratio,  # visibility ratio
                total_active / 100.0,  # total active points (normalized)
                total_scanned / 100.0,  # total scanned (normalized)
                nearest_unscanned_dist / 50.0,  # nearest unscanned distance (normalized)
                1.0 if self.has_scan_target[env_idx] else 0.0,  # has scan target
                0.0,  
                0.0,  
            ], device=self.device)
        
        return scan_state

    def _compute_caric_rewards(self):
        """Improved CARIC reward calculation - balanced and stable design"""
        drone_pos = self.root_state[..., :3].squeeze(1)
        drone_vel = self.root_state[..., 7:10].squeeze(1)
        drone_angvel = self.root_state[..., 10:13].squeeze(1)
        
        # Base survival reward (inspired by navigation project)
        base_reward = torch.ones(self.num_envs, 1, device=self.device) * 0.1
        
        # Target-oriented reward (smooth version)
        target_reward = self._compute_smooth_target_reward(drone_pos, drone_vel)
        
        # Discovery reward (significantly reduced weight)
        discovery_reward = self._compute_discovery_reward() * 0.5  # From 2.0 to 0.5
        
        # Scanning completion reward (reduced weight)
        scanning_reward = self._compute_scanning_reward() * 1.5  # From 5.0 to 1.5
        
        # Exploration reward (small weight)
        exploration_reward = self._compute_exploration_reward() * 0.3
        
        # Flight smoothness reward (new)
        smoothness_reward = self._compute_smoothness_reward(drone_vel, drone_angvel)
        
        # Safety reward (logarithmic)
        safety_reward = self._compute_safety_reward()
        
        # Height reasonableness reward (smooth penalty)
        height_reward = self._compute_height_reward(drone_pos)
        
        # Progressive collision penalty (instead of hard -10)
        collision_penalty = self._compute_progressive_collision_penalty()
        
        # Balanced total reward (all weights are small and balanced)
        self.reward = (
            base_reward +                    # 0.1 base reward
            target_reward * 1.0 +           # 1.0 target guidance
            discovery_reward +              # 0.5 discovery  
            scanning_reward +               # 1.5 scanning
            exploration_reward +            # 0.3 exploration
            smoothness_reward * 0.5 +       # 0.5 smoothness
            safety_reward * 0.8 +           # 0.8 safety
            height_reward * 0.3 +           # 0.3 height reasonableness
            collision_penalty               # Progressive penalty
        )
    def _compute_smooth_target_reward(self, drone_pos, drone_vel):
        """Smooth target-oriented reward"""
        target_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
        for env_idx in range(self.num_envs):
            if self.has_scan_target[env_idx]:
                # Target direction
                target_vec = self.current_scan_target[env_idx] - drone_pos[env_idx]
                target_distance = torch.norm(target_vec)
                
                if target_distance > 1e-6:
                    target_direction = target_vec / target_distance
                    
                    # Velocity alignment reward (inspired by navigation project)
                    vel_alignment = torch.dot(drone_vel[env_idx], target_direction)
                    target_reward[env_idx, 0] = torch.clamp(vel_alignment, 0, 2.0) * 0.2
                    
                    # Distance improvement reward (smooth version)
                    prev_distance = getattr(self, '_prev_target_distance', {}).get(env_idx, target_distance)
                    distance_improvement = (prev_distance - target_distance) / max(prev_distance, 1.0)
                    target_reward[env_idx, 0] += torch.clamp(distance_improvement, -0.1, 0.2)
                    
                    # Update distance history
                    if not hasattr(self, '_prev_target_distance'):
                        self._prev_target_distance = {}
                    self._prev_target_distance[env_idx] = target_distance
        
        return target_reward
    
    def _compute_smoothness_reward(self, drone_vel, drone_angvel):
        """Flight smoothness reward (inspired by navigation project)"""
        # Velocity smoothness (avoid abrupt changes)
        if hasattr(self, '_prev_velocity'):
            vel_change = torch.norm(drone_vel - self._prev_velocity, dim=-1, keepdim=True)
            vel_smoothness = torch.exp(-vel_change / 1.0)  # Smaller change = higher reward
        else:
            vel_smoothness = torch.ones(self.num_envs, 1, device=self.device)
        
        # Angular velocity smoothness
        angvel_magnitude = torch.norm(drone_angvel, dim=-1, keepdim=True)
        angvel_smoothness = torch.exp(-angvel_magnitude / 0.5)
        
        self._prev_velocity = drone_vel.clone()
        
        return (vel_smoothness + angvel_smoothness) / 2.0
    
    def _compute_discovery_reward(self):
        """Discovery reward with reduced impact"""
        current_found = torch.sum(self.interest_point_active.float(), dim=1, keepdim=True)
        prev_found = getattr(self, '_prev_found_points', torch.zeros_like(current_found))
        
        discovery_reward = (current_found - prev_found).clamp_min(0)
        self._prev_found_points = current_found.clone()
        
        return discovery_reward
        
    def _compute_height_reward(self, drone_pos):
        """Height reasonableness reward (smooth penalty)"""
        height_penalty = torch.zeros(self.num_envs, 1, device=self.device)
        
        for env_idx in range(self.num_envs):
            current_height = drone_pos[env_idx, 2]
            min_height = 0.5  # Minimum safe height
            max_height = 8.0  # Maximum reasonable height
            
            # Smooth height penalty (instead of step function)
            if current_height < min_height:
                height_penalty[env_idx, 0] = -((min_height - current_height) ** 2) * 0.1
            elif current_height > max_height:
                height_penalty[env_idx, 0] = -((current_height - max_height) ** 2) * 0.1
        
        return height_penalty
    
    def _compute_progressive_collision_penalty(self):
        """Progressive collision penalty (instead of hard -10 penalty)"""
        # Calculate minimum obstacle distance
        min_distances = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
        
        # Progressive penalty: closer = higher penalty, but no sudden jumps
        collision_penalty = torch.zeros_like(min_distances)
        
        # Danger zone (0-1m): strong penalty
        danger_mask = min_distances < 1.0
        collision_penalty[danger_mask] = -2.0 * (1.0 - min_distances[danger_mask])
        
        # Warning zone (1-2m): light penalty
        warning_mask = (min_distances >= 1.0) & (min_distances < 2.0)
        collision_penalty[warning_mask] = -0.5 * (2.0 - min_distances[warning_mask])
        
        return collision_penalty 
       
    def _compute_safety_reward(self):
        """Safety reward with logarithmic function (inspired by navigation project)"""
        # Logarithmic safety reward for LiDAR data
        safety_distances = self.lidar_range - self.lidar_scan
        safety_reward = torch.log(safety_distances.clamp(min=0.1, max=self.lidar_range)).mean(dim=(2, 3))
        
        # Normalize to reasonable range
        safety_reward = safety_reward / 10.0  # Scale to reasonable range
        
        return safety_reward
    def _compute_scanning_reward(self):
        """Scanning completion reward with reduced impact"""
        current_scanned = torch.sum(self.interest_point_scanned.float(), dim=1, keepdim=True)
        prev_scanned = getattr(self, '_prev_scanned_points', torch.zeros_like(current_scanned))
        
        scanning_reward = (current_scanned - prev_scanned).clamp_min(0)
        self._prev_scanned_points = current_scanned.clone()
        
        return scanning_reward

    def _compute_exploration_reward(self):
        """Exploration reward with reduced impact"""
        exploration_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
        for env_idx in range(self.num_envs):
            total_voxels = self.visited_grid[env_idx].numel()
            visited_voxels = torch.sum(self.visited_grid[env_idx])
            current_exploration = visited_voxels.float() / total_voxels
            
            prev_exploration = getattr(self, '_prev_exploration', torch.zeros(self.num_envs, 1, device=self.device))
            exploration_reward[env_idx, 0] = (current_exploration - prev_exploration[env_idx, 0]).clamp_min(0)
            prev_exploration[env_idx, 0] = current_exploration
        
        if not hasattr(self, '_prev_exploration'):
            self._prev_exploration = torch.zeros(self.num_envs, 1, device=self.device)
        
        return exploration_reward

    def _compute_efficiency_reward(self):
        """Compute efficiency reward"""
        efficiency_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
        for env_idx in range(self.num_envs):
            # scan efficiency
            scanned_count = torch.sum(self.interest_point_scanned[env_idx])
            time_steps = self.progress_buf[env_idx] + 1
            
            efficiency = scanned_count.float() / time_steps.float()
            efficiency_reward[env_idx, 0] = efficiency * 0.1  # small continuous reward
        
        return efficiency_reward

    def _compute_collision_penalty(self):
        """Compute collision penalty"""
        collision_penalty = torch.zeros(self.num_envs, 1, device=self.device)
        
        # check LiDAR minimum distance
        min_distances = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
        collision_mask = min_distances < 1.0  # 1 meter safety distance
        
        collision_penalty[collision_mask] = -10.0
        
        return collision_penalty

    def _compute_target_reward(self, drone_pos):
        """Compute target-oriented reward"""
        target_reward = torch.zeros(self.num_envs, 1, device=self.device)
        
        for env_idx in range(self.num_envs):
            if self.has_scan_target[env_idx]:
                # distance reward to target
                distance = torch.norm(drone_pos[env_idx] - self.current_scan_target[env_idx])
                prev_distance = getattr(self, '_prev_target_distance', {}).get(env_idx, distance)
                
                # reward for approaching target
                distance_improvement = prev_distance - distance
                target_reward[env_idx, 0] = distance_improvement.clamp_min(0) * 0.5
                
                # update previous distance
                if not hasattr(self, '_prev_target_distance'):
                    self._prev_target_distance = {}
                self._prev_target_distance[env_idx] = distance
        
        return target_reward

    def _update_caric_stats(self):
        """Update CARIC statistics"""
        # total coverage
        total_coverage = torch.zeros(self.num_envs, 1, device=self.device)
        for env_idx in range(self.num_envs):
            total_active = torch.sum(self.interest_point_active[env_idx])
            total_scanned = torch.sum(self.interest_point_scanned[env_idx])
            if total_active > 0:
                total_coverage[env_idx, 0] = total_scanned.float() / total_active.float()
        
        # scanning efficiency
        scanning_efficiency = torch.zeros(self.num_envs, 1, device=self.device)
        for env_idx in range(self.num_envs):
            time_steps = self.progress_buf[env_idx] + 1
            scanned_count = torch.sum(self.interest_point_scanned[env_idx])
            scanning_efficiency[env_idx, 0] = scanned_count.float() / time_steps.float()
        
        # path efficiency
        path_efficiency = torch.zeros(self.num_envs, 1, device=self.device)
        for env_idx in range(self.num_envs):
            # simplified path efficiency calculation
            visited_count = torch.sum(self.visited_grid[env_idx])
            scanned_count = torch.sum(self.interest_point_scanned[env_idx])
            if visited_count > 0:
                path_efficiency[env_idx, 0] = scanned_count.float() / visited_count.float()
        
        # exploration progress
        exploration_progress = torch.zeros(self.num_envs, 1, device=self.device)
        for env_idx in range(self.num_envs):
            total_voxels = self.visited_grid[env_idx].numel()
            visited_voxels = torch.sum(self.visited_grid[env_idx])
            exploration_progress[env_idx, 0] = visited_voxels.float() / total_voxels
        
        # update statistics
        self.stats["total_coverage"] = total_coverage
        self.stats["interest_points_found"] = torch.sum(self.interest_point_active.float(), dim=1, keepdim=True)
        self.stats["interest_points_scanned"] = torch.sum(self.interest_point_scanned.float(), dim=1, keepdim=True)
        self.stats["scanning_efficiency"] = scanning_efficiency
        self.stats["path_efficiency"] = path_efficiency
        self.stats["exploration_progress"] = exploration_progress

    def _world_to_voxel(self, world_pos):
        """Convert world coordinates to voxel coordinates"""
        relative_pos = world_pos + torch.tensor([self.map_range[0], self.map_range[1], 0], device=self.device)
        voxel_indices = (relative_pos / self.voxel_size).long()
        return voxel_indices

    def _voxel_to_world(self, voxel_indices):
        """Convert voxel coordinates to world coordinates"""
        relative_pos = voxel_indices.float() * self.voxel_size
        world_pos = relative_pos - torch.tensor([self.map_range[0], self.map_range[1], 0], device=self.device)
        return world_pos

    def _is_valid_voxel(self, voxel_indices):
        """Check if voxel indices are valid"""
        return (voxel_indices >= 0).all() and (voxel_indices < torch.tensor(self.occupancy_grid.shape[1:], device=self.device)).all()

    def _mark_ray_as_free(self, env_idx, start_pos, end_pos):
        """Mark ray path as free space"""
        direction = end_pos - start_pos
        ray_length = torch.norm(direction)
        if ray_length < 1e-6:
            return
            
        direction = direction / ray_length
        
        num_samples = int((ray_length / (self.voxel_size * 0.5)).item()) + 1
        for i in range(1, num_samples):
            t = i * self.voxel_size * 0.5
            if t >= ray_length:
                break
            sample_pos = start_pos + direction * t
            sample_voxel = self._world_to_voxel(sample_pos)
            
            if self._is_valid_voxel(sample_voxel):
                # only mark as free if not already an obstacle
                if self.occupancy_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] != 2:
                    self.occupancy_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] = 1
                    self.confidence_grid[env_idx, sample_voxel[0], sample_voxel[1], sample_voxel[2]] += 0.02

    def _update_visited_status(self, drone_positions):
        """Update visited status"""
        for env_idx in range(self.num_envs):
            drone_voxel = self._world_to_voxel(drone_positions[env_idx])
            if self._is_valid_voxel(drone_voxel):
                self.visited_grid[env_idx, drone_voxel[0], drone_voxel[1], drone_voxel[2]] = True

    def _compute_reward_and_done(self):

            drone_pos = self.root_state[..., :3].squeeze(1)
            
            # collision detection
            min_distances = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "min")
            collision = min_distances < 0.5
            
            # out of bounds
            out_of_bounds = (
                (torch.abs(drone_pos[:, 0:1]) > self.map_range[0]) |
                (torch.abs(drone_pos[:, 1:2]) > self.map_range[1]) |
                (drone_pos[:, 2:3] < 0.5) |
                (drone_pos[:, 2:3] > self.map_range[2])
            )
            
            # mission completion detection (scan coverage reaches threshold)
            mission_complete = self.stats["total_coverage"] > 0.90
            
            self.terminated = collision | out_of_bounds | mission_complete
            self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            
            drone_state = torch.cat([
                self.root_state[..., :3],   # 位置 (3)
                self.root_state[..., 3:7],  # 四元数 (4) 
                self.root_state[..., 7:10], # 线速度 (3)
                self.root_state[..., 10:13] # 角速度 (3)
            ], dim=-1) 
            
            return TensorDict(
                {
                    "agents": {
                        "reward": self.reward
                    },
                    "done": self.terminated | self.truncated,
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                    "info": TensorDict({
                        "drone_state": drone_state
                    }, [self.num_envs])
                },
                self.batch_size,
            )

# Alias for compatibility
BuildingCoverageEnv = CARICRLEnvironment