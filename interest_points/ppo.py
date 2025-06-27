# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tensordict.tensordict import TensorDict
# from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
# from einops.layers.torch import Rearrange
# from torchrl.modules import ProbabilisticActor
# from torchrl.envs.transforms import CatTensors
# from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch
# import math

# class CARICAwarePPO(TensorDictModuleBase):
#     """
#     专门为CARIC-RL设计的PPO策略
    
#     特点：
#     1. CARIC感知的特征提取（理解扫描任务）
#     2. 兴趣点注意力机制（优先处理可见兴趣点）
#     3. 扫描状态感知的价值估计
#     4. 简化控制（仅位置控制，无gimbal）
#     """
    
#     def __init__(self, cfg, observation_spec, action_spec, device):
#         super().__init__()
#         self.cfg = cfg
#         self.device = device
#         self.training_step = 0
        
#         # 动作维度（简化为4D：vx, vy, vz, vyaw）
#         self.n_agents, self.action_dim = action_spec.shape
        
#         # 构建网络组件
#         self._build_feature_encoders()
#         self._build_feature_fusion()
#         self._build_actor_critic()
#         self._setup_training()
        
#         # 网络初始化
#         self._initialize_networks(observation_spec)

#     def _build_feature_encoders(self):
#         """构建特征编码器"""
#         # LiDAR编码器（用于障碍物感知和路径规划）
#         self.lidar_encoder = nn.Sequential(
#             nn.LazyConv2d(16, kernel_size=5, stride=2, padding=2), nn.ELU(),
#             nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1), nn.ELU(),
#             nn.LazyConv2d(64, kernel_size=3, stride=2, padding=1), nn.ELU(),
#             Rearrange("n c h w -> n (c h w)"),
#             nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
#         ).to(self.device)
        
#         # 局部地图编码器（理解CARIC的建图信息）
#         self.map_encoder = nn.Sequential(
#             nn.LazyConv2d(32, kernel_size=5, stride=2, padding=2), nn.ELU(),
#             nn.LazyConv2d(64, kernel_size=3, stride=2, padding=1), nn.ELU(),
#             nn.LazyConv2d(128, kernel_size=3, stride=2, padding=1), nn.ELU(),
#             Rearrange("n c h w -> n (c h w)"),
#             nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
#             nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
#         ).to(self.device)
        
#         # 兴趣点注意力处理器（核心CARIC组件）
#         self.interest_attention = InterestPointProcessor(
#             point_dim=8, hidden_dim=64, output_dim=64
#         ).to(self.device)
        
#         # 状态编码器
#         self.state_encoder = nn.Sequential(
#             nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#         ).to(self.device)
        
#         # CARIC扫描状态编码器
#         self.scan_state_encoder = nn.Sequential(
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#             nn.LazyLinear(16), nn.LayerNorm(16), nn.ELU(),
#         ).to(self.device)

#     def _build_feature_fusion(self):
#         """构建特征融合网络"""
#         self.feature_extractor = TensorDictSequential(
#             # 各特征编码器
#             TensorDictModule(self.lidar_encoder, [("agents", "observation", "lidar")], ["_lidar_feat"]),
#             TensorDictModule(self.map_encoder, [("agents", "observation", "local_map")], ["_map_feat"]),
#             TensorDictModule(self.interest_attention, [("agents", "observation", "visible_interest_points")], ["_interest_feat"]),
#             TensorDictModule(self.state_encoder, [("agents", "observation", "state")], ["_state_feat"]),
#             TensorDictModule(self.scan_state_encoder, [("agents", "observation", "scan_state")], ["_scan_feat"]),
            
#             # 特征融合
#             CatTensors(["_lidar_feat", "_map_feat", "_interest_feat", "_state_feat", "_scan_feat"], 
#                       "_feature", del_keys=False),
            
#             # 最终特征处理
#             TensorDictModule(
#                 nn.Sequential(
#                     nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
#                     nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
#                 ), 
#                 ["_feature"], ["_feature"]
#             ),
#         ).to(self.device)

#     def _build_actor_critic(self):
#         """构建策略和价值网络"""
#         # 策略网络（扫描感知）
#         self.actor = ProbabilisticActor(
#             TensorDictModule(
#                 ScanAwareActor(self.action_dim), 
#                 ["_feature", "_scan_feat", "_interest_feat"], ["loc", "scale"]
#             ),
#             in_keys=["loc", "scale"],
#             out_keys=[("agents", "action")],
#             distribution_class=IndependentNormal,
#             return_log_prob=True
#         ).to(self.device)
        
#         # 价值网络（任务感知）
#         self.critic = TensorDictModule(
#             ScanAwareCritic(), 
#             ["_feature", "_scan_feat"], ["state_value"]
#         ).to(self.device)

#     def _setup_training(self):
#         """设置训练组件"""
#         # 价值归一化
#         self.value_norm = ValueNorm(1).to(self.device)
        
#         # GAE计算
#         self.gae = GAE(0.99, 0.95)
#         self.critic_loss_fn = nn.HuberLoss(delta=10)
        
#         # 优化器
#         self.feature_extractor_optim = torch.optim.Adam(
#             self.feature_extractor.parameters(), 
#             lr=self.cfg.feature_extractor.learning_rate
#         )
#         self.actor_optim = torch.optim.Adam(
#             self.actor.parameters(), 
#             lr=self.cfg.actor.learning_rate
#         )
#         self.critic_optim = torch.optim.Adam(
#             self.critic.parameters(), 
#             lr=self.cfg.critic.learning_rate
#         )

#     def _initialize_networks(self, observation_spec):
#         """初始化网络"""
#         # 使用虚拟输入初始化
#         dummy_input = observation_spec.zero()
#         self.__call__(dummy_input)
        
#         # Orthogonal初始化
#         def init_(module):
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, 0.01)
#                 nn.init.constant_(module.bias, 0.)
        
#         self.actor.apply(init_)
#         self.critic.apply(init_)

#     def __call__(self, tensordict):
#         """前向传播"""
#         # 特征提取
#         self.feature_extractor(tensordict)
        
#         # 策略输出
#         self.actor(tensordict)
        
#         # 价值估计
#         self.critic(tensordict)
        
#         return tensordict

#     def train(self, tensordict):
#         """训练PPO模型"""
#         self.training_step += 1
        
#         # 处理下一状态的价值估计
#         next_tensordict = tensordict["next"]
#         with torch.no_grad():
#             next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
#             next_values = self.critic(next_tensordict)["state_value"]
        
#         # 获取奖励和终止标志
#         rewards = tensordict["next", "agents", "reward"]
#         dones = tensordict["next", "terminated"]
        
#         # 获取当前价值
#         values = tensordict["state_value"]
#         values = self.value_norm.denormalize(values)
#         next_values = self.value_norm.denormalize(next_values)
        
#         # 计算GAE
#         adv, ret = self.gae(rewards, dones, values, next_values)
#         adv_mean = adv.mean()
#         adv_std = adv.std()
#         adv = (adv - adv_mean) / adv_std.clamp(1e-7)
        
#         # 更新价值归一化
#         self.value_norm.update(ret)
#         ret = self.value_norm.normalize(ret)
        
#         # 添加到tensordict
#         tensordict.set("adv", adv)
#         tensordict.set("ret", ret)
        
#         # 训练循环
#         infos = []
#         for epoch in range(self.cfg.training_epoch_num):
#             batch = make_batch(tensordict, self.cfg.num_minibatches)
#             for minibatch in batch:
#                 infos.append(self._update(minibatch))
        
#         infos = torch.stack(infos).to_tensordict()
#         infos = infos.apply(torch.mean, batch_size=[])
        
#         return {k: v.item() for k, v in infos.items()}

#     def _update(self, tensordict):
#         """更新策略和价值函数"""
#         # 前向传播
#         self.feature_extractor(tensordict)
        
#         # 当前策略分布
#         action_dist = self.actor.get_dist(tensordict)
#         log_probs = action_dist.log_prob(tensordict[("agents", "action")])
        
#         # 熵损失
#         action_entropy = action_dist.entropy()
#         entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)
        
#         # Actor损失 (PPO clip)
#         advantage = tensordict["adv"]
#         ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
#         surr1 = advantage * ratio
#         surr2 = advantage * ratio.clamp(
#             1. - self.cfg.actor.clip_ratio, 
#             1. + self.cfg.actor.clip_ratio
#         )
#         actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        
#         # Critic损失
#         b_value = tensordict["state_value"]
#         ret = tensordict["ret"]
#         value = self.critic(tensordict)["state_value"]
#         value_clipped = b_value + (value - b_value).clamp(
#             -self.cfg.critic.clip_ratio, 
#             self.cfg.critic.clip_ratio
#         )
#         critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
#         critic_loss_original = self.critic_loss_fn(ret, value)
#         critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        
#         # 总损失
#         loss = entropy_loss + actor_loss + critic_loss
        
#         # 反向传播
#         self.feature_extractor_optim.zero_grad()
#         self.actor_optim.zero_grad()
#         self.critic_optim.zero_grad()
        
#         loss.backward()
        
#         # 梯度裁剪
#         actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
#             self.actor.parameters(), max_norm=5.
#         )
#         critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
#             self.critic.parameters(), max_norm=5.
#         )
        
#         # 优化器步骤
#         self.feature_extractor_optim.step()
#         self.actor_optim.step()
#         self.critic_optim.step()
        
#         # 解释方差
#         explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        
#         return TensorDict({
#             "actor_loss": actor_loss,
#             "critic_loss": critic_loss,
#             "entropy": entropy_loss,
#             "actor_grad_norm": actor_grad_norm,
#             "critic_grad_norm": critic_grad_norm,
#             "explained_var": explained_var
#         }, [])


# class InterestPointProcessor(nn.Module):
#     """兴趣点处理器（类似CARIC的兴趣点注意力）"""
    
#     def __init__(self, point_dim=8, hidden_dim=64, output_dim=64):
#         super().__init__()
#         self.point_dim = point_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
        
#         # 兴趣点编码器
#         self.point_encoder = nn.Sequential(
#             nn.Linear(point_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ELU(),
#         )
        
#         # 基于扫描优先级的注意力
#         self.priority_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=4,
#             batch_first=True
#         )
        
#         # 输出层
#         self.output_layer = nn.Sequential(
#             nn.Linear(hidden_dim, output_dim),
#             nn.LayerNorm(output_dim),
#             nn.ELU(),
#         )
        
#         # 全局池化注意力
#         self.global_attention = nn.Sequential(
#             nn.Linear(hidden_dim, 1),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, interest_points):
#         """
#         Args:
#             interest_points: (batch_size, num_points, point_dim)
#         Returns:
#             aggregated_features: (batch_size, output_dim)
#         """
#         batch_size, num_points, _ = interest_points.shape
        
#         # 检查有效点（最后一维是有效标志）
#         point_mask = interest_points[..., -1] > 0.5
        
#         # 编码兴趣点
#         encoded_points = self.point_encoder(interest_points)
        
#         # 优先级注意力
#         attended_points, _ = self.priority_attention(
#             encoded_points, encoded_points, encoded_points,
#             key_padding_mask=~point_mask
#         )
        
#         # 应用mask
#         attended_points = attended_points * point_mask.unsqueeze(-1).float()
        
#         # 全局注意力池化
#         attention_weights = self.global_attention(attended_points)
#         attention_weights = attention_weights * point_mask.unsqueeze(-1).float()
        
#         # 归一化注意力权重
#         attention_sum = torch.sum(attention_weights, dim=1, keepdim=True).clamp_min(1e-6)
#         attention_weights = attention_weights / attention_sum
        
#         # 加权聚合
#         aggregated = torch.sum(attended_points * attention_weights, dim=1)
        
#         # 输出变换
#         output = self.output_layer(aggregated)
        
#         return output


# class ScanAwareActor(nn.Module):
#     """扫描感知的策略网络"""
    
#     def __init__(self, action_dim: int):
#         super().__init__()
#         self.action_dim = action_dim
        
#         # 主策略网络
#         self.policy_net = nn.Sequential(
#             nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
#             nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
#         )
        
#         # 扫描状态调制网络
#         self.scan_modulator = nn.Sequential(
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#             nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
#         )
        
#         # 兴趣点导向网络
#         self.interest_modulator = nn.Sequential(
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#             nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
#         )
        
#         # 动作输出头
#         self.action_head = nn.LazyLinear(action_dim)
        
#         # 动态标准差（基于扫描状态调整）
#         self.base_std = nn.Parameter(torch.zeros(action_dim))
#         self.scan_std_modulator = nn.Sequential(
#             nn.LazyLinear(8), nn.ELU(),
#             nn.LazyLinear(1), nn.Sigmoid(),
#         )
    
#     def forward(self, feature, scan_feat, interest_feat):
#         # 主策略特征
#         policy_feat = self.policy_net(feature)
        
#         # 扫描状态调制
#         scan_modulation = self.scan_modulator(scan_feat)
        
#         # 兴趣点调制
#         interest_modulation = self.interest_modulator(interest_feat)
        
#         # 融合特征
#         modulated_feat = policy_feat + scan_modulation + interest_modulation
        
#         # 动作输出
#         loc = self.action_head(modulated_feat)
        
#         # 动态标准差（有扫描目标时更精确，探索时更随机）
#         scan_urgency = self.scan_std_modulator(scan_feat)
#         base_scale = torch.exp(self.base_std)
        
#         # 有紧急扫描任务时降低随机性
#         scale = base_scale * (0.5 + 0.5 * (1.0 - scan_urgency))
#         scale = scale.expand_as(loc)
        
#         return loc, scale


# class ScanAwareCritic(nn.Module):
#     """扫描感知的价值网络"""
    
#     def __init__(self):
#         super().__init__()
        
#         # 主价值网络
#         self.main_value_net = nn.Sequential(
#             nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
#             nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
#         )
        
#         # 扫描进度价值评估
#         self.scan_value_net = nn.Sequential(
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#             nn.LazyLinear(16), nn.LayerNorm(16), nn.ELU(),
#         )
        
#         # 价值融合
#         self.value_fusion = nn.Sequential(
#             nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
#             nn.LazyLinear(1),
#         )
    
#     def forward(self, feature, scan_feat):
#         # 主特征价值
#         main_value_feat = self.main_value_net(feature)
        
#         # 扫描状态价值
#         scan_value_feat = self.scan_value_net(scan_feat)
        
#         # 融合价值估计
#         combined_feat = torch.cat([main_value_feat, scan_value_feat], dim=-1)
#         value = self.value_fusion(combined_feat)
        
#         return value

# # 为了向后兼容，提供BuildingCoveragePPO别名
# BuildingCoveragePPO = CARICAwarePPO




import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, make_batch
import math

class IndependentBeta(torch.distributions.Independent):
    """Beta distribution for bounded actions [0,1]"""
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive, 
        "beta": torch.distributions.constraints.positive
    }

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class CARICBetaActor(nn.Module):
    """CARIC-aware Beta Actor for stable bounded actions"""
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        
        # Main policy network
        self.policy_net = nn.Sequential(
            nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
            nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
        )
        
        # Scan state modulation network
        self.scan_modulator = nn.Sequential(
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
            nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
        )
        
        # Interest point guidance network
        self.interest_modulator = nn.Sequential(
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
            nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
        )
        
        # Beta distribution parameter networks
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, feature, scan_feat, interest_feat):
        # Feature fusion
        policy_feat = self.policy_net(feature)
        scan_modulation = self.scan_modulator(scan_feat)
        interest_modulation = self.interest_modulator(interest_feat)
        
        modulated_feat = policy_feat + scan_modulation + interest_modulation
        
        # Beta distribution parameters (ensure positive and stable)
        alpha = 1. + self.alpha_softplus(self.alpha_layer(modulated_feat)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(modulated_feat)) + 1e-6
        
        # Scan urgency modulation
        scan_urgency = torch.sigmoid(torch.mean(scan_feat, dim=-1, keepdim=True))
        
        # When scanning target exists, make beta larger (more conservative)
        beta = beta + scan_urgency * 1.5
        
        return alpha, beta

class InterestPointProcessor(nn.Module):
    """Interest point processor with attention mechanism"""
    
    def __init__(self, point_dim=8, hidden_dim=64, output_dim=64):
        super().__init__()
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Point encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        
        # Priority-based attention
        self.priority_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
        )
        
        # Global pooling attention
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, interest_points):
        batch_size, num_points, _ = interest_points.shape
        
        # Check valid points (last dimension is validity flag)
        point_mask = interest_points[..., -1] > 0.5
        
        # Encode interest points
        encoded_points = self.point_encoder(interest_points)
        
        # Priority attention
        attended_points, _ = self.priority_attention(
            encoded_points, encoded_points, encoded_points,
            key_padding_mask=~point_mask
        )
        
        # Apply mask
        attended_points = attended_points * point_mask.unsqueeze(-1).float()
        
        # Global attention pooling
        attention_weights = self.global_attention(attended_points)
        attention_weights = attention_weights * point_mask.unsqueeze(-1).float()
        
        # Normalize attention weights
        attention_sum = torch.sum(attention_weights, dim=1, keepdim=True).clamp_min(1e-6)
        attention_weights = attention_weights / attention_sum
        
        # Weighted aggregation
        aggregated = torch.sum(attended_points * attention_weights, dim=1)
        
        # Output transformation
        output = self.output_layer(aggregated)
        
        return output

class ScanAwareCritic(nn.Module):
    """Scan-aware value network"""
    
    def __init__(self):
        super().__init__()
        
        # Main value network
        self.main_value_net = nn.Sequential(
            nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
            nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
        )
        
        # Scan progress value assessment
        self.scan_value_net = nn.Sequential(
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
            nn.LazyLinear(16), nn.LayerNorm(16), nn.ELU(),
        )
        
        # Value fusion
        self.value_fusion = nn.Sequential(
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
            nn.LazyLinear(1),
        )
    
    def forward(self, feature, scan_feat):
        # Main feature value
        main_value_feat = self.main_value_net(feature)
        
        # Scan state value
        scan_value_feat = self.scan_value_net(scan_feat)
        
        # Fuse value estimation
        combined_feat = torch.cat([main_value_feat, scan_value_feat], dim=-1)
        value = self.value_fusion(combined_feat)
        
        return value

class CARICAwarePPO(TensorDictModuleBase):
    """Improved CARIC-Aware PPO with Beta distribution and coordinate transformation"""
    
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.training_step = 0
        
        # Action dimension and limits
        self.n_agents, self.action_dim = action_spec.shape
        self.action_limit = getattr(cfg.actor, 'action_limit', 2.0)
        
        # Build network components
        self._build_feature_encoders()
        self._build_feature_fusion()
        self._build_actor_critic()
        self._setup_training()
        
        # Initialize networks
        self._initialize_networks(observation_spec)

    def _build_feature_encoders(self):
        """Build feature encoders"""
        # LiDAR encoder (for obstacle awareness and path planning)
        self.lidar_encoder = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.LazyConv2d(64, kernel_size=3, stride=2, padding=1), nn.ELU(),
            Rearrange("n c h w -> n (c h w)"),
            nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
        ).to(self.device)
        
        # Local map encoder (understand CARIC mapping information)
        self.map_encoder = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.LazyConv2d(64, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.LazyConv2d(128, kernel_size=3, stride=2, padding=1), nn.ELU(),
            Rearrange("n c h w -> n (c h w)"),
            nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
            nn.LazyLinear(128), nn.LayerNorm(128), nn.ELU(),
        ).to(self.device)
        
        # Interest point attention processor
        self.interest_attention = InterestPointProcessor(
            point_dim=8, hidden_dim=64, output_dim=64
        ).to(self.device)
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.LazyLinear(64), nn.LayerNorm(64), nn.ELU(),
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
        ).to(self.device)
        
        # CARIC scan state encoder
        self.scan_state_encoder = nn.Sequential(
            nn.LazyLinear(32), nn.LayerNorm(32), nn.ELU(),
            nn.LazyLinear(16), nn.LayerNorm(16), nn.ELU(),
        ).to(self.device)

    def _build_feature_fusion(self):
        """Build feature fusion network"""
        self.feature_extractor = TensorDictSequential(
            # Individual feature encoders
            TensorDictModule(self.lidar_encoder, [("agents", "observation", "lidar")], ["_lidar_feat"]),
            TensorDictModule(self.map_encoder, [("agents", "observation", "local_map")], ["_map_feat"]),
            TensorDictModule(self.interest_attention, [("agents", "observation", "visible_interest_points")], ["_interest_feat"]),
            TensorDictModule(self.state_encoder, [("agents", "observation", "state")], ["_state_feat"]),
            TensorDictModule(self.scan_state_encoder, [("agents", "observation", "scan_state")], ["_scan_feat"]),
            
            # Feature fusion
            CatTensors(["_lidar_feat", "_map_feat", "_interest_feat", "_state_feat", "_scan_feat"], 
                      "_feature", del_keys=False),
            
            # Final feature processing
            TensorDictModule(
                nn.Sequential(
                    nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
                    nn.LazyLinear(256), nn.LayerNorm(256), nn.ELU(),
                ), 
                ["_feature"], ["_feature"]
            ),
        ).to(self.device)

    def _build_actor_critic(self):
        """Build actor and critic networks"""
        # Policy network (scan-aware Beta distribution)
        self.actor = ProbabilisticActor(
            TensorDictModule(
                CARICBetaActor(self.action_dim), 
                ["_feature", "_scan_feat", "_interest_feat"], 
                ["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)
        
        # Value network (task-aware)
        self.critic = TensorDictModule(
            ScanAwareCritic(), 
            ["_feature", "_scan_feat"], 
            ["state_value"]
        ).to(self.device)

    def _setup_training(self):
        """Setup training components"""
        # Value normalization
        self.value_norm = ValueNorm(1).to(self.device)
        
        # GAE computation
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        
        # Optimizers with lower learning rates
        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), 
            lr=self.cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.cfg.critic.learning_rate
        )

    def _initialize_networks(self, observation_spec):
        """Initialize networks"""
        # Initialize with dummy input
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)
        
        # Orthogonal initialization
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        """Forward pass with coordinate transformation"""
        # Feature extraction
        self.feature_extractor(tensordict)
        
        # Policy output (Beta distribution, outputs [0,1])
        self.actor(tensordict)
        
        # Value estimation
        self.critic(tensordict)
        
        # Action transformation: [0,1] -> [-action_limit, action_limit]
        action_normalized = tensordict[("agents", "action_normalized")]
        actions = (2 * action_normalized * self.action_limit) - self.action_limit
        
        # Coordinate transformation: local action -> world coordinate action
        scan_direction = self._get_scan_direction(tensordict)
        if scan_direction is not None:
            actions_world = self._vec_to_world(actions, scan_direction)
            tensordict[("agents", "action")] = actions_world
        else:
            tensordict[("agents", "action")] = actions
        
        return tensordict
    
    def _get_scan_direction(self, tensordict):
        """Get scan target direction for coordinate transformation"""
        scan_state = tensordict[("agents", "observation", "scan_state")]
        has_target = scan_state[..., 5] > 0.5  # 6th element is has_target flag
        
        if torch.any(has_target):
            state = tensordict[("agents", "observation", "state")]
            target_direction = state[..., 8:11]  # Relative target direction
            target_direction[~has_target] = torch.tensor([1., 0., 0.], device=self.device)
            return target_direction
        return None
    
    def _vec_to_world(self, vec, goal_direction):
        """Vector coordinate transformation: local -> world"""
        if len(vec.size()) == 2:
            vec = vec.unsqueeze(1)
        
        # Normalize goal direction
        goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
        z_direction = torch.tensor([0, 0, 1.], device=vec.device)
        
        # Build coordinate system
        goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
        goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True)
        
        goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
        goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True)

        # Transform vector (only first 3 dimensions for vx, vy, vz)
        n = vec.size(0)
        vec_3d = vec[..., :3]  # Only transform vx, vy, vz
        
        vec_x_new = torch.bmm(vec_3d.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec_3d.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec_3d.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

        vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)
        
        # Keep vyaw unchanged if it exists
        if self.action_dim == 4:
            vec_new = torch.cat([vec_new, vec[..., 3:4]], dim=-1)
        
        return vec_new.squeeze(1)

    def train(self, tensordict):
        """Train PPO model"""
        self.training_step += 1
        
        # Process next state value estimation
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        
        # Get rewards and termination flags
        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]
        
        # Get current values
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)
        
        # Calculate GAE
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clamp(1e-7)
        
        # Update value normalization
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        
        # Add to tensordict
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        
        # Training loop
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict):
        """Update policy and value function"""
        # Forward pass
        self.feature_extractor(tensordict)
        
        # Current policy distribution
        action_dist = self.actor.get_dist(tensordict)
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])
        
        # Entropy loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)
        
        # Actor loss (PPO clip)
        advantage = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(
            1. - self.cfg.actor.clip_ratio, 
            1. + self.cfg.actor.clip_ratio
        )
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        
        # Critic loss
        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"]
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio, 
            self.cfg.critic.clip_ratio
        )
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        
        # Total loss
        loss = entropy_loss + actor_loss + critic_loss
        
        # Backward pass
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), max_norm=5.
        )
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), max_norm=5.
        )
        
        # Optimizer steps
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        
        # Explained variance
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

# Alias for backward compatibility
BuildingCoveragePPO = CARICAwarePPO