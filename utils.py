import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type
import math

class ValueNorm(nn.Module):
    """Value normalization for stable training"""
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    """Create MLP with layer normalization"""
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    """Independent normal distribution with clamped scale"""
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    """Basic actor network"""
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class GAE(nn.Module):
    """Generalized Advantage Estimation"""
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    """Create minibatches for training"""
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):
    """Evaluate policy performance"""
    env.enable_render(True)
    env.eval()
    env.set_seed(seed)

    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    
    env.enable_render(not cfg.headless)
    env.reset()
    
    done = trajs.get(("next", "done")) 
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # Log video with coverage visualization
    info["recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"), 
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
        format="mp4"
    )
    
    env.train()
    return info

def vec_to_new_frame(vec, goal_direction):
    """Transform vector to new coordinate frame"""
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)
    return vec_new

def vec_to_world(vec, goal_direction):
    """Transform vector from local to world coordinates"""
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coordinate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel

def construct_input(start, end):
    """Construct input string for range"""
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"

# Building Coverage Specific Utilities
def calculate_coverage_metrics(interest_points_scanned, interest_points_visible, interest_point_types):
    """Calculate detailed coverage metrics for building inspection"""
    # Total coverage ratio
    total_visible = torch.sum(interest_points_visible.float(), dim=1, keepdim=True)
    total_scanned = torch.sum(interest_points_scanned.float(), dim=1, keepdim=True)
    coverage_ratio = total_scanned / total_visible.clamp(min=1)
    
    # Coverage by type (wall, corner, roof, entrance)
    coverage_by_type = torch.zeros(interest_points_scanned.shape[0], 4, device=interest_points_scanned.device)
    for type_idx in range(4):
        type_mask = (interest_point_types == type_idx) & interest_points_visible
        type_scanned = interest_points_scanned & type_mask
        coverage_by_type[:, type_idx] = torch.sum(type_scanned.float(), dim=1) / torch.sum(type_mask.float(), dim=1).clamp(min=1)
    
    # Sides covered (based on type coverage > threshold)
    sides_covered = torch.sum((coverage_by_type > 0.5).float(), dim=1, keepdim=True)
    
    return coverage_ratio, coverage_by_type, sides_covered

def calculate_scanning_efficiency(scanned_count, episode_length):
    """Calculate scanning efficiency (points per time step)"""
    return scanned_count / (episode_length + 1).clamp(min=1)

def calculate_path_efficiency(trajectory_positions, interest_points, scanned_mask):
    """Calculate path efficiency based on trajectory and scanning results"""
    if len(trajectory_positions) < 2:
        return torch.tensor(0.0)
    
    # Calculate total path length
    path_segments = trajectory_positions[1:] - trajectory_positions[:-1]
    total_path_length = torch.sum(torch.norm(path_segments, dim=-1))
    
    # Calculate unique scanned points
    unique_scanned = torch.sum(scanned_mask.float())
    
    # Efficiency: scanned points per unit distance
    efficiency = unique_scanned / total_path_length.clamp(min=1e-6)
    return efficiency

def detect_optimal_viewing_positions(building_positions, building_dimensions, scan_distance=4.0):
    """Generate optimal viewing positions for building inspection"""
    optimal_positions = []
    
    for pos, dim in zip(building_positions, building_dimensions):
        x_center, y_center, z_center = pos
        width, length, height = dim
        
        # Generate positions around building perimeter
        num_positions = 12
        for i in range(num_positions):
            angle = 2 * math.pi * i / num_positions
            
            # Position at optimal distance
            radius = max(width, length) / 2 + scan_distance
            x = x_center + radius * math.cos(angle)
            y = y_center + radius * math.sin(angle)
            z = z_center + height * 0.6  # Optimal height for inspection
            
            optimal_positions.append([x, y, z])
    
    return optimal_positions

def calculate_line_of_sight(drone_pos, target_pos, obstacles, obstacle_threshold=0.5):
    """Simple line of sight calculation between drone and target"""
    # Simplified implementation - in practice would use ray casting
    direction = target_pos - drone_pos
    distance = torch.norm(direction, dim=-1)
    
    # Check if any obstacles are too close to the line
    # This is a simplified check - real implementation would be more sophisticated
    line_of_sight = distance < 50.0  # Simple distance threshold
    
    return line_of_sight

def visualize_coverage_progress(interest_points, scanned_mask, drone_position, save_path=None):
    """Create visualization of coverage progress"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert tensors to numpy for plotting
    if torch.is_tensor(interest_points):
        points = interest_points.cpu().numpy()
        scanned = scanned_mask.cpu().numpy()
        drone_pos = drone_position.cpu().numpy()
    else:
        points = interest_points
        scanned = scanned_mask
        drone_pos = drone_position
    
    # Plot unscanned points in red
    unscanned_points = points[~scanned]
    if len(unscanned_points) > 0:
        ax.scatter(unscanned_points[:, 0], unscanned_points[:, 1], unscanned_points[:, 2], 
                  c='red', s=20, alpha=0.6, label='Unscanned')
    
    # Plot scanned points in green
    scanned_points = points[scanned]
    if len(scanned_points) > 0:
        ax.scatter(scanned_points[:, 0], scanned_points[:, 1], scanned_points[:, 2], 
                  c='green', s=20, alpha=0.8, label='Scanned')
    
    # Plot drone position
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], 
              c='blue', s=100, marker='^', label='Drone')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('Building Coverage Progress')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_coverage_heatmap(interest_points, scanned_mask, building_positions, grid_resolution=50):
    """Create a 2D heatmap of coverage density"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy if needed
    if torch.is_tensor(interest_points):
        points = interest_points.cpu().numpy()
        scanned = scanned_mask.cpu().numpy()
    else:
        points = interest_points
        scanned = scanned_mask
    
    # Create grid
    x_min, x_max = points[:, 0].min() - 5, points[:, 0].max() + 5
    y_min, y_max = points[:, 1].min() - 5, points[:, 1].max() + 5
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate coverage density
    coverage_grid = np.zeros((grid_resolution, grid_resolution))
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # Count scanned points within radius
            center = np.array([X[i, j], Y[i, j]])
            distances = np.linalg.norm(points[:, :2] - center, axis=1)
            nearby_mask = distances < 3.0  # 3m radius
            
            if np.sum(nearby_mask) > 0:
                coverage_grid[i, j] = np.sum(scanned[nearby_mask]) / np.sum(nearby_mask)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coverage_grid, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add building positions
    for pos in building_positions:
        ax.plot(pos[0], pos[1], 'ks', markersize=8, label='Building')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Coverage Density Heatmap')
    plt.colorbar(im, ax=ax, label='Coverage Ratio')
    
    return fig

def calculate_building_inspection_score(coverage_by_type, path_efficiency, collision_count, time_taken):
    """Calculate overall building inspection performance score"""
    # Coverage score (weighted by importance)
    type_weights = torch.tensor([0.2, 0.3, 0.2, 0.3])  # wall, corner, roof, entrance
    coverage_score = torch.sum(coverage_by_type * type_weights, dim=1)
    
    # Efficiency score
    efficiency_score = torch.clamp(path_efficiency / 0.1, 0, 1)  # Normalize to 0-1
    
    # Safety score (penalty for collisions)
    safety_score = torch.exp(-collision_count * 0.5)  # Exponential penalty
    
    # Time score (penalty for excessive time)
    max_time = 1000  # Maximum reasonable time
    time_score = torch.clamp(1.0 - time_taken / max_time, 0, 1)
    
    # Combined score
    total_score = (coverage_score * 0.4 + 
                  efficiency_score * 0.2 + 
                  safety_score * 0.3 + 
                  time_score * 0.1)
    
    return total_score, {
        'coverage': coverage_score,
        'efficiency': efficiency_score, 
        'safety': safety_score,
        'time': time_score
    }
