import os
import hydra
import datetime
import wandb
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import BuildingCoveragePPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Force evaluation mode settings
    cfg.headless = False  # Enable visualization for evaluation
    cfg.env.num_envs = 1  # Single environment for detailed evaluation
    
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
    
    print(f"[BuildingCoverage]: Evaluation using device: {cfg.device}")
    
    # Set random seeds for reproducible evaluation
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create simulation app with visualization
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Initialize Wandb for evaluation logging
    run = wandb.init(
        project=cfg.wandb.project + "_eval",
        name=f"BuildingCoverage_Eval/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity,
        config=cfg,
        mode=cfg.wandb.mode,
    )

    try:
        # Environment initialization
        from env import BuildingCoverageEnv
        env = BuildingCoverageEnv(cfg)
        
        # Controller setup
        controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
        vel_transform = VelController(controller, yaw_control=True)
        transformed_env = TransformedEnv(env, vel_transform).eval()  # Set to eval mode
        transformed_env.set_seed(cfg.seed)   

        print("\n=== Environment Observation Space ===")
        print(transformed_env.observation_spec) 
        
        # Policy initialization
        print("\n=== Initializing BuildingCoveragePPO Policy ===")
        policy = BuildingCoveragePPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
        
        # Load trained model
        checkpoint_path = input("Enter path to trained model checkpoint (or press Enter for default): ").strip()
        if not checkpoint_path:
            # Default checkpoint path
            checkpoint_path = "./checkpoint_best.pt"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Model loaded successfully!")
            if 'best_coverage' in checkpoint:
                print(f"Best training coverage: {checkpoint['best_coverage']:.4f}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Proceeding with randomly initialized policy...")
        
        print("=== Policy initialized successfully ===\n")
        
        # Evaluation metrics storage
        evaluation_results = {
            'coverage_over_time': [],
            'sides_covered_over_time': [],
            'scanning_efficiency_over_time': [],
            'collision_events': [],
            'trajectory_data': [],
            'interest_point_data': []
        }
        
        # Run multiple evaluation episodes
        num_eval_episodes = 5
        print(f"Running {num_eval_episodes} evaluation episodes...")
        
        for episode in range(num_eval_episodes):
            print(f"\n--- Episode {episode + 1}/{num_eval_episodes} ---")
            
            # Reset environment
            env.reset()
            
            # Run single episode evaluation
            episode_data = run_single_episode_evaluation(transformed_env, policy, cfg, episode)
            
            # Store results
            for key in evaluation_results.keys():
                if key in episode_data:
                    evaluation_results[key].append(episode_data[key])
            
            # Print episode summary
            print_episode_summary(episode_data, episode)
        
        # Analyze and visualize results
        print("\n=== Analysis and Visualization ===")
        analysis_results = analyze_evaluation_results(evaluation_results)
        
        # Create visualizations
        create_evaluation_plots(evaluation_results, analysis_results)
        
        # Log final results to wandb
        log_evaluation_results(run, analysis_results)
        
        print("\n=== Final Evaluation Summary ===")
        print_final_summary(analysis_results)

    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        wandb.finish()
        sim_app.close()
        print("Evaluation script finished.")


def run_single_episode_evaluation(env, policy, cfg, episode_num):
    """Run a single episode evaluation with detailed tracking"""
    print(f"Running evaluation episode {episode_num + 1}...")
    
    # Episode data storage
    episode_data = {
        'coverage_over_time': [],
        'sides_covered_over_time': [],
        'scanning_efficiency_over_time': [],
        'collision_events': [],
        'trajectory_data': [],
        'interest_point_data': [],
        'reward_components': []
    }
    
    # Reset environment
    tensordict = env.reset()
    step = 0
    total_reward = 0
    done = False
    
    # Enable visualization for this episode
    env.enable_render(True)
    
    while not done and step < env.max_episode_length:
        # Get action from policy
        with torch.no_grad():
            action_tensordict = policy(tensordict)
        
        # Step environment
        tensordict = env.step(action_tensordict)
        
        # Extract data for analysis
        reward = tensordict["next", "agents", "reward"].item()
        total_reward += reward
        
        # Get current state information
        stats = tensordict["next", "stats"]
        
        # Store trajectory point
        drone_state = tensordict["next", "info", "drone_state"][0, 0]  # First env, first drone
        episode_data['trajectory_data'].append({
            'step': step,
            'position': drone_state[:3].cpu().numpy(),
            'velocity': drone_state[7:10].cpu().numpy(),
            'reward': reward
        })
        
        # Store coverage metrics
        episode_data['coverage_over_time'].append(stats["total_coverage"].item())
        episode_data['sides_covered_over_time'].append(stats["sides_covered"].item())
        episode_data['scanning_efficiency_over_time'].append(stats["scanning_efficiency"].item())
        
        # Check for collisions
        if stats["collision"].item() > 0:
            episode_data['collision_events'].append(step)
        
        # Store interest point information
        if hasattr(env, 'interest_point_scanned'):
            scanned_count = torch.sum(env.interest_point_scanned[0]).item()
            total_points = torch.sum(env.interest_point_visible[0]).item()
            episode_data['interest_point_data'].append({
                'step': step,
                'scanned': scanned_count,
                'total': total_points,
                'coverage_ratio': scanned_count / max(total_points, 1)
            })
        
        # Print progress every 100 steps
        if step % 100 == 0:
            coverage = stats["total_coverage"].item()
            sides = stats["sides_covered"].item()
            efficiency = stats["scanning_efficiency"].item()
            print(f"  Step {step:3d}: Coverage={coverage:.3f}, Sides={sides:.1f}/5, Efficiency={efficiency:.3f}")
        
        # Check termination
        done = tensordict["next", "done"].any().item()
        step += 1
        
        # Update tensordict for next iteration
        tensordict = tensordict["next"]
    
    # Final episode statistics
    episode_data['total_reward'] = total_reward
    episode_data['episode_length'] = step
    episode_data['final_coverage'] = episode_data['coverage_over_time'][-1] if episode_data['coverage_over_time'] else 0
    episode_data['final_sides_covered'] = episode_data['sides_covered_over_time'][-1] if episode_data['sides_covered_over_time'] else 0
    episode_data['num_collisions'] = len(episode_data['collision_events'])
    
    return episode_data


def print_episode_summary(episode_data, episode_num):
    """Print summary of a single episode"""
    print(f"Episode {episode_num + 1} Summary:")
    print(f"  Total Reward: {episode_data['total_reward']:.2f}")
    print(f"  Episode Length: {episode_data['episode_length']} steps")
    print(f"  Final Coverage: {episode_data['final_coverage']:.3f}")
    print(f"  Final Sides Covered: {episode_data['final_sides_covered']:.1f}/5")
    print(f"  Collisions: {episode_data['num_collisions']}")


def analyze_evaluation_results(results):
    """Analyze evaluation results across all episodes"""
    analysis = {}
    
    # Coverage analysis
    final_coverages = [ep[-1] if ep else 0 for ep in results['coverage_over_time']]
    analysis['mean_coverage'] = np.mean(final_coverages)
    analysis['std_coverage'] = np.std(final_coverages)
    analysis['max_coverage'] = np.max(final_coverages)
    analysis['min_coverage'] = np.min(final_coverages)
    
    # Sides covered analysis
    final_sides = [ep[-1] if ep else 0 for ep in results['sides_covered_over_time']]
    analysis['mean_sides_covered'] = np.mean(final_sides)
    analysis['std_sides_covered'] = np.std(final_sides)
    
    # Efficiency analysis
    final_efficiencies = [ep[-1] if ep else 0 for ep in results['scanning_efficiency_over_time']]
    analysis['mean_efficiency'] = np.mean(final_efficiencies)
    analysis['std_efficiency'] = np.std(final_efficiencies)
    
    # Collision analysis
    total_collisions = sum(len(ep) for ep in results['collision_events'])
    analysis['total_collisions'] = total_collisions
    analysis['collision_rate'] = total_collisions / len(results['collision_events'])
    
    return analysis


def create_evaluation_plots(results, analysis):
    """Create visualization plots for evaluation results"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Building Coverage Evaluation Results', fontsize=16)
    
    # Plot 1: Coverage over time
    ax1 = axes[0, 0]
    for i, coverage_data in enumerate(results['coverage_over_time']):
        ax1.plot(coverage_data, label=f'Episode {i+1}', alpha=0.7)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Coverage Ratio')
    ax1.set_title('Coverage Progress Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Sides covered over time
    ax2 = axes[0, 1]
    for i, sides_data in enumerate(results['sides_covered_over_time']):
        ax2.plot(sides_data, label=f'Episode {i+1}', alpha=0.7)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Sides Covered')
    ax2.set_title('Building Sides Covered Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Final performance distribution
    ax3 = axes[1, 0]
    final_coverages = [ep[-1] if ep else 0 for ep in results['coverage_over_time']]
    final_sides = [ep[-1] if ep else 0 for ep in results['sides_covered_over_time']]
    
    x = np.arange(len(final_coverages))
    width = 0.35
    ax3.bar(x - width/2, final_coverages, width, label='Coverage', alpha=0.8)
    ax3.bar(x + width/2, [s/5.0 for s in final_sides], width, label='Sides (normalized)', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Performance')
    ax3.set_title('Final Performance by Episode')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Trajectory visualization (for first episode)
    ax4 = axes[1, 1]
    if results['trajectory_data']:
        traj_data = results['trajectory_data'][0]  # First episode
        positions = np.array([point['position'] for point in traj_data])
        ax4.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, label='Trajectory')
        ax4.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        ax4.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
    ax4.set_xlabel('X Position (m)')
    ax4.set_ylabel('Y Position (m)')
    ax4.set_title('Drone Trajectory (Episode 1)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def log_evaluation_results(run, analysis):
    """Log evaluation results to wandb"""
    wandb.log({
        'eval/mean_coverage': analysis['mean_coverage'],
        'eval/std_coverage': analysis['std_coverage'],
        'eval/max_coverage': analysis['max_coverage'],
        'eval/min_coverage': analysis['min_coverage'],
        'eval/mean_sides_covered': analysis['mean_sides_covered'],
        'eval/std_sides_covered': analysis['std_sides_covered'],
        'eval/mean_efficiency': analysis['mean_efficiency'],
        'eval/std_efficiency': analysis['std_efficiency'],
        'eval/total_collisions': analysis['total_collisions'],
        'eval/collision_rate': analysis['collision_rate'],
    })


def print_final_summary(analysis):
    """Print final evaluation summary"""
    print("="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Coverage Performance:")
    print(f"  Mean: {analysis['mean_coverage']:.4f} ± {analysis['std_coverage']:.4f}")
    print(f"  Range: [{analysis['min_coverage']:.4f}, {analysis['max_coverage']:.4f}]")
    print(f"\nSides Covered:")
    print(f"  Mean: {analysis['mean_sides_covered']:.2f} ± {analysis['std_sides_covered']:.2f} / 5")
    print(f"\nScanning Efficiency:")
    print(f"  Mean: {analysis['mean_efficiency']:.4f} ± {analysis['std_efficiency']:.4f}")
    print(f"\nSafety Performance:")
    print(f"  Total Collisions: {analysis['total_collisions']}")
    print(f"  Collision Rate: {analysis['collision_rate']:.2f} per episode")
    print("="*60)


if __name__ == "__main__":
    main()
