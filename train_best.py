import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from basketball_env import BasketballShooterEnvV2

# Dossiers
log_dir = "logs_v2/"
models_dir = "models_v2/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Création de l'environnement
# Pas de rendu pendant l'entraînement pour aller 100x plus vite
env = BasketballShooterEnvV2(render_mode=None)
env = Monitor(env, log_dir)

# Configuration du modèle "Best Possible"
# Learning rate réduit (0.0003 -> 0.0001) pour une meilleure précision finale
# Batch size augmenté pour des gradients plus stables
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,   # Plus lent mais plus précis
    n_steps=2048,           
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    tensorboard_log=log_dir,
    device="auto"
)

# Entraînement long pour la perfection (100k - 200k steps sont recommandés)
TIMESTEPS = 150000 
print(f"Lancement de l'entraînement High-Perf pour {TIMESTEPS} steps...")

model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

# Sauvegarde
save_path = f"{models_dir}/best_shooter_v2"
model.save(save_path)
print(f"Modèle sauvegardé sous : {save_path}")