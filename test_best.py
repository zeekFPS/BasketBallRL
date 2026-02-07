from stable_baselines3 import PPO
from basketball_env import BasketballShooterEnvV2
import time
import numpy as np

# On charge l'environnement
env = BasketballShooterEnvV2(render_mode="human")

# On charge le modele
try:
    model_path = "models_v2/best_shooter_v2"
    model = PPO.load(model_path, env=env)
    print(f"--- Modèle chargé : {model_path} ---")
except FileNotFoundError:
    print("Erreur : Fichier modèle introuvable. Lancez train_best.py d'abord.")
    exit()

obs, _ = env.reset()

print("Démarrage de la démo (50 lancers)...")
print("Format: Start X | Action [Angle, Power] | Obs [X, Dist] | Terminated | Reward")
print("-" * 80)

# On boucle 50 fois
for i in range(50):
    # On sauvegarde la position X de départ pour l'affichage
    start_x = obs[0]
    
    # Prédiction de l'action
    action, _ = model.predict(obs, deterministic=True)
    
    # Exécution du tir
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    # Affichage avec le numéro du tir (i+1)
    print(f"Tir {i+1}/50 : {start_x:.2f} | {action} | {new_obs} | {terminated} | {reward:.2f}")
    
    if terminated:
        time.sleep(0.2) # Petite pause visuelle
        obs, _ = env.reset()
    else:
        obs = new_obs

print("-" * 80)
print("Série de 50 lancers terminée.")
env.close()