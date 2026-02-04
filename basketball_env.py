import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
import warnings

# Suppression du warning pygame
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# Constantes
WIDTH, HEIGHT = 1000, 700
FPS = 60
BALL_RADIUS = 8
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20 # Rayon du cercle de détection

# Couleurs
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
GREEN = (34, 139, 34)

class BasketballShooterEnvV2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.gravity = 0.3
        
        # Action: [angle (-1 à 1), puissance (-1 à 1)]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: [Position X balle, Distance euclidienne au panier]
        self.observation_space = spaces.Box(low=0, high=1500, shape=(2,), dtype=np.float32)

        self.score = 0
        self.reset_shot()
        
    def reset_shot(self):
        self.start_x = random.randint(100, 600)
        self.start_y = HEIGHT - 100

        self.ball_x = self.start_x
        self.ball_y = self.start_y
        self.trajectory = [(self.ball_x, self.ball_y)]
        
        self.shot_complete = False
        self.scored = False
        self.frame_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_shot()
        return self._get_observation(), {}
        
    def _get_observation(self):
        distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
        return np.array([float(self.ball_x), distance], dtype=np.float32)
    
    def step(self, action):
        # Paramètres physiques
        angle = np.interp(action[0], [-1, 1], [20, 80])
        power = np.interp(action[1], [-1, 1], [10, 28])

        angle_rad = math.radians(angle)
        self.vx = power * math.cos(angle_rad)
        self.vy = -power * math.sin(angle_rad)

        min_distance = float('inf')
        
        # Simulation
        while not self.shot_complete:
            self.vy += self.gravity
            self.ball_x += self.vx
            self.ball_y += self.vy
            
            # Distance actuelle au centre du panier
            curr_dist = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
            if curr_dist < min_distance:
                min_distance = curr_dist

            self.trajectory.append((int(self.ball_x), int(self.ball_y)))
            self.frame_count += 1
            
            # --- CORRECTION DE LA LOGIQUE DE SCORE ---
            # Si la balle est proche du centre ET qu'elle descend (vy > 0)
            if not self.scored and self.vy > 0:
                # On augmente légèrement la tolérance du rayon pour être sûr de capter le passage
                if curr_dist < (HOOP_RADIUS + BALL_RADIUS): 
                    self.scored = True
                    self.score += 1
                    self.shot_complete = True # On arrête dès que ça rentre

            # Conditions d'arrêt (Sol, Mur droit, ou temps écoulé)
            if self.ball_y > HEIGHT or self.ball_x > WIDTH or self.frame_count > 400:
                self.shot_complete = True
            
            # Affichage
            if self.render_mode == "human":
                self.render()

        # --- REWARD ---
        reward = 0
        if self.scored:
            reward = 100.0
        else:
            # Reward shaping (guidage par la distance minimale)
            reward = -0.1 * min_distance
            # Pénalité hors cadre
            if self.ball_x > WIDTH or self.ball_y > HEIGHT:
                reward -= 5.0

        terminated = True
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Basketball Shooter AI - Optimized")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.screen.fill(GREEN)
        pygame.draw.rect(self.screen, BROWN, (0, HEIGHT - 50, WIDTH, 50))
        
        # Panier
        pygame.draw.circle(self.screen, RED, (HOOP_X, HOOP_Y), HOOP_RADIUS, 3)
        # Planche / Support
        pygame.draw.line(self.screen, BROWN, (HOOP_X - 30, HOOP_Y - 50), (HOOP_X - 20, HOOP_Y), 3)
        pygame.draw.line(self.screen, BROWN, (HOOP_X + 30, HOOP_Y - 50), (HOOP_X + 20, HOOP_Y), 3)
        
        # Trajectoire
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, WHITE, False, self.trajectory, 1)
        
        # Balle
        pygame.draw.circle(self.screen, ORANGE, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)
        
        status = "SCORED!" if self.scored else "MISSED"
        color = (0, 255, 0) if self.scored else WHITE
        text = self.font.render(f"{status} | Score: {self.score}", True, color)
        self.screen.blit(text, (20, 20))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.event.pump()

    def close(self):
        if self.screen is not None:
            pygame.quit()