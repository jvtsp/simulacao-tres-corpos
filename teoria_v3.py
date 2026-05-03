# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
from matplotlib import cm # Colormaps

# --- Constantes e Parâmetros (mesmos de antes) ---
G = 1.0
simulation_time = 25.0 # Tempo reduzido para testar a performance
dt = 0.02 # Passo de tempo maior para reduzir frames
num_frames = int(simulation_time / dt)

# --- Condições Iniciais (mesmas de antes) ---
m1 = 5.0
m2 = 3.5
m3 = 1.3
masses = np.array([m1, m2, m3])
r1_init = np.array([0.0, 0.0, 0.0])
r2_init = np.array([5.0, 0.0, 0.5])
r3_init = np.array([-3.0, 3.0, -0.3])
initial_positions = np.concatenate([r1_init, r2_init, r3_init])
v1_init = np.array([0.01, 0.01, 0])
v2_init = np.array([0.0, 0.8, 0.1])
v3_init = np.array([-0.6, -0.6, -0.1])
initial_velocities = np.concatenate([v1_init, v2_init, v3_init])
initial_state = np.concatenate([initial_positions, initial_velocities])

# --- Função das Equações Diferenciais (sem alterações) ---
def three_body_derivs(t, y, masses, G):
    # ... (código idêntico ao anterior) ...
    num_bodies = len(masses)
    dim = 3
    num_vars = num_bodies * dim
    positions = y[:num_vars].reshape((num_bodies, dim))
    velocities = y[num_vars:].reshape((num_bodies, dim))
    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            r_ij = positions[j] - positions[i]
            dist_sq = np.sum(r_ij**2)
            dist = np.sqrt(dist_sq)
            # Pequena constante para suavizar o potencial e evitar divisão por zero
            # Ajuste 'softening_factor' conforme necessário
            softening_factor = 0.1
            dist_softened = np.sqrt(dist_sq + softening_factor**2)

            if dist_softened > 1e-10: # Segurança
                force_mag = G * masses[i] * masses[j] / dist_softened**2 # Usa dist_softened aqui também? Ou só no potencial? Testar. Usar dist normal para força.
                force_mag_calc = G * masses[i] * masses[j] / dist_sq # Força usa dist normal

                force_vec = force_mag_calc * (r_ij / dist)
                accelerations[i] += force_vec / masses[i]
                accelerations[j] -= force_vec / masses[j]
    derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return derivatives


# --- Solução Numérica (mesma de antes) ---
print("Resolvendo as equações diferenciais...")
start_time = time.time()
t_span = (0, simulation_time)
t_eval = np.linspace(t_span[0], t_span[1], num_frames)

sol = solve_ivp(three_body_derivs, t_span, initial_state,
                args=(masses, G),
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8, # Reduzir um pouco a precisão para talvez acelerar
                atol=1e-8)

end_time = time.time()
print(f"Solução calculada em {end_time - start_time:.2f} segundos.")

if not sol.success:
    print(f"Falha ao resolver as EDOs: {sol.message}")
    exit()

num_bodies = len(masses)
dim = 3
num_vars = num_bodies * dim
positions_history = sol.y[:num_vars].reshape((num_bodies, dim, num_frames))

# --- Cálculo dos Limites (mesmo de antes, mas talvez ajustar Z para o potencial) ---
print("Calculando limites...")
all_pos_flat = positions_history.reshape(num_bodies * dim, num_frames)
min_vals_space = np.min(all_pos_flat, axis=1)
max_vals_space = np.max(all_pos_flat, axis=1)
center_space = (max_vals_space + min_vals_space) / 2.0
max_range_space = np.max(max_vals_space - min_vals_space) * 0.6
plot_limits_space = np.array([(c - max_range_space, c + max_range_space) for c in center_space])
plot_limits_space[:, 0] -= 1
plot_limits_space[:, 1] += 1

# --- Função para Calcular Potencial Gravitacional ---
def calculate_potential(grid_x, grid_y, current_positions, masses, G):
    """Calcula o potencial gravitacional Newtoniano em uma grade 2D (z=0)."""
    potential = np.zeros(grid_x.shape)
    softening_factor = 0.5 # Fator maior para evitar picos muito profundos na visualização

    for i in range(len(masses)):
        body_pos = current_positions[i]
        dx = grid_x - body_pos[0]
        dy = grid_y - body_pos[1]
        dz = 0 - body_pos[2] # Distância Z da grade (z=0) ao corpo
        dist_sq = dx**2 + dy**2 + dz**2
        dist_softened = np.sqrt(dist_sq + softening_factor**2) # Adiciona suavização
        potential -= (G * masses[i]) / dist_softened # Potencial é negativo
    # Limita o quão fundo o potencial pode ir para visualização
    potential = np.clip(potential, -50, 0) # Ajuste o limite inferior conforme necessário
    return potential

# --- Setup da Grade para Superfície de Potencial ---
grid_resolution = 40 # Menor resolução para performance (40x40)
x_grid = np.linspace(plot_limits_space[0, 0], plot_limits_space[0, 1], grid_resolution)
y_grid = np.linspace(plot_limits_space[1, 0], plot_limits_space[1, 1], grid_resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# --- Configuração da Visualização 3D ---
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# --- Estilo Espacial (como antes) ---
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white') # Nomear Z como 'Espaço / Potencial' ?
ax.title.set_color('white')
ax.tick_params(axis='x', colors='grey')
ax.tick_params(axis='y', colors='grey')
ax.tick_params(axis='z', colors='grey')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z / Potencial') # Indicar duplo significado

# --- Estrelas de Fundo (como antes, usar limites espaciais) ---
num_stars = 400
star_x = np.random.uniform(plot_limits_space[0, 0], plot_limits_space[0, 1], num_stars)
star_y = np.random.uniform(plot_limits_space[1, 0], plot_limits_space[1, 1], num_stars)
# Colocar estrelas em várias profundidades Z
star_z = np.random.uniform(plot_limits_space[2, 0], plot_limits_space[2, 1], num_stars)
star_sizes = np.random.uniform(0.5, 2.0, num_stars)
star_alphas = np.random.uniform(0.3, 0.7, num_stars)
# Plotar estrelas ANTES da superfície, para que fiquem "atrás"
ax.scatter(star_x, star_y, star_z, s=star_sizes, c='white', alpha=star_alphas, marker='.', depthshade=False)

# --- Objetos Principais (como antes) ---
colors = ['yellow', 'cyan', 'magenta']
sizes = [60 + m**0.5 * 40 for m in masses] # Um pouco maiores ainda?
# Trails com alpha
trail_alpha = 0.4

# Variável global para a superfície (para tentar atualizar em vez de recriar, embora possa não funcionar bem)
# surface_plot = None # Inicialmente nulo

# --- Função de Animação ---
frame_count_start_time = time.time()
frame_counter = 0

def animate(frame):
    global frame_counter #, surface_plot
    frame_start_time = time.time()

    # Limpa o eixo COMPLETAMENTE - mais simples, porém mais lento
    ax.clear()

    # --- Re-aplica todas as configurações de estilo e fundo ---
    ax.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    ax.tick_params(axis='z', colors='grey')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z / Potencial')
    ax.set_title(f'Simulação 3 Corpos c/ Potencial (T: {sol.t[frame]:.2f})', color='white')

    # --- Recalcula e redesenha a Superfície de Potencial ---
    current_pos = positions_history[:, :, frame]
    Z_potential = calculate_potential(X_grid, Y_grid, current_pos, masses, G)
    # Usa um colormap e alpha para transparência
    # cmap='viridis', 'plasma', 'coolwarm', 'Blues_r' (invertido)
    surface_plot = ax.plot_surface(X_grid, Y_grid, Z_potential, cmap=cm.Blues_r, alpha=0.6, rstride=1, cstride=1, linewidth=0, antialiased=True)

    # --- Redesenha Estrelas de Fundo ---
    # (Necessário após ax.clear()) - Considerar plotar uma vez fora do loop se não usar clear
    ax.scatter(star_x, star_y, star_z, s=star_sizes, c='white', alpha=star_alphas, marker='.', depthshade=False)

    # --- Redesenha Corpos e Rastros ---
    trail_length = 100
    points_artists = []
    lines_artists = []
    for i in range(num_bodies):
        # Corpo atual
        point = ax.scatter(positions_history[i, 0, frame:frame+1],
                           positions_history[i, 1, frame:frame+1],
                           positions_history[i, 2, frame:frame+1],
                           s=sizes[i], c=colors[i], label=f'Corpo {i+1}' if frame == 0 else "", # Legenda só no primeiro frame
                           depthshade=False)
        points_artists.append(point)

        # Rastro
        start_index = max(0, frame - trail_length)
        line, = ax.plot(positions_history[i, 0, start_index:frame+1],
                        positions_history[i, 1, start_index:frame+1],
                        positions_history[i, 2, start_index:frame+1],
                        c=colors[i], lw=1, alpha=trail_alpha)
        lines_artists.append(line)

    # --- Define Limites (precisa fazer a cada frame por causa do ax.clear()) ---
    # Limites X e Y baseados no espaço
    ax.set_xlim(plot_limits_space[0])
    ax.set_ylim(plot_limits_space[1])
    # Limite Z precisa acomodar tanto posições Z quanto profundidade do potencial Z
    min_z_space = plot_limits_space[2, 0]
    max_z_space = plot_limits_space[2, 1]
    min_z_potential = np.min(Z_potential)
    max_z_potential = np.max(Z_potential) # Deveria ser perto de 0
    ax.set_zlim(min(min_z_space, min_z_potential - 2), # Adiciona margem abaixo do potencial
                max(max_z_space, max_z_potential + 1)) # Adiciona margem acima


    # Recria legenda no primeiro frame
    if frame == 0:
       legend = ax.legend()
       plt.setp(legend.get_texts(), color='white')
       legend.get_frame().set_alpha(0.3)


    # Calcula FPS médio
    frame_counter += 1
    current_total_time = time.time() - frame_count_start_time
    fps = frame_counter / current_total_time if current_total_time > 0 else 0
    # print(f"Frame {frame}/{num_frames}, Tempo Frame: {time.time() - frame_start_time:.3f}s, FPS Médio: {fps:.2f}", end='\r') # Comentar para não poluir saida

    # Retorna lista vazia porque clear/redraw não funciona bem com blit=True
    return [] # Ou return [surface_plot] + points_artists + lines_artists se tentar não usar clear


# --- Configuração Final e Execução ---
print("\nIniciando animação... (Pode ser LENTO devido à superfície dinâmica)")
print("Use a janela para rotacionar/zoom.")

# interval: Aumentar se estiver muito lento (ex: 50 ou 100ms)
ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                              interval=50, # Aumentado para dar mais tempo de render
                              blit=False, # TEM que ser False com ax.clear()
                              repeat=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nAnimação interrompida.")

print("Visualização concluída.")

print("\n\n--- Sugestões para Melhor Performance/Visualização ---")
print("1. Reduza 'grid_resolution' ou 'num_frames' para acelerar.")
print("2. Para performance muito superior e visuais mais ricos:")
print("   - Python: Use Mayavi ou VisPy (requer instalação e aprendizado).")
print("   - Web: Use Python (Flask/Django) para física + JavaScript (Three.js) para visualização 3D no navegador.")
print("   - Motores de Jogo: Use Ursina Engine (Python) ou Godot.")