import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time # Para medir o tempo de cálculo

# --- Constantes e Parâmetros ---
G = 1.0
simulation_time = 50.0 # Tempo mais curto para teste inicial?
dt = 0.01
num_frames = int(simulation_time / dt)

# --- Condições Iniciais ---
m1 = 5.0 # Massa maior para o "sol"?
m2 = 0.5 # Planeta 1
m3 = 0.3 # Planeta 2
masses = np.array([m1, m2, m3])

# Posições iniciais [x, y, z]
r1_init = np.array([0.0, 0.0, 0.0])  # Corpo central parado (inicialmente)
r2_init = np.array([5.0, 0.0, 0.5])  # Afastado no plano X, um pouco fora Z
r3_init = np.array([-3.0, 3.0, -0.3]) # Em outra direção
initial_positions = np.concatenate([r1_init, r2_init, r3_init])

# Velocidades iniciais [vx, vy, vz]
# Cuidado: velocidades iniciais são cruciais para órbitas interessantes vs ejeção
v1_init = np.array([0.01, 0.01, 0]) # Pequeno drift no corpo central
v2_init = np.array([0.0, 0.8, 0.1])  # Velocidade inicial para orbitar (ajustar!)
v3_init = np.array([-0.6, -0.6, -0.1]) # Velocidade inicial para orbitar (ajustar!)
initial_velocities = np.concatenate([v1_init, v2_init, v3_init])

initial_state = np.concatenate([initial_positions, initial_velocities])

# --- Função das Equações Diferenciais (sem alterações) ---
def three_body_derivs(t, y, masses, G):
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
            if dist > 1e-10: # Evita divisão por zero
                force_mag = G * masses[i] * masses[j] / dist_sq
                force_vec = force_mag * (r_ij / dist)
                accelerations[i] += force_vec / masses[i]
                accelerations[j] -= force_vec / masses[j]
    derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return derivatives

# --- Solução Numérica ---
print("Resolvendo as equações diferenciais... Isso pode levar um tempo.")
start_time = time.time()
t_span = (0, simulation_time)
t_eval = np.linspace(t_span[0], t_span[1], num_frames)

sol = solve_ivp(three_body_derivs, t_span, initial_state,
                args=(masses, G),
                t_eval=t_eval,
                method='RK45', # 'DOP853' pode ser mais preciso, mas lento
                rtol=1e-9,  # Tolerância relativa mais rigorosa
                atol=1e-9)  # Tolerância absoluta mais rigorosa

end_time = time.time()
print(f"Solução calculada em {end_time - start_time:.2f} segundos.")

if not sol.success:
    print(f"Falha ao resolver as EDOs: {sol.message}")
    exit()

# Extrai as trajetórias
num_bodies = len(masses)
dim = 3
num_vars = num_bodies * dim
positions_history = sol.y[:num_vars].reshape((num_bodies, dim, num_frames))

# --- Cálculo dos Limites dos Eixos ---
print("Calculando limites para a visualização...")
all_pos = positions_history.reshape(-1, num_frames) # Achatando todas as coordenadas
min_vals = np.min(all_pos, axis=1)
max_vals = np.max(all_pos, axis=1)

# Adiciona margem
center = (max_vals + min_vals) / 2.0
max_range = np.max(max_vals - min_vals) * 0.6 # Pega 60% da maior extensão em qualquer eixo
plot_limits = np.array([(c - max_range, c + max_range) for c in center])
# Garante que os limites não sejam idênticos se algo não se moveu muito
plot_limits[:, 0] -= 1 # Adiciona pequena margem mínima
plot_limits[:, 1] += 1

print(f"Limites X: {plot_limits[0]}")
print(f"Limites Y: {plot_limits[1]}")
print(f"Limites Z: {plot_limits[2]}")


# --- Configuração da Visualização 3D - Estilo Espacial ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Fundo preto
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Eixos e Ticks brancos/cinzas, sem painéis de fundo
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
# ax.xaxis.pane.set_edgecolor('w') # Linhas de borda do cubo (opcional)
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')

# Remove a grade
ax.grid(False)

# --- Estrelas de Fundo ---
num_stars = 500
# Gera estrelas dentro dos limites calculados
star_x = np.random.uniform(plot_limits[0, 0], plot_limits[0, 1], num_stars)
star_y = np.random.uniform(plot_limits[1, 0], plot_limits[1, 1], num_stars)
star_z = np.random.uniform(plot_limits[2, 0], plot_limits[2, 1], num_stars)
star_sizes = np.random.uniform(0.5, 2.5, num_stars) # Tamanhos variados
star_alphas = np.random.uniform(0.3, 0.8, num_stars) # Brilhos variados

# Plota as estrelas uma vez, no fundo
ax.scatter(star_x, star_y, star_z, s=star_sizes, c='white', alpha=star_alphas, marker='.')

# --- Objetos Principais (Planetas/Estrelas) ---
# Cores mais vibrantes
colors = ['yellow', 'cyan', 'magenta']
# Tamanhos baseados na massa, talvez um pouco maiores
sizes = [50 + m**0.5 * 40 for m in masses]

# Inicializa os pontos (scatter) e linhas (plot)
points = [ax.scatter([], [], [], s=sizes[i], c=colors[i], label=f'Corpo {i+1} (m={masses[i]:.1f})', depthshade=False) # depthshade=False pode ajudar a manter a cor brilhante
          for i in range(num_bodies)]
# Linhas com transparência e cor correspondente
lines = [ax.plot([], [], [], c=colors[i], lw=1, alpha=0.5)[0] for i in range(num_bodies)]

# --- Função de Animação (Atualizada para usar limites fixos) ---
def animate(frame):
    trail_length = 150 # Rastro um pouco mais longo?

    for i in range(num_bodies):
        # Atualiza posição do corpo
        points[i]._offsets3d = (positions_history[i, 0, frame:frame+1],
                                positions_history[i, 1, frame:frame+1],
                                positions_history[i, 2, frame:frame+1])

        # Atualiza rastro
        start_index = max(0, frame - trail_length)
        lines[i].set_data(positions_history[i, 0, start_index:frame+1],
                          positions_history[i, 1, start_index:frame+1])
        lines[i].set_3d_properties(positions_history[i, 2, start_index:frame+1])

    # Não precisa mais ajustar limites aqui, pois são fixos

    # Atualiza título com o tempo (opcional)
    ax.set_title(f'Simulação 3D - Três Corpos (Tempo: {sol.t[frame]:.2f})', color='white')

    return points + lines

# --- Configuração Final do Gráfico e Execução ---
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define os limites calculados
ax.set_xlim(plot_limits[0])
ax.set_ylim(plot_limits[1])
ax.set_zlim(plot_limits[2])

# Legenda com texto branco
legend = ax.legend()
plt.setp(legend.get_texts(), color='white')
legend.get_frame().set_alpha(0.3) # Fundo da legenda semi-transparente

# Cria e executa a animação
ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                              interval=20, # Delay entre frames (ms). Aumente se estiver lento.
                              blit=False, # Blit=False é mais seguro para 3D
                              repeat=False)

print("Mostrando animação... Pressione Ctrl+C no terminal para parar se necessário.")
try:
    plt.show()
except KeyboardInterrupt:
    print("\nAnimação interrompida pelo usuário.")

print("Simulação visualizada.")