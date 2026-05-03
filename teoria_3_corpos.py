import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# --- Constantes e Parâmetros ---
G = 1.0  # Constante gravitacional (pode ajustar ou usar valor real se usar unidades SI)
simulation_time = 50.0  # Tempo total da simulação
dt = 0.01  # Passo de tempo para os pontos de avaliação (não o passo interno do solver)
num_frames = int(simulation_time / dt)

# --- Condições Iniciais ---
# Massas dos três corpos
m1 = 2.0
m2 = 1.5
m3 = 1.0
masses = np.array([m1, m2, m3])

# Posições iniciais [x, y, z] para cada corpo
# Exemplo: Um sistema quase plano com pequenas perturbações fora do plano
r1_init = np.array([-1.0, 0.0, 0.1])
r2_init = np.array([1.0, 0.0, -0.1])
r3_init = np.array([0.0, 0.0, 0.0])
initial_positions = np.concatenate([r1_init, r2_init, r3_init])

# Velocidades iniciais [vx, vy, vz] para cada corpo
# Exemplo: Para tentar uma órbita mais estável (mas ainda assim provavelmente caótica)
v1_init = np.array([0.2, 0.3, 0.1])
v2_init = np.array([0.2, -0.3, -0.1])
v3_init = np.array([-0.4, 0.0, 0.0]) # Corpo central mais lento para compensar
initial_velocities = np.concatenate([v1_init, v2_init, v3_init])

# Estado inicial combinado [x1,y1,z1, x2,y2,z2, x3,y3,z3, vx1,vy1,vz1, vx2,vy2,vz2, vx3,vy3,vz3]
initial_state = np.concatenate([initial_positions, initial_velocities])

# --- Função das Equações Diferenciais ---
def three_body_derivs(t, y, masses, G):
    """
    Calcula as derivadas do estado (velocidades e acelerações) para o problema dos três corpos.
    y: array do estado [pos1, pos2, pos3, vel1, vel2, vel3] onde cada pos/vel é [x,y,z]
    masses: array das massas [m1, m2, m3]
    G: constante gravitacional
    """
    num_bodies = len(masses)
    dim = 3 # Dimensão (3D)
    num_vars = num_bodies * dim

    # Extrai posições e velocidades do vetor de estado y
    positions = y[:num_vars].reshape((num_bodies, dim))
    velocities = y[num_vars:].reshape((num_bodies, dim))

    # Inicializa acelerações como zeros
    accelerations = np.zeros_like(positions)

    # Calcula as forças/acelerações entre cada par de corpos
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            # Vetor diferença de posição
            r_ij = positions[j] - positions[i]
            # Distância ao quadrado (adiciona pequena constante para evitar divisão por zero)
            dist_sq = np.sum(r_ij**2)
            dist = np.sqrt(dist_sq)

            if dist > 1e-10: # Evita divisão por zero se colisões ocorrerem
                # Magnitude da força gravitacional (F = G*m1*m2 / r^2)
                force_mag = G * masses[i] * masses[j] / dist_sq
                # Vetor força (direção r_ij / dist)
                force_vec = force_mag * (r_ij / dist)

                # Aceleração = Força / massa (Lei de Newton)
                accelerations[i] += force_vec / masses[i]
                accelerations[j] -= force_vec / masses[j] # Força oposta em j

    # As derivadas das posições são as velocidades
    # As derivadas das velocidades são as acelerações
    derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return derivatives

# --- Solução Numérica ---
print("Resolvendo as equações diferenciais...")
t_span = (0, simulation_time)
t_eval = np.linspace(t_span[0], t_span[1], num_frames)

# Usando solve_ivp para resolver o sistema de EDOs
# 'RK45' é um bom método padrão. 'rtol' e 'atol' controlam a precisão.
sol = solve_ivp(three_body_derivs, t_span, initial_state,
                args=(masses, G),
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8, atol=1e-8)

if not sol.success:
    print(f"Falha ao resolver as EDOs: {sol.message}")
    exit()

print("Solução calculada.")

# Extrai as trajetórias de cada corpo
num_bodies = len(masses)
dim = 3
num_vars = num_bodies * dim
positions_history = sol.y[:num_vars].reshape((num_bodies, dim, num_frames))

# --- Configuração da Visualização 3D ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Cores e tamanhos para os corpos (relativos à massa, por exemplo)
colors = ['red', 'blue', 'green']
sizes = [20 + m**0.5 * 20 for m in masses] # Tamanho visual baseado na massa

# Inicializa os pontos (scatter) para os corpos e as linhas (plot) para as trajetórias
points = [ax.scatter([], [], [], s=sizes[i], c=colors[i], label=f'Corpo {i+1} (m={masses[i]:.1f})') for i in range(num_bodies)]
lines = [ax.plot([], [], [], c=colors[i], lw=1)[0] for i in range(num_bodies)] # lw é a espessura da linha

# --- Função de Animação ---
def animate(frame):
    trail_length = 100 # Quantos pontos anteriores mostrar na trajetória

    for i in range(num_bodies):
        # Atualiza a posição atual do corpo i
        points[i]._offsets3d = (positions_history[i, 0, frame:frame+1],
                                positions_history[i, 1, frame:frame+1],
                                positions_history[i, 2, frame:frame+1])

        # Atualiza a linha da trajetória do corpo i
        start_index = max(0, frame - trail_length)
        lines[i].set_data(positions_history[i, 0, start_index:frame+1],
                          positions_history[i, 1, start_index:frame+1])
        lines[i].set_3d_properties(positions_history[i, 2, start_index:frame+1])

    # Ajusta limites dos eixos dinamicamente (opcional, pode fixar se preferir)
    all_x = positions_history[:, 0, :frame+1].flatten()
    all_y = positions_history[:, 1, :frame+1].flatten()
    all_z = positions_history[:, 2, :frame+1].flatten()

    if len(all_x) > 0 : # Evitar erro no primeiro frame
        min_lim_x, max_lim_x = np.min(all_x), np.max(all_x)
        min_lim_y, max_lim_y = np.min(all_y), np.max(all_y)
        min_lim_z, max_lim_z = np.min(all_z), np.max(all_z)

        # Adiciona uma margem
        margin_x = (max_lim_x - min_lim_x) * 0.1 + 1 # Adiciona 1 para evitar limites 0 se parado
        margin_y = (max_lim_y - min_lim_y) * 0.1 + 1
        margin_z = (max_lim_z - min_lim_z) * 0.1 + 1

        ax.set_xlim(min_lim_x - margin_x, max_lim_x + margin_x)
        ax.set_ylim(min_lim_y - margin_y, max_lim_y + margin_y)
        ax.set_zlim(min_lim_z - margin_z, max_lim_z + margin_z)


    return points + lines # Retorna os artistas que foram modificados

# --- Configuração Final do Gráfico e Execução ---
ax.set_title('Simulação 3D do Problema dos Três Corpos')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.grid(True)

# Garante proporção de eixos aproximadamente igual (importante para visualização 3D)
# Isso pode ser complicado no matplotlib 3D, uma alternativa é definir limites fixos.
# ax.set_aspect('equal', adjustable='box') # Pode causar problemas dependendo do backend

# Cria e executa a animação
# interval: delay entre frames em milissegundos
# blit=False é geralmente necessário para animações 3D no matplotlib
ani = animation.FuncAnimation(fig, animate, frames=num_frames,
                              interval=20, blit=False, repeat=False) # repeat=False para não reiniciar

print("Mostrando animação... (Pode levar um momento para iniciar)")
plt.show()

print("Simulação concluída.")