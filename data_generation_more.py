import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ---------------- Global thermodynamic conditions ----------------
T_0 = 7500           # K  - Initial temperature
rho_0 = 0.0013       # kg/m^3 - Initial density
E_const = 0          # J/kg - Energy offset

mech = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data/airNASA9ions.yaml'
fixed_vars = 'UV'    # Constant U,V evolution

# ---------------- Random initial-condition setup ----------------
# Species whose initial mole fractions we randomize (must sum to 1)
init_species = ['CO2', 'N2', 'Ar', 'O2']
n_ICs = 10           # number of random initial conditions
n_points = int(20000)

# Reproducibility
rng = np.random.default_rng(42)

# Dirichlet(1,...,1) gives uniform sampling on the simplex (sum = 1)
random_ICs = rng.dirichlet(np.ones(len(init_species)), size=n_ICs)

# Output directory
out_dir = os.path.join(os.getcwd(), 'training_data_random_ICs')
os.makedirs(out_dir, exist_ok=True)

# Species tracked in the output (full set)
species_names = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N']
species_colors = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink']
ic_colors = ['#d62728', '#2ca02c', '#7f7f7f', '#1f77b4']  # CO2, N2, Ar, O2

print(f"Mechanism: {mech}")
print(f"Generating {n_ICs} random initial conditions over species {init_species}\n")

all_data = []

for ic_idx, ic in enumerate(random_ICs):
    print("=" * 60)
    print(f"IC {ic_idx + 1}/{n_ICs}")

    # Build Cantera composition string
    q = ','.join([f"{sp}:{frac:.8f}" for sp, frac in zip(init_species, ic)])
    print(f"  composition: {q}  (sum = {ic.sum():.6f})")

    # ----- Reactor setup -----
    gas_react = ct.Solution(mech)
    gas_react.X = q
    gas_react.TD = T_0, rho_0
    v_0 = gas_react.v
    P_0 = gas_react.P
    e_0 = gas_react.u + E_const
    h_0 = gas_react.h + E_const

    if fixed_vars == 'HP':
        reactor = ct.ConstPressureReactor(gas_react)
    else:
        reactor = ct.IdealGasReactor(gas_react)
        reactor.volume = v_0
    reactor_net = ct.ReactorNet([reactor])

    # ----- Equilibrium reference -----
    gas_eq = ct.Solution(mech)
    gas_eq.X = q
    if fixed_vars == 'HP':
        gas_eq.HP = h_0 - E_const, P_0
        gas_eq.equilibrate('HP')
    else:
        gas_eq.UV = e_0, v_0
        gas_eq.equilibrate('UV')
    T_eq = gas_eq.T

    # ----- Time grid -----
    t_end = 1e-2 * np.exp(2000 / T_0) * (0.1 / rho_0 ** 1.5)
    dt = 1e-14 * np.exp(2000 / T_0) * (0.1 / rho_0 ** 1.5)
    time = np.logspace(np.log10(dt), np.log10(t_end), n_points)

    # ----- Storage -----
    TEMP = np.zeros(n_points)
    PRESSURE = np.zeros(n_points)
    DENSITY = np.zeros(n_points)
    ENERGY = np.zeros(n_points)
    ENTHALPY = np.zeros(n_points)
    log10_X = {sp: np.zeros(n_points) for sp in species_names}

    species_indices = {}
    for name in species_names:
        try:
            species_indices[name] = gas_react.species_index(name)
        except ValueError:
            species_indices[name] = None

    min_fraction = 1e-16

    # ----- Time evolution -----
    for i in range(n_points):
        reactor_net.advance(time[i])
        TEMP[i] = reactor.thermo.T
        PRESSURE[i] = reactor.thermo.P
        DENSITY[i] = reactor.thermo.density
        ENERGY[i] = reactor.thermo.u + E_const
        ENTHALPY[i] = reactor.thermo.h + E_const

        X = reactor.thermo.X
        for sp in species_names:
            idx = species_indices[sp]
            if idx is not None:
                log10_X[sp][i] = np.log10(max(X[idx], min_fraction))
            else:
                log10_X[sp][i] = -16

    print(f"  T_eq = {T_eq:.1f} K, T_final = {TEMP[-1]:.1f} K, "
          f"P_final = {PRESSURE[-1]/1000:.2f} kPa")

    # ----- Plot: IC composition + species + temperature -----
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1.5], hspace=0.45)

    # (a) Initial composition bar chart
    ax0 = fig.add_subplot(gs[0])
    bars = ax0.bar(init_species, ic, color=ic_colors, edgecolor='black')
    for bar, frac in zip(bars, ic):
        ax0.text(bar.get_x() + bar.get_width() / 2, frac,
                 f'{frac:.4f}', ha='center', va='bottom', fontsize=10)
    ax0.set_ylabel('Mole Fraction')
    ax0.set_ylim(0, max(ic) * 1.2)
    ax0.set_title(f'IC #{ic_idx + 1}  —  Initial Composition  '
                  f'(sum = {ic.sum():.4f}, T₀ = {T_0} K, ρ₀ = {rho_0} kg/m³)')
    ax0.grid(True, axis='y', alpha=0.3)

    # (b) Species evolution
    ax1 = fig.add_subplot(gs[1])
    for sp, c in zip(species_names, species_colors):
        ax1.semilogx(time, log10_X[sp], color=c, linewidth=2, label=sp)
    ax1.set_xlabel('Time (s) — log scale')
    ax1.set_ylabel('log₁₀(Mole Fraction)')
    ax1.set_title('Species Evolution')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (c) Temperature evolution
    ax2 = fig.add_subplot(gs[2])
    ax2.semilogx(time, TEMP, 'r-', linewidth=2, label='T(t)')
    ax2.axhline(T_0, color='green', linestyle='--', alpha=0.7,
                label=f'T₀ = {T_0:.0f} K')
    ax2.axhline(T_eq, color='blue', linestyle='-', alpha=0.7,
                label=f'T_eq = {T_eq:.0f} K')
    ax2.set_xlabel('Time (s) — log scale')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature Evolution')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig_path = os.path.join(out_dir, f'IC_{ic_idx + 1:02d}_evolution.png')
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ figure → {fig_path}")

    # ----- Save per-IC CSV -----
    X_data_full = np.column_stack([10 ** log10_X[sp] for sp in species_names])
    df = pd.DataFrame({
        'time': time,
        'log10_t': np.log10(time),
        **{f'X_{sp}': X_data_full[:, j] for j, sp in enumerate(species_names)},
        **{f'log10_X_{sp}': log10_X[sp] for sp in species_names},
        'T_K': TEMP,
        'P_Pa': PRESSURE,
        'rho_kgm3': DENSITY,
        'energy_Jkg': ENERGY,
        'enthalpy_Jkg': ENTHALPY,
    })
    df['IC_index'] = ic_idx + 1
    for sp, frac in zip(init_species, ic):
        df[f'IC0_{sp}'] = frac

    csv_path = os.path.join(out_dir, f'IC_{ic_idx + 1:02d}_training_data.csv')
    df.to_csv(csv_path, index=False, float_format='%.8e')
    print(f"  ✓ csv    → {csv_path}")

    all_data.append(df)

# ---------------- Combined dataset ----------------
combined = pd.concat(all_data, ignore_index=True)
combined_path = os.path.join(out_dir, 'training_data_all_ICs.csv')
combined.to_csv(combined_path, index=False, float_format='%.8e')

# ---------------- Summary plot of the 10 ICs ----------------
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(n_ICs)
bottom = np.zeros(n_ICs)
for j, sp in enumerate(init_species):
    ax.bar(x, random_ICs[:, j], bottom=bottom, color=ic_colors[j],
           label=sp, edgecolor='black', linewidth=0.5)
    bottom += random_ICs[:, j]
ax.set_xticks(x)
ax.set_xticklabels([f'IC {i+1}' for i in range(n_ICs)])
ax.set_ylabel('Mole Fraction')
ax.set_title(f'Random Initial Compositions (Dirichlet, n = {n_ICs}, sum = 1)')
ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
summary_path = os.path.join(out_dir, 'IC_summary_stacked.png')
plt.savefig(summary_path, dpi=120, bbox_inches='tight')
plt.close(fig)

print("\n" + "=" * 60)
print(f"Done. {n_ICs} initial conditions × {n_points} time points")
print(f"  combined CSV   → {combined_path}")
print(f"  IC summary fig → {summary_path}")
print(f"  per-IC outputs → {out_dir}")
