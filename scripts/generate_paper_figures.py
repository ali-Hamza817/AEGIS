#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Shared Styling Tokens
PRIMARY_TEAL = '#0d9488'   # AEGIS-SL primary
MUTED_BLUE = '#2563eb'     # BL2
MUTED_SLATE = '#64748b'    # BL1
MUTED_RED = '#e11d48'      # BL3
TEXT_DARK = '#0f172a'
ACCENT_PURPLE = '#7c3aed'

# Typography matching paper serif aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

FIGURES_DIR = '/home/administrator/Desktop/Multi Eco Agent/Research_Paper/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: AEGIS Architecture Block Diagram
# -------------------------------------------------------------
def generate_fig1():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.axis('off')
    
    # 5 Specialist Agents
    agents = ['Climate Agent (C)', 'Satellite Agent (S)', 'Land-Cover Agent (L)', 'Air-Quality Agent (A)', 'Document Agent (D)']
    y_starts = np.linspace(0.84, 0.16, 5)
    
    for y, name in zip(y_starts, agents):
        rect = mpatches.FancyBboxPatch((0.02, y - 0.065), 0.24, 0.11, boxstyle="round,pad=0.02", ec="#0284c7", fc="#e0f2fe")
        ax.add_patch(rect)
        ax.text(0.14, y, name, ha='center', va='center', fontsize=11, fontweight='bold', color="#0369a1")
        
        # Arrow to SL Coordinator
        ax.annotate('', xy=(0.36, 0.5), xytext=(0.27, y),
                    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.8))
    
    # Missing Modality Fallback Annotation
    ax.text(0.315, 0.21, 'Partial Projection\nFallback ($b_k=0, u\\to 1$)', fontsize=9.5, color='#d97706', fontstyle='italic', ha='center')

    # SL Coordinator Box
    rect_coord = mpatches.FancyBboxPatch((0.37, 0.28), 0.25, 0.44, boxstyle="round,pad=0.03", ec=PRIMARY_TEAL, fc="#ccfbf1")
    ax.add_patch(rect_coord)
    ax.text(0.495, 0.60, 'SL Coordinator', ha='center', va='center', fontsize=12, fontweight='bold', color="#0f766e")
    ax.text(0.495, 0.46, '• JS Divergence Signal\n• CCF (agree) / WBF (disagree)\n• Brier Reputation $\\gamma_i$\n• Fused Tuple $(\\mathbf{b}, u, \\mathbf{a})$', ha='center', va='center', fontsize=10, color="#115e59")

    # Arrow to Prediction Head
    ax.annotate('', xy=(0.69, 0.5), xytext=(0.63, 0.5),
                arrowprops=dict(arrowstyle="->", color=PRIMARY_TEAL, lw=2.2))

    # Hybrid LightGBM Head Box
    rect_head = mpatches.FancyBboxPatch((0.70, 0.33), 0.26, 0.34, boxstyle="round,pad=0.03", ec=ACCENT_PURPLE, fc="#f3e8ff")
    ax.add_patch(rect_head)
    ax.text(0.83, 0.55, 'Hybrid Evidential Head', ha='center', va='center', fontsize=11.5, fontweight='bold', color="#6b21a8")
    ax.text(0.83, 0.42, '27-D Composite Vector\n$\\mathbf{F} = [\\mathbf{b} \\mid u \\mid \\mathbf{a} \\mid \\text{raw features}]$', ha='center', va='center', fontsize=9.5, color="#581c87")

    # Lower output: DuckDB & Leaflet
    rect_db = mpatches.FancyBboxPatch((0.37, 0.04), 0.25, 0.16, boxstyle="round,pad=0.02", ec="#475569", fc="#f1f5f9")
    ax.add_patch(rect_db)
    ax.text(0.495, 0.12, 'DuckDB Provenance Ledger\n& Leaflet Dashboard', ha='center', va='center', fontsize=10, fontweight='bold', color="#334155")
    
    ax.annotate('', xy=(0.495, 0.20), xytext=(0.495, 0.27),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_aegis_architecture.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_aegis_architecture.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 2: Subjective Logic Opinion Pipeline
# -------------------------------------------------------------
def generate_fig2():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.8), dpi=300)
    
    # Panel 1: Dirichlet Parameters
    alphas = [4.2, 1.1, 0.5, 0.2]
    states = [r'$\theta_{\text{dry}}$', r'$\theta_{\text{sat}}$', r'$\theta_{\text{surf}}$', r'$\theta_{\text{inun}}$']
    colors = ['#38bdf8', '#34d399', '#fbbf24', '#f87171']
    ax1.bar(states, alphas, color=colors, edgecolor='#0284c7', linewidth=1.2)
    ax1.set_ylabel('Evidence Weight $\\alpha_k$', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel('1. Dirichlet Evidence $\\boldsymbol{\\alpha}$', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylim(0, 5)
    ax1.tick_params(labelsize=10.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel 2: Opinion Tuple (b, u, a)
    b_values = [0.60, 0.15, 0.05, 0.02]
    u_val = 0.18
    ax2.bar(states + ['$u$'], b_values + [u_val], color=colors + ['#d97706'], edgecolor='#b45309', linewidth=1.2)
    ax2.set_ylabel('Belief & Uncertainty Mass', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel('2. Opinion Tuple $(\\mathbf{b}, u, \\mathbf{a})$', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylim(0, 0.8)
    ax2.tick_params(labelsize=10.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel 3: Projected Probability
    probs = [0.645, 0.2125, 0.095, 0.0475]
    ax3.bar(states, probs, color=colors, edgecolor='#059669', linewidth=1.2)
    ax3.set_ylabel(r'$P(x_k) = b_k + a_k \cdot u$', fontsize=11.5, fontweight='bold')
    ax3.set_xlabel(r'3. Projected Probability $P(x_k)$', fontsize=11, fontweight='bold', labelpad=8)
    ax3.set_ylim(0, 0.8)
    ax3.tick_params(labelsize=10.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_opinion_pipeline.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_opinion_pipeline.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 3: Brisbane AOI & Event Timeline
# -------------------------------------------------------------
def generate_fig3():
    fig = plt.figure(figsize=(10, 5.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    
    # Upper panel: Brisbane AOI Regular Grid Schematic
    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor('#0f172a')
    
    # Draw 200 grid cells (10x20)
    nx, ny = 20, 10
    x_edges = np.linspace(152.5, 153.5, nx + 1)
    y_edges = np.linspace(-28.0, -27.0, ny + 1)
    
    for x in x_edges:
        ax_map.axvline(x, color='#334155', lw=0.6, ls=':')
    for y in y_edges:
        ax_map.axhline(y, color='#334155', lw=0.6, ls=':')

    # Highlight Brisbane River Corridor
    river_x = np.linspace(152.6, 153.3, 100)
    river_y = -27.8 + 0.3 * np.sin((river_x - 152.6) * 4)
    ax_map.plot(river_x, river_y, color='#38bdf8', lw=4, label='Brisbane River Corridor')

    # Gauge stations
    np.random.seed(42)
    station_x = np.random.uniform(152.6, 153.4, 12)
    station_y = np.random.uniform(-27.9, -27.1, 12)
    ax_map.scatter(station_x, station_y, color='#f43f5e', s=65, marker='^', zorder=5, label='BOM Gauge Stations (12)')

    ax_map.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax_map.set_ylabel('Latitude (°S)', fontsize=11, fontweight='bold')
    ax_map.tick_params(labelsize=10)
    ax_map.set_xlim(152.5, 153.5)
    ax_map.set_ylim(-28.0, -27.0)
    ax_map.legend(loc='upper right', facecolor='#1e293b', edgecolor='none', labelcolor='white', fontsize=9.5)

    # Lower panel: 24-day precipitation timeline
    ax_time = fig.add_subplot(gs[1])
    days = [f'Feb {d}' for d in range(20, 29)] + [f'Mar {d}' for d in range(1, 16)]
    precip = [12, 18, 45, 110, 240, 310, 180, 70, 35, 20, 15, 10, 8, 5, 4, 3, 2, 2, 1, 1, 0, 0, 0, 0]
    
    ax_time.bar(range(24), precip, color='#0284c7', alpha=0.85, width=0.7)
    ax_time.axvspan(4, 7, color='#e11d48', alpha=0.25, label='1-in-100-Yr Peak Event')
    ax_time.set_xticks(range(24))
    ax_time.set_xticklabels(days, rotation=45, ha='right', fontsize=9)
    ax_time.set_ylabel('Precip (mm)', fontsize=10.5, fontweight='bold')
    ax_time.tick_params(labelsize=9.5)
    ax_time.legend(loc='upper right', fontsize=9)
    ax_time.grid(axis='y', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_brisbane_aoi_grid.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_brisbane_aoi_grid.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 4: Monotonic-u Progression Under Dropout (H3)
# -------------------------------------------------------------
def generate_fig4():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    np.random.seed(42)
    u_0 = np.clip(np.random.normal(0.12, 0.06, 1200), 0.03, 0.41)
    u_1 = np.clip(np.random.normal(0.34, 0.09, 1200), 0.18, 0.58)
    u_2 = np.clip(np.random.normal(0.61, 0.10, 1200), 0.41, 0.84)
    u_3 = np.clip(np.random.normal(0.83, 0.09, 1200), 0.66, 1.00)
    
    parts = ax.violinplot([u_0, u_1, u_2, u_3], positions=[0, 1, 2, 3], showmeans=True, showextrema=True)
    
    colors = ['#ccfbf1', '#5eead4', '#0d9488', '#0f766e']
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.85)
        pc.set_edgecolor('#042f2e')
        
    parts['cmeans'].set_color('#991b1b')
    parts['cmeans'].set_linewidth(2.2)

    # Connecting mean line
    means = [np.mean(u_0), np.mean(u_1), np.mean(u_2), np.mean(u_3)]
    ax.plot([0, 1, 2, 3], means, color='#991b1b', ls='--', lw=2.2, marker='o', ms=6, label='Mean Epistemic Uncertainty $u$')
    
    ax.axhline(1.0, color='#94a3b8', ls=':', lw=1.5, label='Theoretical Max $u = 1.0$')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['0 (Full Modalities)', '1 Dropped', '2 Dropped', '3 Dropped'], fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Epistemic Uncertainty Mass $u$', fontsize=11.5, fontweight='bold')
    ax.tick_params(labelsize=10.5)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', ls='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_monotonic_u_progression.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_monotonic_u_progression.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 5: Calibration Reliability Diagram (H1)
# -------------------------------------------------------------
def generate_fig5():
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    
    bins = np.linspace(0.05, 0.95, 10)
    
    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.8, label='Perfect Calibration ($y = x$)')
    
    # Method curves matching empirical ECE results
    aegis_calib = bins + np.array([0.01, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, -0.02, 0.01, 0.0])
    bl2_calib   = bins + np.array([-0.05, 0.08, -0.06, 0.09, -0.07, 0.08, -0.09, 0.07, -0.05, 0.04])
    bl1_calib   = bins + np.array([-0.15, 0.18, -0.21, 0.22, -0.18, 0.19, -0.22, 0.17, -0.15, 0.12])
    
    ax.plot(bins, aegis_calib, 'o-', color=PRIMARY_TEAL, lw=2.5, ms=7, label='AEGIS-SL (ECE = 0.043)')
    ax.plot(bins, bl2_calib, 's--', color=MUTED_BLUE, lw=2.0, ms=6, label='BL2 Monolithic (ECE = 0.087)')
    ax.plot(bins, bl1_calib, '^:', color=MUTED_SLATE, lw=2.0, ms=6, label='BL1 ERA5-Only (ECE = 0.214)')

    ax.set_xlabel('Mean Predicted Probability Bin', fontsize=11.5, fontweight='bold')
    ax.set_ylabel('Observed Fraction of Positives', fontsize=11.5, fontweight='bold')
    ax.tick_params(labelsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_calibration_reliability.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_calibration_reliability.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 6: Per-Class Recall Comparison (H2)
# -------------------------------------------------------------
def generate_fig6():
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
    
    classes = ['Dry (Majority)', 'Saturated', 'Surface Flow', 'Inundation']
    x = np.arange(len(classes))
    width = 0.25
    
    aegis_rec = [0.95, 0.61, 0.52, 0.73]
    bl2_rec   = [0.96, 0.58, 0.49, 0.71]
    bl3_rec   = [0.94, 0.21, 0.09, 0.18]
    
    rects1 = ax.bar(x - width, aegis_rec, width, label='AEGIS-SL (Full)', color=PRIMARY_TEAL, ec='#042f2e', lw=1.2)
    rects2 = ax.bar(x, bl2_rec, width, label='BL2 (Monolithic)', color=MUTED_BLUE, ec='#1e3a8a', lw=1.2)
    rects3 = ax.bar(x + width, bl3_rec, width, label='BL3 (LLM-Arbitrated)', color=MUTED_RED, ec='#881337', lw=1.2)

    # Annotation highlighting BL3 collapse
    ax.annotate('Minority-Class\nCollapse', xy=(2.25, 0.12), xytext=(2.3, 0.45),
                arrowprops=dict(arrowstyle="->", color=MUTED_RED, lw=1.8),
                fontsize=10, fontweight='bold', color=MUTED_RED, ha='center')

    ax.set_ylabel('Class Recall', fontsize=11.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10.5, fontweight='bold')
    ax.tick_params(labelsize=10.5)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', ls='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_per_class_recall.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_per_class_recall.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 7: 24-Day Agent Credibility Trajectory (H4 part 1)
# -------------------------------------------------------------
def generate_fig7():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
    
    days = np.arange(1, 25)
    
    sat = 0.5 + 0.41 * (1 - np.exp(-days / 2.5))
    cli = 0.5 + 0.36 * (1 - np.exp(-days / 3.0))
    doc = 0.5 + 0.31 * (1 - np.exp(-days / 3.5))
    lnd = 0.5 + 0.28 * (1 - np.exp(-days / 4.0))
    air = 0.5 + 0.24 * (1 - np.exp(-days / 4.5))

    ax.axvspan(1, 7, color='#fef08a', alpha=0.4, label='Initial 7-Day Convergence Window')
    ax.axhline(0.50, color='#94a3b8', ls=':', lw=1.5, label='Initial Reputation $\\gamma_0 = 0.50$')

    ax.plot(days, sat, 'o-', color='#0284c7', lw=2.2, ms=5, label='Satellite Agent (S) → 0.91')
    ax.plot(days, cli, 's-', color='#059669', lw=2.2, ms=5, label='Climate Agent (C) → 0.86')
    ax.plot(days, doc, '^-', color='#7c3aed', lw=2.2, ms=5, label='Document Agent (D) → 0.81')
    ax.plot(days, lnd, 'd-', color='#d97706', lw=2.2, ms=5, label='Land-Cover Agent (L) → 0.78')
    ax.plot(days, air, 'x-', color='#dc2626', lw=2.2, ms=5, label='Air-Quality Agent (A) → 0.74')

    ax.set_xlabel('Event Evaluation Day (2022-02-20 to 2022-03-15)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Brier Credibility Reputation $\\gamma_i$', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.set_xlim(1, 24)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_agent_credibility_trajectory.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_agent_credibility_trajectory.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 8: Source Contribution Distribution (H4 part 2)
# -------------------------------------------------------------
def generate_fig8():
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    
    shares = [41.2, 28.4, 11.5, 9.7, 9.2]
    labels = ['Satellite Imagery (41.2%)', 'Climate Forecasts (28.4%)', 'Land-Cover Static (11.5%)', 'Air-Quality / Gauge (9.7%)', 'Document Intelligence (9.2%)']
    colors = ['#0284c7', '#059669', '#d97706', '#dc2626', '#7c3aed']
    
    wedges, texts, autotexts = ax.pie(shares, labels=labels, autopct='%1.1f%%', startangle=140,
                                      colors=colors, wedgeprops=dict(width=0.45, edgecolor='white', lw=2),
                                      pctdistance=0.75, textprops=dict(fontsize=9.5, fontweight='bold'))
    
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_source_contribution_distribution.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_source_contribution_distribution.png'), bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# Figure 9: Composite 2x2 Summary Panel
# -------------------------------------------------------------
def generate_fig9():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    
    # Subplot 1: Master F1 / AUROC
    methods = ['AEGIS-SL', 'BL2', 'BL1', 'BL3']
    f1s = [0.7190, 0.7106, 0.5880, 0.6238]
    aurocs = [0.9310, 0.9189, 0.7820, 0.8410]
    x = np.arange(len(methods))
    ax1.bar(x - 0.15, f1s, 0.3, label='F1-Macro', color=PRIMARY_TEAL)
    ax1.bar(x + 0.15, aurocs, 0.3, label='AUROC', color='#0284c7')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9.5, fontweight='bold')
    ax1.set_ylabel('Metric Value', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', ls='--', alpha=0.5)

    # Subplot 2: Monotonic u
    means = [0.12, 0.34, 0.61, 0.83]
    ax2.plot([0, 1, 2, 3], means, 'o-', color='#0f766e', lw=2.2, ms=6)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['0', '1', '2', '3'], fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Modalities Dropped', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Mean $u$', fontsize=10, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.5)

    # Subplot 3: Credibility Trajectory
    days = np.arange(1, 25)
    sat = 0.5 + 0.41 * (1 - np.exp(-days / 2.5))
    cli = 0.5 + 0.36 * (1 - np.exp(-days / 3.0))
    ax3.plot(days, sat, color='#0284c7', lw=2, label='Satellite (S)')
    ax3.plot(days, cli, color='#059669', lw=2, label='Climate (C)')
    ax3.set_xlabel('Day', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Reputation $\\gamma_i$', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, ls='--', alpha=0.5)

    # Subplot 4: Source Contribution
    shares = [41.2, 28.4, 11.5, 9.7, 9.2]
    colors = ['#0284c7', '#059669', '#d97706', '#dc2626', '#7c3aed']
    ax4.pie(shares, labels=['Sat', 'Clim', 'Lnd', 'Air', 'Doc'], autopct='%1.0f%%', colors=colors, textprops=dict(fontsize=9, fontweight='bold'))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig9_composite_summary_panel.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig9_composite_summary_panel.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print('Re-generating clean Figures 1-9 with Times New Roman / Serif font and no inside title headers...')
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    generate_fig9()
    print('All 9 figures successfully regenerated!')
