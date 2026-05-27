"""Architecture topology: how the strategy archetypes share core infrastructure.

Renders a dependency network of the engine's nine strategy archetypes and the
shared components they rely on. Hub components (used by many strategies) surface
naturally as high-degree nodes. Output: docs/assets/architecture_topology.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(OUT, exist_ok=True)

NAVY, TEAL, AMBER, GREY = "#1f3a5f", "#2a9d8f", "#e76f51", "#6c757d"

strategies = [
    "pegged", "ar_olivia", "olivia_follow", "wide_spread", "basket_arb",
    "conversion_arb", "options", "pairs_arb", "generic_mm",
]
components = [
    "fair_value", "validate_orders", "BOCPD", "kalman_filter", "ar_regression",
    "black_scholes", "iv_inversion", "vol_smile_fit", "markout_sizing",
    "informed_detector", "product_classifier",
]

# strategy -> components it uses (from the engine's actual structure)
uses = {
    "pegged":         ["fair_value", "validate_orders", "markout_sizing", "BOCPD", "product_classifier"],
    "ar_olivia":      ["ar_regression", "kalman_filter", "informed_detector", "validate_orders", "markout_sizing", "BOCPD"],
    "olivia_follow":  ["informed_detector", "validate_orders", "product_classifier"],
    "wide_spread":    ["fair_value", "validate_orders", "markout_sizing"],
    "basket_arb":     ["fair_value", "validate_orders", "informed_detector", "BOCPD"],
    "conversion_arb": ["validate_orders", "product_classifier", "markout_sizing"],
    "options":        ["black_scholes", "iv_inversion", "vol_smile_fit", "validate_orders", "markout_sizing"],
    "pairs_arb":      ["validate_orders", "product_classifier"],
    "generic_mm":     ["fair_value", "validate_orders", "markout_sizing", "product_classifier"],
}

G = nx.Graph()
for s in strategies:
    G.add_node(s, kind="strategy")
for c in components:
    G.add_node(c, kind="component")
for s, cs in uses.items():
    for c in cs:
        G.add_edge(s, c)

deg = dict(G.degree())
pos = nx.spring_layout(G, k=0.9, seed=7, iterations=200)

fig, ax = plt.subplots(figsize=(13, 9))
# edges
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfcfcf", width=1.0)
# component nodes (sized by how many strategies use them -> hubs are big)
comp_nodes = [n for n in G if G.nodes[n]["kind"] == "component"]
nx.draw_networkx_nodes(G, pos, nodelist=comp_nodes, ax=ax,
                       node_color=TEAL, node_size=[400 + deg[n]*260 for n in comp_nodes],
                       edgecolors="white", linewidths=1.5)
# strategy nodes
strat_nodes = [n for n in G if G.nodes[n]["kind"] == "strategy"]
nx.draw_networkx_nodes(G, pos, nodelist=strat_nodes, ax=ax,
                       node_color=NAVY, node_size=900, edgecolors="white", linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5, font_color="#111",
                        font_weight="bold")

ax.set_title("Engine Architecture Topology: Strategy Archetypes and Shared Infrastructure",
             fontsize=14, fontweight="bold")
# legend + hub callout
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color=NAVY, label="Strategy archetype (9)"),
    mpatches.Patch(color=TEAL, label="Shared component (node size = strategies using it)"),
], loc="upper left", framealpha=0.95, fontsize=10)
hub = max(comp_nodes, key=lambda n: deg[n])
ax.annotate(f"'{hub}' is the central hub\n(used by {deg[hub]} of 9 strategies)",
            xy=pos[hub], xytext=(0.02, 0.04), textcoords="axes fraction",
            fontsize=9.5, color=AMBER,
            arrowprops=dict(arrowstyle="->", color=AMBER))
ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_topology.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("architecture_topology.png written to", OUT)
