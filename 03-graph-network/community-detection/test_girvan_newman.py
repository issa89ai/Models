import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

communities = list(nx.community.girvan_newman(G))

modularity_df = pd.DataFrame(
    [
        [k + 1, nx.community.modularity(G, communities[k])]
        for k in range(len(communities))
    ],
    columns=["k", "modularity"],
)

best_k = modularity_df.loc[modularity_df["modularity"].idxmax(), "k"]
best_modularity = modularity_df["modularity"].max()
print(f"Best split: k={int(best_k)} communities, modularity={best_modularity:.4f}")


def create_community_node_colors(graph, communities):
    number_of_colors = len(communities)
    colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


def visualize_communities(graph, communities, i):
    node_colors = create_community_node_colors(graph, communities)
    modularity = round(nx.community.modularity(graph, communities), 6)
    title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
    pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=2)
    plt.subplot(3, 1, i)
    plt.title(title)
    nx.draw(
        graph,
        pos=pos,
        node_size=1000,
        node_color=node_colors,
        with_labels=True,
        font_size=20,
        font_color="black",
    )


fig, ax = plt.subplots(3, figsize=(15, 20))

visualize_communities(G, communities[0], 1)
visualize_communities(G, communities[3], 2)

modularity_df.plot.bar(
    x="k",
    ax=ax[2],
    color="#F2D140",
    title="Modularity Trend for Girvan-Newman Community Detection",
)
plt.savefig('girvan_newman_result.png')
print("Saved girvan_newman_result.png")
plt.show()
