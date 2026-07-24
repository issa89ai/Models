import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

edges_file = "data/1775731.edges"
G = nx.read_edgelist(edges_file, create_using=nx.Graph())
print(f"Raw graph -- nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

initial_loops = nx.number_of_selfloops(G)
G.remove_edges_from(nx.selfloop_edges(G))
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)
print(f"Removed {initial_loops} self-loops, {len(isolated_nodes)} isolated nodes")
print(f"Cleaned graph -- nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

degree_centrality = nx.degree_centrality(G)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 by Degree Centrality:")
for node, score in top_degree:
    print(f"  Node {node}: {score:.4f}")

pagerank = nx.pagerank(G)
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 by PageRank:")
for node, score in top_pagerank:
    print(f"  Node {node}: {score:.4f}")

density = nx.density(G)
print(f"\nNetwork density: {density:.4f}")

communities = list(greedy_modularity_communities(G))
mod_score = modularity(G, communities)
print(f"\nGreedy modularity score: {mod_score:.4f}")
print(f"Number of communities found: {len(communities)}")
print(f"Community sizes: {sorted([len(c) for c in communities], reverse=True)}")

community_map = {}
for i, c in enumerate(communities):
    for node in c:
        community_map[node] = i
colors = [community_map[node] for node in G.nodes()]

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=30, with_labels=False)
plt.title(f"Twitter Ego-Network: {len(communities)} communities (greedy modularity={mod_score:.4f})")
plt.savefig('web_mining_communities.png')
print("\nSaved web_mining_communities.png")
plt.show()
