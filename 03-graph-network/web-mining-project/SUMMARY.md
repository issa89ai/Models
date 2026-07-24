# Web Mining Project (Twitter Ego-Network) — What We Learned

## Data

`data/1775731.edges` -- a real Twitter ego-network from the SNAP dataset
collection (copied from the original Master's CS source folder; excluded
from git via `.gitignore`'s `data/` rule). An "ego network" represents the
connections around one central user -- everyone in it shares a link to that
same person.

## Concept: PageRank vs. Degree Centrality

Degree Centrality is the simplest importance measure: count direct
connections. PageRank (Google's original web-ranking algorithm, works on any
graph) is recursive instead: **a node is important if important nodes point
to it** -- computed iteratively, redistributing importance scores among
neighbors round after round until they stabilize. A node with fewer total
connections can still outrank a more-connected node, if its connections are
to genuinely influential neighbors.

## Concept: a second, different community-detection algorithm

Girvan-Newman (see `../community-detection/SUMMARY.md`) works top-down:
start with the whole graph as one community, repeatedly remove bridge edges.
`greedy_modularity_communities` works bottom-up: start with every node as its
own community, repeatedly merge whichever pair increases modularity the
most, stopping when no merge helps. Same goal (maximize modularity), opposite
direction.

## Experiment: real Twitter ego-network (46 nodes, 423 edges)

```
Top 5 by Degree Centrality:
  14401912: 0.8000    236657680: 0.8000    18252775: 0.7333
  27696993: 0.6889    20778387: 0.6889

Top 5 by PageRank:
  14401912: 0.0437    236657680: 0.0392    18252775: 0.0372
  807095: 0.0352       27696993: 0.0346

Network density: 0.4087
Greedy modularity: 0.1823, 3 communities (sizes 21/14/11)
```

**Top 3 are identical in both rankings** -- these nodes are unambiguously
central by any measure. But **rank 4 diverges**: node `807095` doesn't even
appear in Degree Centrality's top 5, yet PageRank ranks it above two nodes
that have more raw direct connections (`27696993`, `20778387`). This is real
data directly demonstrating the concept above: `807095`'s fewer connections
happen to be to already-influential neighbors, boosting its recursive
importance past nodes with a higher raw connection count.

**Density and modularity are directly connected**: this network's density
(0.4087) is remarkably high for a real social network -- roughly 3x Zachary's
Karate Club's density (~0.139) -- because an ego network is dense almost by
construction (everyone shares a connection to the same central person, so
"friend of a friend already knows each other" happens constantly). This
directly explains why this network's achievable modularity (0.1823) is much
lower than the karate club's (0.385): **very dense graphs are inherently
harder to cleanly split into communities**, since there's no natural sparse
cut to exploit -- not a weaker algorithm, a structurally different
(denser) kind of graph. The plot (`web_mining_communities.png`) shows this
directly: one dense tangled cluster with only a couple of nodes dangling off
on single threads, unlike the karate club's visibly clean two-faction split.

## Files in this folder

- `WMProject.ipynb` -- original CS536 coursework notebook
- `data/1775731.edges` -- real Twitter ego-network data (gitignored)
- `test_web_mining.py` -- standalone script version, described above
- `web_mining_communities.png` -- saved visualization
