# Girvan-Newman Community Detection — What We Learned

## Status of the original files

`1Girvan_Newman_Assignment.ipynb` is an assignment template with mostly
`# TODO` stubs (uses `nx.football_graph()`). `plot_girvan_newman.ipynb` is a
complete, working example (Zachary's Karate Club) -- this is what we ran and
extended with `test_girvan_newman.py`.

## The concept

Graphs are nodes + edges (relationships), a genuinely new data structure
compared to everything in `01-classical-ml/` and `02-deep-learning-cv/`.
Girvan-Newman finds communities by **removing edges** rather than grouping
similar things: it repeatedly deletes whichever edge has the highest
"betweenness centrality" (the edge lying on the most shortest-paths between
other node pairs -- i.e. the bridge connecting two otherwise-separate
clusters). **Modularity** scores how good any given split is (density of
edges within proposed communities vs. random chance); the algorithm doesn't
know when to stop on its own, so you compute modularity at every step and
pick whichever split scored highest.

## Experiment: Zachary's Karate Club (34 nodes, 78 edges)

```
Best split: k=4 (step index), modularity=0.3850, 5 actual communities
```

**The first single edge removal already recovers the club's real historical
split almost exactly** -- this dataset is based on a real 1970s karate club
that actually split into two rival factions (a dispute between the
instructor and the club president, nodes 0 and 33 in the graph). Pure graph
structure, no knowledge of who argued with whom, reconstructs the real-world
outcome -- why this dataset is used in nearly every community-detection
tutorial.

The best-scoring split (5 communities, modularity=0.385) found slightly finer
structure than the famous 2-way split -- real cliques within each main
faction. The modularity-vs-steps bar chart (`girvan_newman_result.png`) rises
for the first several steps, peaks, then steadily falls as further edge
removal fragments the graph into meaningless pieces (down to 34 isolated
single-node "communities" near zero modularity) -- the same
"there's a sweet spot, more isn't better" shape as k in KNN or max_depth in
Decision Trees, except here the algorithm optimizes modularity directly
since there's no separate labeled test set (unsupervised).

## See also

`../web-mining-project/SUMMARY.md` for a second, different community
detection algorithm (greedy modularity, bottom-up instead of top-down) run
on a real, much denser Twitter ego-network -- and why that denser graph's
achievable modularity is structurally much lower than the karate club's.

## Files in this folder

- `1Girvan_Newman_Assignment.ipynb` -- original CS536 assignment template (mostly TODO stubs)
- `plot_girvan_newman.ipynb` -- original complete working example
- `test_girvan_newman.py` -- standalone script version, plus best-split detection
- `girvan_newman_result.png` -- saved visualization
