#ifndef RECURSIVE_CLUSTERING_CPP
#define RECURSIVE_CLUSTERING_CPP

/*
  Copyright (c) 2018 Dmitrii Avdiukhin

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include "utils.cpp"
#include "Partition.cpp"
#include "GradientDescentImpl.cpp"
#include "Projections.cpp"

struct RecursiveClustering {
  static Partition apply (Graph& g, double eps, double step, int k, vector<tuple<int, double>>& cuts, int depth = 0) {
    int n = g.n;
    if (k == 1)
      return Partition(n, 1, eps);
    // Left corresponds to 0, right - to 1
    vector<int> sz = { k/2, k - k/2 };
    cerr << "n = " << g.n <<"; k = " << k << endl;
    Partition split = GradientDescentImpl(step, Projections::precise1D).apply(g, eps);
    double cut = split.cut(g);
    while (cuts.size() <= depth)
      cuts.emplace_back(0, 0);
    get<0>(cuts[depth]) += 1;
    get<1>(cuts[depth]) += cut;
    cerr << "cut = " << cut << endl;
    vector<int> map(n);
    vector<int> cnt(2);
    for (int i = 0; i < n; ++i) {
      map[i] = cnt[split.map[i]]++;
    }
    vector<tuple<int, int> > edges[2];
    for (const Vertex& v : g.vertices)
      for (int u : v.e)
        if (u < v.id && split.map[u] == split.map[v.id])
          edges[split.map[u]].emplace_back(map[u], map[v.id]);
    vector<Partition> part(2);
//#pragma omp parallel for
    for (int t = 0; t < 2; ++t) {
      Graph g1(cnt[t], edges[t]);
      part[t] = apply(g1, eps, step, sz[t], cuts, depth+1);
    }
    Partition res(n, k, eps * 4.5);
    for (int i = 0; i < n; ++i) {
      int p = split.map[i] * sz[0] + part[split.map[i]].map[map[i]];
      res.move(i, p);
    }
    return res;
  }
};

#endif
