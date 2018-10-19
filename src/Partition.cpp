#ifndef PARTITION_CPP
#define PARTITION_CPP

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
#include "Graph.cpp"

struct Partition {
  /// List of all vertices in the partition
  vector <vector<int>> part;
  /// Partition of the vertex
  vector<int> map;
  /// auxilary array to add or delete vertex faster from parts
  vector<int> pos;
  int n, k, upperBound, lowerBound;

  Partition() {}

  Partition(const Partition &other) = default;

  Partition(int n, int k, double eps) : n(n), k(k) {
    map = vector<int>(n);
    pos = vector<int>(n);
    part = vector<vector<int>>(k);
    part[0] = vector<int>(n);
    for (int i = 0; i < n; i++) {
      map[i] = 0;
      pos[i] = i;
      part[0][i] = i;
    }
    lowerBound = static_cast<int>((1 - eps) * n / k);
    upperBound = static_cast<int>(ceil((1 + eps) * n / k));
  }

  inline void move(int node, int newPart) {
    if (map[node] == newPart)
      return;
    vector<int>& old = part[map[node]];
    int nodePos = pos[node];
    swap(old[nodePos], old[old.size() - 1]);
    pos[old[nodePos]] = nodePos;
    old.pop_back();

    map[node] = newPart;
    pos[node] = static_cast<int>(part[newPart].size());
    part[newPart].push_back(node);
  }

  inline int getPart(int node) const {
    return map[node];
  }

  bool check() {
    for(auto& p : part)
      if (p.size() > upperBound || p.size() < lowerBound)
        return false;
    return true;
  }

  double cut(Graph& g) const {
    return parallel_sum(g.vertices, [&] (const Vertex& v) {
      double res = 0;
      for (int u : v.edges)
        if (map[u] != map[v.id])
          res++;
      return res;
    }) / 2;
  }

  void fixPartition() {
    for (int p = 0; p < k; p++) {
      while (part[p].size() < lowerBound) {
        int j = 0;
        while (part[j].size() <= lowerBound)
          j = Rand::next(k);
        move(part[j][part[j].size() - 1], p);
      }
      while (part[p].size() > upperBound) {
        int j = 0;
        while (part[j].size() >= upperBound)
          j = Rand::next(k);
        move(part[p][part[p].size() - 1], j);
      }
    }
  }

  vector<double> innerDegrees(const Graph& g) const {
    vector<double> res(k);
    for (const Vertex& v : g.vertices)
      res[map[v.id]] += g.vertices[v.id].degree;
    return res;
  }

  vector<double> innerEdges(const Graph& g) const {
    vector<double> res(k);
    for (const Vertex& v : g.vertices)
      for (int u : v.edges)
        if (map[u] == map[v.id] && u < v.id)
          res[map[u]] ++;
    return res;
  }
};

#endif
