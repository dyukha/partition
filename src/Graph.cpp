#ifndef Graph_CPP
#define Graph_CPP

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

const int BufSize = 1000000;
char buffer[BufSize];

struct Vertex {
  int id; // vertex index
  int deg; // degree
//  int part; // corresponding partition
  vector<int> e; // list of neighbors
  double p; // for gradient descent - probability in [-1; 1]
  double grad; // for gradient descent - gradient
  vector<double> w; // Weights for constraints.
//  double prevP; // for gradient descent - previous value of p (while checking that the value changed)
  vector<double> incPlane; // Increments for Dykstra projection. For each plane constraint.
  double incCube; // Increments for Dykstra projection. For cube constraint.
};

struct Graph {
  /// To add each edge exactly once
  struct DuplicateChecker {
    unordered_set<long long int> was;
    void add(int u, int v, vector<tuple<int, int>>& edges) {
      if (u == v)
        return;
      if (u > v)
        swap(u, v);
      auto insertRes = was.insert(((long long int)u << 32) + v);
      if (!insertRes.second)
        return;
      edges.emplace_back(u, v);
    }
  };

  int n;
  size_t constraintsCount;
  vector<Vertex> vertices;
  vector<double> imbalance;

  Graph(int n, const vector<tuple<int, int>>& e) : n(n) {
    vertices = vector<Vertex>(n);
    for (int i = 0; i < n; ++i) {
      vertices[i].id = i;
    }
    for (auto p : e) {
      int u, v;
      tie(u, v) = p;
      vertices[u].e.push_back(v);
      vertices[v].e.push_back(u);
    }
    for (auto& v : vertices) {
      v.deg = v.e.size();
    }
  }

  static ifstream prepareStream(const string &path) {
    ifstream in;
    std::ios::sync_with_stdio(false);
    in.open(path);
    in.rdbuf()->pubsetbuf(buffer, BufSize);
    return in;
  }

  static Graph read(const string &path) {
    string a, b;
    auto in = prepareStream(path);
    unordered_map<string, int> map;
    DuplicateChecker duplicateChecker;
    vector<tuple<int, int>> e;
    while (in >> a) {
      in >> b;
      int u = getOrAdd(a, map);
      int v = getOrAdd(b, map);
      duplicateChecker.add(u, v, e);
    }
    in.close();
    cerr << "Graph is read. " << map.size() << " vertices, " << e.size() << " edges." << endl;
    Graph res = Graph(map.size(), e);
    return res;
  }

  static Graph readNeighboursList(const string &path) {
    string a, b;
    auto in = prepareStream(path);
    unordered_map<string, int> map;
    DuplicateChecker duplicateChecker;
    vector<tuple<int, int>> e;
    while (in >> a) {
      getline(in, b);
//      if (b.empty())
      size_t start = 0U;
      while (start < b.length() && !isdigit(b[start]))
        start++;
      if (start == b.length())
        continue;
      int u = getOrAdd(a, map);
      size_t end = b.find(',');
      while (end != string::npos) {
        int v = getOrAdd(b.substr(start, end - start), map);
        duplicateChecker.add(u, v, e);
        start = end + 1;
        end = b.find(',', start);
//        cerr << end << " ";
      }
      int v = getOrAdd(b.substr(start, end - start), map);
      auto str = b.substr(start, end - start);
      duplicateChecker.add(u, v, e);
    }
    in.close();
    cerr << "Graph is read. " << map.size() << " vertices, " << e.size() << " edges." << endl;
    Graph res = Graph(static_cast<int>(map.size()), e);
    return res;
  }

  void prepare(const vector<vector<double>>& weights, double eps) {
    constraintsCount = weights.size();
    vector<double> len(constraintsCount);
    vector<double> sum(constraintsCount);
    imbalance = vector<double>(constraintsCount);
    for (int c = 0; c < constraintsCount; ++c) {
      assert(n == weights[c].size());
      double sqLen = 0;
      for (int i = 0; i < n; ++i) {
        double w = weights[c][i];
        sum[c] += w;
        sqLen += w * w;
      }
      len[c] = sqrt(sqLen);
      sum[c] /= len[c];
      imbalance[c] = sum[c] * eps * 0.95; // Allow a bit more imbalance to avoid problems during rounding
    }
    parallel_for(vertices, [&](Vertex& v) {
      v.incCube = 0;
      v.incPlane = vector<double>(constraintsCount);
      v.w = vector<double>(constraintsCount);
      for (int c = 0; c < constraintsCount; ++c) {
        v.w[c] = weights[c][v.id] / len[c];
      }
    });
  }
};

#endif //Graph
