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
  vector<vector<int>> g;

  Graph(int n, const vector<tuple<int, int>>& e) : n(n) {
    g = vector<vector<int>>(n);
    for (auto p : e) {
      int u, v;
      tie(u, v) = p;
      g[u].push_back(v);
      g[v].push_back(u);
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
};

#endif //Graph
