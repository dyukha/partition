#ifndef GRADIENT_DESCENT_MANY_CLUSTERS_CPP
#define GRADIENT_DESCENT_MANY_CLUSTERS_CPP

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

#include <omp.h>
#include "utils.cpp"
#include "Graph.cpp"
#include "Partition.cpp"
#include "Projections.cpp"

class GradientDescentManyClusters {

public:
  GradientDescentManyClusters(double stepSize) : stepSize(stepSize) {
  }

/// Partition graph g with imbalance eps.
  /// Can split into not-equal sizes, as defined by proportion parameter.
  Partition apply(const Graph &g, double eps, int k) {
    // inner imbalance
    n = g.n;
    this->k = k;
    p = vector<vector<double>>(n);
    prevP = vector<vector<double>>(n);
    grad = vector<vector<double>>(n);
    for (int i = 0; i < n; ++i) {
      p[i] = vector<double>(k);
      prevP[i] = vector<double>(k);
      grad[i] = vector<double>(k);
    }

    for (int i = 0; i < n; i++)
      for (int j = 0; j < k; ++j)
        prevP[i][j] = p[i][j] = 1. / k + 0.01 * (Rand::nextRand() - 0.5);
    cerr << "Objective each 10 iterations: ";
    project(eps);

    for (int iter = 0; ; iter++) {
      if (iter % 10 == 0 && (isFinished(g) && iter >= 200)) // The strange order to output cut
        break;
      computeGradient(g);
      double stepSize = step(iter);
#pragma omp parallel for
      for (int u = 0; u < n; ++u)
        for (int j = 0; j < k; ++j)
          p[u][j] += grad[u][j] * stepSize;
      project(eps);
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        for (int j = 0; j < k; ++j) {
          if (abs(p[u][j] - prevP[u][j]) > 0.01) {
            double dif = p[u][j] - prevP[u][j];
            prevP[u][j] = p[u][j];
            for (auto v : g.g[u])
#pragma omp atomic
              grad[v][j] += dif;
          }
        }
      }
    }
    cerr << endl;
    Partition part(n, k, eps);
    for (int u = 0; u < n; u++) {
      double val = Rand::nextRand();
      for (int j = 0; j < k; ++j) {
        val -= p[u][j];
        if (val <= 0) {
          part.move(u, j);
          break;
        }
      }
    }
    return part;
  }

protected:
  /// Step size (depending on the iteration).
  inline double step(int iteration) {
    return iteration < 400 ? stepSize * 10 : stepSize;
  }
  /// Gradient computation function
  void computeGradient(Graph &g) {
    parallel_for(g.vertices, [](Vertex& v) {
      for (int j = 0; j < k; ++j) {
        double res = 0;
        for (int u : v.edges)
          res += p[v][j];
        grad[u][j] = res;
      }
    });
  }
  /// Probabilistic cut
  double cut(const Graph &g) {
    double res = 0;
#pragma omp parallel for reduction(+:res)
    for (int u = 0; u < n; ++u) {
      double locRes = g.g[u].size();
      for (int j = 0; j < k; ++j)
        for (int v : g.g[u])
          locRes -= p[u][j] * p[v][j];
      res += locRes / 2;
    }
    return res;
  }
  /// Projection function. The argument is imbalance
  void project(double) {
    int n = p.size();
    for (int it = 0; it < 10; it++) {
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        double sum = 0;
        for (int j = 0; j < k; ++j)
          sum += p[u][j];
        double dif = (sum - 1) / k;
        for (int j = 0; j < k; ++j)
          p[u][j] = max(0.0, min(1.0, p[u][j] - dif));
      }
      double* sum = new double[k];
      vector<double> dif(k);
      for (int i = 0; i < k; i++)
        sum[i] = 0;
      // because of some problem (memory, I guess) it's muc more efficietn to reverse loops order
#pragma omp parallel for reduction(+:sum[:k])
      for (int u = 0; u < n; ++u)
        for (int j = 0; j < k; ++j)
          sum[j] += p[u][j];
      
      for (int i = 0; i < k; i++)
        dif[i] = (sum[i] - n / (double)k) / n;
#pragma omp parallel for
      for (int u = 0; u < n; ++u)
        for (int j = 0; j < k; j++)
          p[u][j] -= dif[j];
      delete[] sum;
   }
  }

  /// Array of probabilities
  vector<vector<double>> p;
  /// Array of probabilities on the previous step
  vector<vector<double>> prevP;
  /// Gradient array
  vector<vector<double>> grad;
  /// Gradient array
  map<double, int> cntVal;
  /// Number of vertices
  int n;
  /// Number of clusters
  int k;

  /// Projection function. The argument is imbalance
  bool isFinished(const Graph& g) {
    double val = cut(g);
    cerr << static_cast<long long int>(val) << " ";
    if (val > 0.1)
      val *= pow(10, 4 - (int)log10(val)); // Scale val to be between 1000 and 10000, I hope
    return ++cntVal[(int)val] > 2;
  }

  double stepSize;
};

#endif
