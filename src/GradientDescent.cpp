#ifndef GRADIENT_DESCENT_CPP
#define GRADIENT_DESCENT_CPP

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

class GradientDescent {
public:
  /// Partition graph g with imbalance eps.
  /// Can split into not-equal sizes, as defined by proportion parameter.
  Partition apply(const Graph &g, double eps, double proportion) {
    // inner imbalance
    eps *= 0.95;
    n = g.n;
    p = vector<double>(n);
    prevP = vector<double>(n);
    grad = vector<double>(n);

    for (int i = 0; i < n; i++) {
      p[i] = 0.02 * (Rand::nextRand() - 1 + proportion);
      prevP[i] = p[i];
    }
    cerr << "Objective each 10 iterations: ";
    project(eps, proportion);

    computeGradient(g);
    for (int iter = 0; ; iter++) {
      if (iter % 10 == 0 && (isFinished(g) && iter >= 500)) // The strange order to output cut
        break;
      if (iter % 20 == 0)
        computeGradient(g);
      double stepSize = step(iter);
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        p[u] += grad[u] * stepSize;
      }
      project(eps, proportion);
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        if (abs(p[u] - prevP[u]) > 1e-6) {
          double dif = p[u] - prevP[u];
          prevP[u] = p[u];
          for (auto v : g.g[u])
#pragma omp atomic
            grad[v] += dif;
        }
      }
    }
    cerr << endl;
    Partition part(n, 2, eps);
    for (int u = 0; u < n; u++)
      part.move(u, Rand::check(0.5 * (1 + p[u])) ? 1 : 0);
    return part;
  }

protected:
  /// Step size (depending on the iteration).
  virtual double step(int iteration) = 0;
  /// Gradient computation function
  virtual void computeGradient(const Graph &g) = 0;
  /// Probabilistic cut
  virtual double cut(const Graph &g) = 0;
  /// Projection function. The argument is imbalance
  virtual void project(double eps, double proportion) = 0;

  /// Array of probabilities
  vector<double> p;
  /// Array of probabilities on the previous step
  vector<double> prevP;
  /// Gradient array
  vector<double> grad;
  /// Gradient array
  map<double, int> cntVal;
  /// Number of vertices
  int n;

  /// Projection function. The argument is imbalance
  virtual bool isFinished(const Graph& g) {
    double val = cut(g);
    cerr << static_cast<long long int>(val) << " ";
    if (val > 0.1)
      val *= pow(10, 4 - (int)log10(val)); // Scale val to be between 1000 and 10000, I hope
    return ++cntVal[(int)val] > 2;
  }
};

#endif
