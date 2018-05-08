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
  Partition apply(Graph &g, const vector<vector<double>> weights, double eps) {
    g.prepare(weights, eps);
    // inner imbalance
    n = g.n;

    for (auto& v : g.vertices) {
      v.p = 0.02 * (Rand::nextRand() - 1);
    }
    cerr << "Objective each 10 iterations: ";
    project(g);

    for (int iter = 0; ; iter++) {
      if (iter % 10 == 0 && (isFinished(g) && iter >= 500)) // The strange order to output cut
        break;
      computeGradient(g);
      double stepSize = step(iter);
      parallel_for(g.vertices, [stepSize] (Vertex& v) {
        v.p += v.grad * stepSize;
      });
      project(g);
    }
    cerr << endl;
    Partition part(n, 2, eps);
    for (auto& v : g.vertices) {
      part.move(v.id, Rand::check(0.5 * (1 + v.p)) ? 1 : 0);
    }
    return part;
  }

protected:
  /// Step size (depending on the iteration).
  virtual double step(int iteration) = 0;
  /// Gradient computation function
  virtual void computeGradient(Graph &g) = 0;
  /// Probabilistic cut
  virtual double cut(Graph &g) = 0;
  /// Projection function. The argument is imbalance
  virtual void project(Graph &g) = 0;

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
  virtual bool isFinished(Graph& g) {
    double val = cut(g);
    cerr << static_cast<long long int>(val) << " ";
    if (val > 0.1)
      val *= pow(10, 4 - (int)log10(val)); // Scale val to be between 1000 and 10000, I hope
    return ++cntVal[(int)val] > 2;
  }
};

#endif
