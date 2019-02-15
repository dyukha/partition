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
  Partition apply(Graph &g, const vector<vector<double>> weights, double finalEps, int totalDepth, bool adapt = true, bool fix = true) {
    // In the worst case imbalance on each step is multiplied
    // (1+eps)^depth <= 1+finalEps   <=>    eps <= (1+finalEps)^(1/depth) - 1
    double eps = pow(1 + finalEps, 1.0 / totalDepth) - 1;
    g.prepare(weights, eps);
    // inner imbalance
    n = g.n;
    for (auto &v : g.vertices) {
      v.bucket = 0;
    }
    for (int depth = 0; depth < totalDepth; ++depth) {
      for (auto &v : g.vertices) {
        v.p = v.last_p = 0.01 * 2 * (Rand::nextRand() - 1);
        v.fixed = false;
      }
      double val = 0.2;
      g.vertices[0].p = val;
      g.vertices[1].p = -val;
      g.vertices[2].p = val;
      g.vertices[3].p = -val;

//      project(g, depth);
      parallel_for(g.vertices, [&](Vertex &v) {
        v.grad = 0;
        for (int u : v.edges) {
          v.grad += g.vertices[u].p;
        }
      });
      int numBuckets = 1 << depth;
      vector<double> coefs(numBuckets);
      for (int bucket = 0; bucket < numBuckets; ++bucket) {
        coefs[bucket] = 1;
      }

      cerr << "Weights: ";
      for (auto& v : g.vertices) {
        cerr << v.w[0] << " ";
      }
      cerr << endl;
      double allowedImbalance = eps / pow(2, totalDepth - depth - 1);
      for (int iter = 0; iter < 101; iter++) {
//        if (iter % 3 == 0) {
//          printStats(g, "Iteration " + my_to_string(iter), depth, coefs);
//        }
        cerr << "Iteration " << iter << endl;
        cerr << "Values: ";
        for (auto& v : g.vertices) {
          cerr << v.p << " ";
        }
        cerr << endl;
        cerr << "Gradient: ";
        for (auto& v : g.vertices) {
          cerr << v.grad << " ";
        }
        cerr << endl;
        computeGradient(g);
        double stepSize = adapt ? step(iter) * sqrt(n >> depth) : step(iter);
        parallel_for(g.vertices, [](Vertex &v) {
          v.prev_p = v.p;
        });

        vector<double> len(numBuckets);
        g.for_not_fixed([&](Vertex& v) {
          len[v.bucket] += v.grad * v.grad;
        });
        for (int bucket = 0; bucket < numBuckets; ++bucket) {
          len[bucket] = sqrt(len[bucket]);
        }
        parallel_for(g.vertices, [&](Vertex &v) {
          if (!v.fixed) {
            v.p += v.grad * stepSize  * (adapt ? coefs[v.bucket] / len[v.bucket] : 1);
          }
        });
        cerr << "After gradient step: ";
        for (auto& v : g.vertices) {
          cerr << v.grad << " ";
        }
        cerr << endl;
        project(g, depth);
        project(g, depth);
        //if (getMax(imbalance(g, depth)) > allowedImbalance) {
        //  if (iter > 80) {
        //  }
        //}
        vector<double> dif(numBuckets);
        g.for_not_fixed([&](Vertex& v) {
          dif[v.bucket] += (v.p - v.prev_p) * (v.p - v.prev_p);
        });
        for (int bucket = 0; bucket < numBuckets; ++bucket) {
          dif[bucket] = sqrt(dif[bucket]);
          if (dif[bucket] > 1e-14) {
            coefs[bucket] *= stepSize / dif[bucket];
          }
        }
        if (fix) {
          g.for_not_fixed([&](Vertex &v) {
            if (abs(v.p) > 1 - 0.001) {
              v.fixed = true;
              v.p = v.p > 0 ? 1 : -1;
            }
          });
        }
      }
      cerr << "Allowed imbalance: " << allowedImbalance << endl;
      for (Vertex& v : g.vertices) {
        v.fixed = false;
      }
      for (int t = 0; t < 10; ++t) {
        project(g, depth);
      }
      for (int t = 0; t < 20 && getMax(imbalance(g, depth)) > allowedImbalance; t++) {
//          cerr << getMax(imbalance(g, depth)) << " ";
        project(g, depth);
      }
      printStats(g, "After fixing imbalance", depth, coefs);
      for (auto &v : g.vertices) {
        v.p = (Rand::check(0.5 * (1 + v.p)) ? 1 : -1);
      }
      printStats(g, "Final", depth, coefs);
      for (auto &v : g.vertices) {
        v.bucket = 2 * v.bucket + (int)lrint((v.p + 1) / 2);
      }
    }
    cerr << endl;
    Partition part(n, 1 << totalDepth, finalEps);
    for (auto &v : g.vertices) {
      part.move(v.id, v.bucket);
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

  /// Projection function. The arguments are graph and imbalance
  virtual void project(Graph &g, int depth) = 0;

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
  virtual bool isFinished(Graph &g) {
    double val = cut(g);
    cerr << static_cast<long long int>(val) << " ";
    if (val > 0.1)
      val *= pow(10, 4 - (int) log10(val)); // Scale val to be between 1000 and 10000, I hope
    return ++cntVal[(int) val] > 2;
  }

private:
  vector<double> imbalance(const Graph &g, int depth) {
    updateMem();
    int bucketCount = 1 << (depth + 1);
    vector<double> res(g.constraintsCount);
    for (int c = 0; c < (int)g.constraintsCount; ++c) {
      vector<double> imb(bucketCount);
      double sum = 0;
      for (const Vertex &v : g.vertices) {
        sum += v.w[c];
        imb[v.bucket << 1] += v.w[c] * (1 + v.p) / 2;
        imb[(v.bucket << 1) + 1] += v.w[c] * (1 - v.p) / 2;
      }
      sum /= bucketCount;
      res[c] = abs(imb[0] / sum - 1);
      for (int bucket = 1; bucket < bucketCount; ++bucket) {
        res[c] = max(res[c], abs(imb[bucket] / sum - 1));
      }
    }
    return res;
  }

  void printStats(Graph &g, string iter, int depth, const vector<double>& coefs) {
    auto imb = imbalance(g, depth);
    int numBuckets = 1 << depth;
    vector<double> integral(numBuckets);
    vector<int> cnt(numBuckets);
    for (Vertex& v : g.vertices) {
      integral[v.bucket] += 1 - abs(v.p);
      cnt[v.bucket] ++;
    }
    for (int bucket = 0; bucket < numBuckets; ++bucket) {
      integral[bucket] /= cnt[bucket];
    }
    cerr << "Depth " << depth << "   " << iter << endl;
    double cutVal = cut(g);
    cerr << "Cut " << cutVal << " " << cutVal / g.edgeCount * 100 << "%" << endl;
    cerr << "Imbalance ";
    for (double v : imb) {
      cerr << v << " ";
    }
    cerr << endl;
    cerr << "Integral (min/max) " << getMin(integral) << " " << getMax(integral) << endl;
    cerr << "Coefs (min/max) " << getMin(coefs) << " " << getMax(coefs) << endl;
  }
};

#endif
