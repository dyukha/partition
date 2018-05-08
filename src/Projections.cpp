#ifndef PROJECTIONS_CPP
#define PROJECTIONS_CPP

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

struct Projections {
  static inline double roundCube(double v) {
    return max(-1.0, min(1.0, v));
  }

  static void simple(Graph& g) {
    for (int i = 0; i < 1; i++) {
      for (int c = 0; c < g.constraintsCount; ++c) {
        double sum = parallel_sum(g.vertices, [&](const Vertex &v) { return v.w[c] * v.p; });
        parallel_for(g.vertices, [&](Vertex &v) { v.p -= v.w[c] * sum; });
      }
      parallel_for(g.vertices, [](Vertex &v) { v.p = roundCube(v.p); });
    }
  }

  static double getDif(Graph& g, double lam) {
    return parallel_sum(g.vertices, [&](const Vertex& v) {return v.p - roundCube(v.p - lam);});
  }

  static void precise1D(Graph& g) {
    assert(g.constraintsCount == 1);
    double idealSum = 0; //- n * proportion + n * (1 - proportion);
    double sum = parallel_sum(g.vertices, [](const Vertex &v) { return v.p; });
    sum -= idealSum;
    double dif = getDif(g, 0);
    double lam;
    double imbalance = g.imbalance[0];
    // dif = initSum - resSum
    // We want resSum = idealSum => dif = initSum - resSum
    if (abs(dif - sum) < imbalance * 1.01) {
      lam = 0;
    } else {
      double maxMod = 1 + parallel_reduce<double>(g.vertices,
                                          [](Vertex& v) {return abs(v.p);},
                                          [](double a, double b) {return max(a, b);});
      double left = dif < sum ? 0 : -maxMod-1;
      double right = dif < sum ? maxMod+1 : 0;
      double cmp = dif < sum ? sum - imbalance : sum + imbalance;
      for (int it = 0; it < 30; ++it) {
        double lam = (left + right) / 2;
        double dif = getDif(g, lam);
        if (abs(dif - cmp) < imbalance * 1.01)
          break;
        if (dif > cmp) {
          right = lam;
        } else {
          left = lam;
        }
      }
      lam = (left + right) / 2;
    }
    parallel_for(g.vertices, [&](Vertex &v) { v.p = roundCube(v.p - lam); });
    // check
    sum = parallel_sum(g.vertices, [](const Vertex &v) { return v.p; });
    assert(abs(sum - idealSum) < imbalance * 1.02);
  }

  static tuple<double, double> getDif2D(Graph& g, double lam1, double lam2) {
    return parallel_sum_tuple(g.vertices, [&](const Vertex& v) {
      double dif = v.p - roundCube(v.p - (v.w[0] * lam1 + v.w[1] * lam2));
      return make_tuple(v.w[0] * dif, v.w[1] * dif);
    });
  }

  // Returns
  // * the value of lam2, for which the first balance function is 0
  // * the value of the second balance function
  static tuple<double, double> getSndLam(Graph& g, double lam1, double sum1, bool adjust, double sum2) {
    double lam2;
    if (abs(get<0>(getDif2D(g, lam1, 0)) - sum1) < g.imbalance[0] && !adjust) {
      lam2 = 0;
    } else {
      double left = -1, right = 1;
      while (sign(get<0>(getDif2D(g, lam1, left)) - sum1) == sign(get<0>(getDif2D(g, lam1, right)) - sum1)) {
        left *= 2, right *= 2;
      }
      // Since we allow some imbalance, make the segment wider
      left *= 2, right *= 2;
      double k = sign(get<0>(getDif2D(g, lam1, right)) - sum1);
      if (k == 0)
        k = 1;
      for (int it = 0; it < 20; it++) {
        double mid = (left + right) / 2;
        double dif = get<0>(getDif2D(g, lam1, mid)) - sum1;
        if (dif * k > g.imbalance[0]) {
          right = mid;
        } else if (dif * k < -g.imbalance[0]) {
          left = mid;
        } else {
          if (!adjust) {
            break;
          } else {
            double dif2 = get<1>(getDif2D(g, lam1, mid)) - sum2;
            if (dif2 > g.imbalance[1]) {
              right = mid;
            } else if (dif2 < -g.imbalance[1]) {
              left = mid;
            } else {
              break;
            }
          }
        }
      }
      lam2 = (left + right) / 2;
    }
    return make_tuple(lam2, get<1>(getDif2D(g, lam1, lam2)));
  }

  static double lam1Dif(Graph& g, double lam1, double sum1) {
    return get<1>(getSndLam(g, lam1, sum1, false, 0));
  }

  static void precise2D(Graph& g) {
    assert(g.constraintsCount == 2);
    double sum1, sum2;
    tie(sum1, sum2) = parallel_sum_tuple(g.vertices, [](const Vertex& v) {
      return make_tuple(v.w[0] * v.p, v.w[1] * v.p);
    });
    double lam1;
    bool adjust;
    if (abs(lam1Dif(g, 0, sum1) - sum2) < g.imbalance[1]) {
      lam1 = 0;
      adjust = false;
    } else {
      adjust = true;
      double left = -1, right = 1;
      while (sign(lam1Dif(g, left, sum1) - sum2) == sign(lam1Dif(g, right, sum1) - sum2)) {
        left *= 2;
        right *= 2;
      }
      int k = sign(lam1Dif(g, right, sum1) - sum2);
      if (k == 0)
        k = 1;
      for (int it = 0; it < 20; it++) {
//        cerr << "! " << it;
        double mid = (left + right) / 2;
        double dif = lam1Dif(g, mid, sum1) - sum2;
//        cerr << " " << dif << " " << left << " " << right << endl;
        if (dif * k > g.imbalance[1]) {
          right = mid;
        } else if (dif * k < -g.imbalance[1]) {
          left = mid;
        } else {
          break;
        }
      }
      lam1 = (left + right) / 2;
    }
    double lam2 = get<0>(getSndLam(g, lam1, sum1, adjust, sum2));
    parallel_for(g.vertices, [&](Vertex& v) {
      v.p = roundCube(v.p - (v.w[0] * lam1 + v.w[1] * lam2));
    });
    // check
    sum1 = sum2 = 0;
    tie(sum1, sum2) = parallel_sum_tuple(g.vertices, [](const Vertex& v) {
      return make_tuple(v.w[0] * v.p, v.w[1] * v.p);
    });
    assert(abs(sum1) < g.imbalance[0] * 1.2);
    assert(abs(sum2) < g.imbalance[1] * 1.2);
  }

  static void dykstra(Graph& g, double) {
    int constraintsCount = g.constraintsCount;
    parallel_for(g.vertices, [&](Vertex& v) {
      for (int c = 0; c < constraintsCount; ++c) {
        v.incPlane[c] = 0;
      }
      v.incCube = 0;
    });
    for (int it = 0; it < 3; ++it) {
      for (int c = 0; c < constraintsCount; ++c) {
        double sum = parallel_sum(g.vertices, [&](Vertex &v) {
          v.p -= v.incPlane[c];
          v.incPlane[c] = 0;
          return v.p;
        });
        if (abs(sum) < g.imbalance[c]) {
          continue;
        }
        double dif = sign(sum) * (abs(sum) - g.imbalance[c]);
        parallel_for(g.vertices, [&](Vertex& v) {
          double newP = v.p - dif;
          v.incPlane[c] = newP - v.p;
          v.p = newP;
        });
      }
      parallel_for(g.vertices, [](Vertex& v) {
        v.p = v.p - v.incCube;
        double newP = roundCube(v.p);
        v.incCube = newP - v.p;
        v.p = newP;
      });
    }
  }
};

#endif
