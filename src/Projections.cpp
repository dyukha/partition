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

struct Projections {
  static void simple(vector<double>& p, double eps, double proportion) {
    int n = p.size();
    double idealSum = - n * proportion + n * (1 - proportion);
    for (int i = 0; i < 5; i++) {
      double sum = 0;
#pragma omp parallel for reduction(+:sum)
      for (int u = 0; u < n; ++u)
        sum += p[u];
      double dif = (sum - idealSum) / n;
      if (abs(dif) < eps)
        return;
#pragma omp parallel for
      for (int u = 0; u < n; ++u)
        p[u] = roundCube(p[u] - dif);
    }
  }


  static inline double roundCube(double v) {
    return max(0.0, min(1.0, v));
  }

  static double getDif(const vector<double>& p, double lam) {
    int n = p.size();
    double dif = 0;
#pragma omp parallel for reduction(+ : dif)
    for (int u = 0; u < n; ++u)
      dif += p[u] - roundCube(p[u] - lam);
    return dif;
  }

  static void presize(vector<double>& p, double eps, double proportion) {
    int n = p.size();
    double idealSum = - n * proportion + n * (1 - proportion);
    double sum = 0;
    for (int u = 0; u < n; ++u)
      sum += p[u];
    sum -= idealSum;
    double dif = getDif(p, 0);
    double lam;
    double imbalance = n * eps;
    // dif = initSum - resSum
    // We want resSum = idealSum => dif = initSum - resSum
    if (abs(dif - sum) < imbalance + n * 0.001) {
      lam = 0;
    } else {
      double maxMod = 0;
      for (int u = 0; u < n; ++u)
        maxMod = max(maxMod, abs(p[u]) - 1);
      double left = dif < sum ? 0 : -maxMod-1;
      double right = dif < sum ? maxMod+1 : 0;
      double cmp = dif < sum ? sum - imbalance : sum + imbalance;
      for (int it = 0; it < 30; ++it) {
        double lam = (left + right) / 2;
        double dif = getDif(p, lam);
        if (abs(dif - cmp) < n * 0.001)
          break;
        if (dif > cmp) {
          right = lam;
        } else {
          left = lam;
        }
      }
      lam = (left + right) / 2;
    }
#pragma omp parallel for
    for (int u = 0; u < n; ++u)
      p[u] = roundCube(p[u] - lam);
    // check
    sum = 0;
    for (int i = 0; i < n; ++i)
      sum += p[i];
    assert(abs(sum - idealSum) < n * (eps + 0.002));
  }
};

#endif