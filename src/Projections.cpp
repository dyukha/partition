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
#pragma omp parallel for
      for (int u = 0; u < n; ++u)
        p[u] = roundCube(p[u] - dif);
    }
  }

  static inline double roundCube(double v) {
    return max(-1.0, min(1.0, v));
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
#pragma omp parallel for reduction(+:sum)
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
#pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < n; ++i)
      sum += p[i];
    assert(abs(sum - idealSum) < n * (eps + 0.002));
  }

  static tuple<double, double> getDif2D(const vector<double>& p, double lam1, double lam2, const vector<double>& w1, const vector<double>& w2) {
    int n = p.size();
    double dif1 = 0, dif2 = 0;
#pragma omp parallel for reduction(+ : dif1,dif2)
    for (int u = 0; u < n; ++u) {
      double dif = p[u] - roundCube(p[u] - (w1[u] * lam1 + w2[u] * lam2));
      dif1 += w1[u] * dif;
      dif2 += w2[u] * dif;
    }
    return make_tuple(dif1, dif2);
  }

  // Returns
  // * the value of lam2, for which the first balance function is 0
  // * the value of the second balance function
  static tuple<double, double> getSndLam(const vector<double>& p, double lam1, const vector<double>& w1, const vector<double>& w2, double sum1, double imb1, bool adjust, double sum2, double imb2) {
    double lam2;
    if (abs(get<0>(getDif2D(p, lam1, 0, w1, w2)) - sum1) < imb1 && !adjust) {
      lam2 = 0;
    } else {
      double left = -1, right = 1;
      while (sign(get<0>(getDif2D(p, lam1, left, w1, w2)) - sum1) == sign(get<0>(getDif2D(p, lam1, right, w1, w2)) - sum1)) {
//        cerr << "! " << get<0>(getDif2D(p, lam1, left, w1, w2)) - sum1 << " " << get<0>(getDif2D(p, lam1, right, w1, w2)) - sum1 << endl;
        left *= 2, right *= 2;
//        cerr << "! " << get<0>(getDif2D(p, lam1, left, w1, w2)) - sum1 << " " << get<0>(getDif2D(p, lam1, right, w1, w2)) - sum1 << endl;
      }
//      cerr << endl;
      // Since we allow some imbalance, make the segment wider
      left *= 2, right *= 2;
      double k = sign(get<0>(getDif2D(p, lam1, right, w1, w2)) - sum1); 
      if (k == 0)
        k = 1;
      for (int it = 0; it < 20; it++) {
        double mid = (left + right) / 2;
        double dif = get<0>(getDif2D(p, lam1, mid, w1, w2)) - sum1;
//        cerr << it << " " << dif * k << " " << imb1 << endl;
        if (dif * k > imb1) {
          right = mid;
        } else if (dif * k < -imb1) {
          left = mid;
        } else {
          if (!adjust) {
//            cerr << endl << "adj = " << adjust << endl;
            break;
          } else {
            double dif2 = get<1>(getDif2D(p, lam1, mid, w1, w2)) - sum2;
//            cerr << adjust << " " << dif2 << endl;
            if (dif2 > imb2) {
              right = mid;
            } else if (dif2 < -imb2) {
              left = mid;
            } else {
              break;
            }
          }
        }
      }
      lam2 = (left + right) / 2;
    }
//    cerr << "! " << get<0>(getDif2D(p, lam1, lam2, w1, w2)) - sum1  << endl;
    return make_tuple(lam2, get<1>(getDif2D(p, lam1, lam2, w1, w2)));
  }

  static double lam1Dif(const vector<double>& p, double lam1, const vector<double>& w1, const vector<double>& w2, double sum1, double imb1) {
    return get<1>(getSndLam(p, lam1, w1, w2, sum1, imb1, false, 0, 0));
  }

  static void precise2D(vector<double>& p, double eps, const vector<double>& w1, const vector<double>& w2) {
    int n = p.size();
    double sum1 = 0, sum2 = 0;
#pragma omp parallel for reduction(+: sum1,sum2)
    for (int u = 0; u < n; ++u) {
      sum1 += w1[u] * p[u];
      sum2 += w2[u] * p[u];
    }
//    cerr << "(" << sum1 << ", " << sum2 << ") ";
    double imb1 = 0, imb2 = 0;
#pragma omp parallel for reduction(+: imb1,imb2)
    for (int u = 0; u < n; u++) {
      imb1 += w1[u];
      imb2 += w2[u];
    }
    imb1 *= eps, imb2 *= eps;
    double lam1;
    bool adjust;
    if (abs(lam1Dif(p, 0, w1, w2, sum1, imb1) - sum2) < imb2) {
      lam1 = 0;
      adjust = false;
    } else {
      adjust = true;
      double left = -1, right = 1;
      while (sign(lam1Dif(p, left, w1, w2, sum1, imb1) - sum2) == sign(lam1Dif(p, right, w1, w2, sum1, imb1) - sum2)) {
//        cerr << lam1Dif(p, left, w1, w2, sum1, imb1) - sum2 << " " << lam1Dif(p, right, w1, w2, sum1, imb1) - sum2 << endl;
        left *= 2;
        right *= 2;
      }
//      cerr << endl;
      int k = sign(lam1Dif(p, right, w1, w2, sum1, imb1) - sum2); 
      if (k == 0)
        k = 1;
      for (int it = 0; it < 20; it++) {
//        cerr << "! " << it;
        double mid = (left + right) / 2;
        double dif = lam1Dif(p, mid, w1, w2, sum1, imb1) - sum2;
//        cerr << " " << dif << " " << left << " " << right << endl;
        if (dif * k > imb2) {
          right = mid;
        } else if (dif * k < -imb2) {
          left = mid;
        } else {
          break;
        }
      }
      lam1 = (left + right) / 2;
    }
//    cerr << adjust << endl;
    double lam2 = get<0>(getSndLam(p, lam1, w1, w2, sum1, imb1, adjust, sum2, imb2));
//    cerr << adjust << endl;
//    cerr << get<1>(getSndLam(p, lam1, w1, w2, sum1, imb1, adjust, sum2, imb2)) << " ";
#pragma omp parallel for
    for (int u = 0; u < n; ++u)
      p[u] = roundCube(p[u] - (w1[u] * lam1 + w2[u] * lam2));
    // check
    sum1 = sum2 = 0;
#pragma omp parallel for reduction(+: sum1,sum2)
    for (int u = 0; u < n; ++u) {
      sum1 += w1[u] * p[u];
      sum2 += w2[u] * p[u];
    }
//    cerr << "(" << sum1 << ", " << sum2 << " | " << imb1 << ", " << imb2 << ") ";
    //cerr << "s1: " << sum1 << " " << imb1 << endl;
    //cerr << "s2: " << sum2 << " " << imb2 << endl;
    assert(abs(sum1) < imb1 * 1.2);
    assert(abs(sum2) < imb2 * 1.2);
  }

  static void dykstra(vector<double>& p, double eps, double) {
    int n = p.size();
    vector<double> incCube(n), incPlane(n);
    for (int it = 0; it < 3; ++it) {
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        double newP = roundCube(p[u] - incCube[u]);
        incCube[u] = newP - p[u];
        p[u] = newP;
      }
      int n = p.size();
      double sum = 0;
#pragma omp parallel for reduction(+: sum)
      for (int u = 0; u < n; ++u) {
        p[u] -= incPlane[u];
        incPlane[u] = 0;
        sum += p[u];
      }
      double dif = sum / n;
      if (abs(dif) <= eps)
        break;
      dif = sign(dif) * (abs(dif) - eps);
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        double newP = p[u] - dif;
        incPlane[u] = newP - p[u];
        p[u] = newP;
      }
    }
#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
      p[u] = roundCube(p[u] - incCube[u]);
    }
    double sum = 0;
#pragma omp parallel for reduction(+: sum)
    for (int u = 0; u < n; ++u) {
      sum += p[u];
    }
    if (abs(sum / n) > 0.0101)
      cerr << sum / n << " ";
  }

  static void dykstra2D(vector<double>& p, double eps, vector<double>& w1, vector<double>& w2) {
    int n = p.size();
    vector<double> incCube(n), incPlane1(n), incPlane2(n);
//    cerr << "(" << sum1 << ", " << sum2 << ") ";
    double total1 = 0, total2 = 0;
    double len1 = 0, len2 = 0;
#pragma omp parallel for reduction(+: total1,total2,len1,len2)
    for (int u = 0; u < n; u++) {
      total1 += w1[u];
      total2 += w2[u];
      len1 += w1[u] * w1[u];
      len2 += w2[u] * w2[u];
    }
    double imb1 = total1 * eps, imb2 = total2 * eps;
    for (int it = 0; it < 5; ++it) {
#pragma omp parallel for
      for (int u = 0; u < n; ++u) {
        double newP = roundCube(p[u] - incCube[u]);
        incCube[u] = newP - p[u];
        p[u] = newP;
      }
      double sum1 = 0;
#pragma omp parallel for reduction(+: sum1)
      for (int u = 0; u < n; ++u) {
        p[u] -= incPlane1[u];
        incPlane1[u] = 0;
        sum1 += w1[u] * p[u];
      }
      if (abs(sum1) > imb1) {
        double dif1 = sign(sum1) * (abs(sum1) - imb1) / len1;
#pragma omp parallel for
        for (int u = 0; u < n; ++u) {
          double newP = p[u] - w1[u] * dif1;
          incPlane1[u] = newP - p[u];
          p[u] = newP;
        }
      }
      double sum2 = 0;
#pragma omp parallel for reduction(+: sum2)
      for (int u = 0; u < n; ++u) {
        p[u] -= incPlane2[u];
        incPlane2[u] = 0;
        sum2 += w2[u] * p[u];
      }
      if (abs(sum2) > imb2) {
        double dif2 = sign(sum2) * (abs(sum2) - imb2) / len2;
#pragma omp parallel for
        for (int u = 0; u < n; ++u) {
          double newP = p[u] - w2[u] * dif2;
          incPlane2[u] = newP - p[u];
          p[u] = newP;
        }
      }
    }
#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
      p[u] = roundCube(p[u] - incCube[u]);
    }
    double sum1 = 0, sum2 = 0;
#pragma omp parallel for reduction(+: sum1,sum2)
    for (int u = 0; u < n; ++u) {
      sum1 += w1[u] * p[u];
      sum2 += w2[u] * p[u];
    }
    if (abs(sum1) > imb1 * 1.1)
      cerr << sum1 / total1 << "? ";
    if (abs(sum2) > imb2 * 1.1)
      cerr << sum2 / total2 << "! ";
  }
};

#endif
