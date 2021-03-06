#ifndef GRADIENT_DESCENT_ONE_DIM_CPP
#define GRADIENT_DESCENT_ONE_DIM_CPP

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

#include "GradientDescent.cpp"

class GradientDescentImpl : public GradientDescent {
  double stepSize;
  function<void (Graph& g, int)> projection;
public:
  explicit GradientDescentImpl(double stepSize, const function<void(Graph&, int)>& projection)
          : stepSize(stepSize), projection(projection) {}

protected:
  double step(int) override {
    return stepSize;
  }

  void computeGradient(Graph &g) override {
    parallel_for(g.vertices, [&] (Vertex& v) {
      double delta = v.p - v.last_p;
      if (abs(delta) > 1e-5) {
        v.last_p = v.p;
        for (int u : v.edges) {
          if (g.vertices[u].bucket == v.bucket) {
            g.vertices[u].grad += delta; // Not exactly correct, but should be fine
          }
        }
      }
    });
//    parallel_for(g.vertices, [&] (Vertex& v) {
//      v.grad = 0;
//      for (int u : v.edges) {
//        v.grad += g.vertices[u].p;
//      }
//    });
  }

  double cut(Graph &g) override {
    return parallel_sum(g.vertices, [&] (Vertex& v) {
      double sum = 0;
      for (int u : v.edges) {
        if (g.vertices[u].bucket == v.bucket) {
          sum += (1 - v.p * g.vertices[u].p) / 2;
        } else {
          sum ++;
        }
      }
      return sum;
    }) / 2;
  }

  void project(Graph &g, int depth) override {
    projection(g, depth);
  }
};

#endif
