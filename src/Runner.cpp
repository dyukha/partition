#ifndef RUNNER_CPP
#define RUNNER_CPP

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
#include "Partition.cpp"
//#include "GradientDescentManyClusters.cpp"
#include "GradientDescent.cpp"
#include "GradientDescentImpl.cpp"
//#include "RecursiveClustering.cpp"
#include "Projections.cpp"

bool was[1000000];

ofstream out;

enum Projection {
  PRECISE,
  DYKSTRA,
  ALTER
};

string projectionNames[] = { "Precise", "Dykstra", "Alternating" };

void printRes(const Partition& best, const Graph& g, double cut, ostream& out) {
  out << cut << " (";
  for (const auto& p : best.part)
    out << static_cast<double>(p.size()) * best.k / best.n - 1 << " ";
  out << "| ";
  vector<double> innerDegrees = best.innerDegrees(g);
  double sumDegrees = 0;
  for (double x : innerDegrees)
    sumDegrees += x;
  for (double x : innerDegrees)
    out << x * best.k / sumDegrees - 1 << " ";
  out << ") " << endl;
  out.flush();
}

/// Run solve solutionNumber times. The result of solver is
void runMany(int solutionNumber, const string& fileName, const string& method, const Graph& g, const function<double (const Partition&)>& objective, const function<Partition ()>& solve) {
  measureTime(fileName.substr(fileName.find("/") + 1) + " " + method, [&]() {
    resetMem();
    out << "  " << method << ": ";
    cerr << method << ":" << endl;
    string outs(g.n, ' ');
    ofstream partitionFile;
    partitionFile.open(fileName + "_partitions.txt", ofstream::out);
    vector<pair<double, Partition> > partitions;
    for (int i = 0; i < solutionNumber; i++) {
      try {
        measureTime(to_string(i) + "-th iteration", [&]() {
          Partition partition = solve();
          // Enforce balance
  //      partition.fixPartition();
//        assert(partition.check());
          double val = objective(partition);
          out << val << " ";
          out.flush();
          partitions.emplace_back(val, partition);
          for (int u = 0; u < g.n; u++) {
            outs[u] = static_cast<char> ('0' + partition.map[u]);
          }
          partitionFile << outs << endl;
          partitionFile.flush();
          cerr << endl << i << "-th res = ";
          printRes(partition, g, val, cerr);
        });
      } catch (std::exception& e) {
        cerr << string(e.what()) << endl;
      }
    }
    partitionFile.close();
    int mid = solutionNumber / 2;
    nth_element(partitions.begin(), partitions.begin() + mid, partitions.end(),
                [&](const pair<double, Partition> &a, const pair<double, Partition> &b) { return a.first < b.first; });
    cerr << "result = ";
    printRes(partitions[mid].second, g, partitions[mid].first, cerr);
    out << "| ";
    printRes(partitions[mid].second, g, partitions[mid].first, out);
    cerr << "Memory usage: " << maxMemory / 1e6 << " GB\n";
  });
  cerr << endl;
}

std::function<double (const Partition&)> cut(Graph& g) {
  return [&g] (const Partition& p) { return p.cut(g); };
}

void gradientDescent(Graph &g, double eps, int solutionNumber, double step, const string& fileName, Projection proj, int depth) {
  int n = g.n;
  vector<double> w1(n);
  for (int i = 0; i < n; i++) {
    w1[i] = 1;
  }
  vector<vector<double>> w = {w1};
  function<void (Graph&, int)> projection;
  switch (proj) {
    case PRECISE:
      projection = Projections::precise1D;
      break;
    case DYKSTRA:
      projection = Projections::dykstra;
      break;
    case ALTER:
      projection = Projections::simple;
      break;
  }
  runMany(solutionNumber, fileName, projectionNames[proj] + " (step=" + my_to_string(step) +", eps=" + my_to_string(eps) + ", depth=" + my_to_string(depth) + ")", g, cut(g), [&] {
    return GradientDescentImpl(step, projection).apply(g, w, eps, depth);
  });
}

void gradientDescentMD(Graph &g, int constraints, double eps, int solutionNumber, double step, const string& fileName,
                       Projection proj, int depth, bool adapt = true, bool fixVertices = true) {
  int n = g.n;
  vector<vector<double>> w(constraints, vector<double>(n));
  for (int i = 0; i < n; i++) {
    w[0][i] = 1;
    const Vertex& u = g.vertices[i];
    if (constraints >= 2) {
      w[1][i] = u.degree;
    }
    if (constraints >= 3) {
      w[2][i] = u.dist2size;
    }
    if (constraints >= 4) {
      w[3][i] = u.pagerank;
    }
  }
  function<void (Graph&, int)> projection;
  string methodName;
  switch (proj) {
    case PRECISE:
      if (constraints == 2) {
        projection = Projections::precise2D;
      } else if (constraints == 1) {
        projection = Projections::precise1D;
      } else {
        cerr << "Precize works only for 1D and 2D" << endl;
        assert(false); // Precize works only for 1D and 2D
      }
      break;
    case DYKSTRA:
      projection = Projections::dykstra;
      break;
    case ALTER:
      projection = Projections::simple;
      break;
  }
  runMany(solutionNumber, fileName, projectionNames[proj] +
                                    " (constraints=" + my_to_string(constraints) +
                                    ", step=" + my_to_string(step) +
                                    ", eps=" + my_to_string(eps) +
                                    ", depth=" + my_to_string(depth) +
                                    ", adaptive=" + my_to_string(adapt) +
                                    ", fixVertices=" + my_to_string(fixVertices) +
                                    ")", g, cut(g), [&] {
    return GradientDescentImpl(step, projection).apply(g, w, eps, depth, adapt, fixVertices);
  });
}

//void gradientDescentManyPartsSimultanious(const Graph &g, double eps, int solutionNumber, double step, int k, const string& fileName) {
//  runMany(solutionNumber, fileName, "Grad", g, cut(g), [&] {
//    return GradientDescentManyClusters(step).apply(g, eps, k);
//  });
//}

/*
void gradientDescentManyParts(Graph &g, double eps, int solutionNumber, double step, int k, const string& fileName) {
  runMany(solutionNumber, fileName, "Grad", g, cut(g), [&] {
    vector<tuple<int, double> > cuts;
    auto res = RecursiveClustering::apply(g, eps / 4.5, step, k, cuts);
    get<0>(cuts[0])++;
    for (int i = 0; i < (int)cuts.size() - 1; ++i) {
      get<0>(cuts[i+1]) += get<0>(cuts[i]);
      get<1>(cuts[i+1]) += get<1>(cuts[i]);
    }
    cerr << "Cuts for the corresponding cluster count: ";
    double edgeCount = parallel_sum(g.vertices, [](const Vertex& v) {return v.degree;}) / 2;
    for (auto cut : cuts)
      cerr << "[" << get<0>(cut) << ": " << get<1>(cut) << " (" << (int)(10000. * get<1>(cut) / edgeCount) / 100. << "%)] ";
    cerr << endl;
    return res;
  });
}
*/

void printMetis(Graph &g, const string& fileName, int k) {
  ofstream fout;
  fout.open(fileName);
  fout << g.vertices.size() << " " << (int)round(parallel_sum(g.vertices, [](Vertex& v) { return v.degree; }) / 2) << " ";
  fout << "010 " << k << endl;

  for (const Vertex& u : g.vertices) {
    fout << "1";
    if (k >= 2) {
      fout << " " << u.degree;
    }
    if (k >= 3) {
      fout << " " << u.dist2size;
    }
    if (k >= 4) {
      fout << " " << (int)(1e9 * u.pagerank);
    }
    for (int v : u.edges) {
      fout << " " << v + 1;
    }
    fout << endl;
  }
  fout.close();
}

inline string getPath(const string& fileName) {
  return string("data/") + fileName + string(".txt");
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "Output directory must be specified";
    return -1;
  }
  string dir = argv[1];
  createDir(dir);
  out.open(dir + "/_res.txt", ofstream::out);
  out << "<Graph>: <Method>: <Solutions...> | <Median solution M> (<vertex imbalance for each cluster of M>) (<degree imbalance of M>)" << endl;
  cerr.precision(10);
  cout.precision(10);
  out.precision(7);
  vector<string> files;
  vector<string> darwiniFiles = {"10k", "20k", "50k", "100k", "200k", "500k", "1M", "2M", "5M"};
  vector<string> snapFiles = {/*"facebook", "gplus", "LiveJournal", "orkut", */"so"};
//  vector<string> snapFiles = {"wikiVote", "facebook", "twitter", "gplus", "LiveJournal", "orkut"};
  vector<string> bigFiles = {"p_500000_10", "p_500000_20", "p_500000_50", "p_500000_100", "p500k", "p5M"};
  vector<string> testFiles = {"list"};

//  for (auto& f : darwiniFiles) files.push_back(f);
//  for (auto &f : snapFiles) files.push_back(f);
//  for (auto& f : bigFiles) files.push_back(f);

//  vector<string> darwiniLargeFiles = {"Darwini_10M", "Darwini_50M"};
//  for (auto& f : darwiniLargeFiles) files.push_back(f);
  for (auto &f : testFiles) files.push_back(f);

  vector<string> notExistingFiles;
  for (auto &name : files)
    if (!fileExist(getPath(name)))
      notExistingFiles.push_back(name);
  if (!notExistingFiles.empty()) {
    cerr << "The following input files are not found:" << endl;
    for (const auto &name : notExistingFiles)
      cerr << getPath(name) << endl;
    return -2;
  }

  measureTime("Total", [&]() {
    for (const string &name: files) {
      out << name << ": " << endl;
      cerr << name << ": " << endl;
      Graph g = Graph::read(getPath(name));
      string fileName = dir + "/" + name;
//      double step = 0.02;
      double step = 1;
      double eps = 0;
      int cnt = 1;
      int depth = 1;
      int constraints = 1;
      gradientDescentMD(g, constraints, eps, cnt, step, fileName, PRECISE, depth, false, false);
#if false
      measureTime("MultiDimensional", [&]() {
        for (int constraints = 1; constraints < 5; ++constraints) {
//          printMetis(g, fileName + my_to_string(constraints) + ".metis", constraints);
          gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth);
        }
      });
      measureTime("Projection", [&]() {
        for (double eps : {0.1, 0.01, 0.001}) {
          gradientDescentMD(g, constraints, eps, cnt, step, fileName, DYKSTRA, depth, false, false);
        }
        gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth);
//        gradientDescentMD(g, constraints, 0.01, cnt, step, fileName, PRECISE, depth);
      });
      measureTime("Adaptive", [&]() {
        gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth, false, false);
        gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth, true, false);
        gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth, true, true);
      });
      measureTime("Steps", [&]() {
        for (double step : {1.0, 0.1, 0.05, 0.02, 0.01, 0.001}) {
          gradientDescentMD(g, constraints, eps, cnt, step, fileName, ALTER, depth);
        }
      });
#endif
      out.flush();
    }
  });
  out.close();
  return 0;
}

#endif //RUNNER_CPP

