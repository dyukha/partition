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
#include "GradientDescentManyClusters.cpp"
#include "GradientDescent.cpp"
#include "GradientDescentImpl.cpp"
#include "RecursiveClustering.cpp"
#include "Projections.cpp"

bool was[1000000];

ofstream out;

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
void runMany(int solutionNumber, const string& fileName, const string& name, const Graph& g, const function<double (const Partition&)>& objective, const function<Partition ()>& solve) {
  measureTime(fileName.substr(fileName.find("/") + 1), [&]() {
    out << name << ": ";
    string outs(g.n, ' ');
    ofstream partitionFile;
    partitionFile.open(fileName + "_partitions.txt", ofstream::out);
    vector<pair<double, Partition> > partitions;
    for (int i = 0; i < solutionNumber; i++) {
      measureTime(to_string(i) + "-th iteration", [&]() {
        Partition partition = solve();
        // Enforce balance
        partition.fixPartition();
        assert(partition.check());
#pragma omp critical
        {
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
        }
      });
    }
    partitionFile.close();
    int mid = solutionNumber / 2;
    nth_element(partitions.begin(), partitions.begin() + mid, partitions.end(),
                [&](const pair<double, Partition> &a, const pair<double, Partition> &b) { return a.first < b.first; });
    cerr << "result = ";
    printRes(partitions[mid].second, g, partitions[mid].first, cerr);
    out << "| ";
    printRes(partitions[mid].second, g, partitions[mid].first, out);
  });
  cerr << endl;
}

std::function<double (const Partition&)> cut(const Graph& g) {
  return [&g] (const Partition& p) { return p.cut(g); };
}


void gradientDescent(const Graph &g, double eps, int solutionNumber, double step, const string& fileName) {
  runMany(solutionNumber, fileName, "Grad", g, cut(g), [&] {
    return GradientDescentImpl(step, Projections::presize).apply(g, eps, 0.5);
  });
}

void gradientDescentManyPartsSimultanious(const Graph &g, double eps, int solutionNumber, double step, int k, const string& fileName) {
  runMany(solutionNumber, fileName, "Grad", g, cut(g), [&] {
    return GradientDescentManyClusters(step).apply(g, eps, k);
  });
}

void gradientDescentManyParts(const Graph &g, double eps, int solutionNumber, double step, int k, const string& fileName) {
  runMany(solutionNumber, fileName, "Grad", g, cut(g), [&] {
    vector<tuple<int, double> > cuts;
    auto res = RecursiveClustering::apply(g, eps / 4.5, step, k, cuts);
    get<0>(cuts[0])++;
    for (int i = 0; i < (int)cuts.size() - 1; ++i) {
      get<0>(cuts[i+1]) += get<0>(cuts[i]);
      get<1>(cuts[i+1]) += get<1>(cuts[i]);
    }
    cerr << "Cuts for the corresponding cluster count: ";
    double edgeCount = 0;
    for (int i = 0; i < g.n; ++i) {
      edgeCount += g.g[i].size();
    }
    edgeCount /= 2;
    for (auto cut : cuts)
      cerr << "[" << get<0>(cut) << ": " << get<1>(cut) << " (" << (int)(10000. * get<1>(cut) / edgeCount) / 100. << "%)] ";
    cerr << endl;
    return res;
  });
}

/*
void gradientDescentMemAggressive(const Graph &g, int innerIterNumber, double eps, int solutionNumber, const string& fileName) {
  runMany(solutionNumber, fileName, "GradMem", g, cut(g), [&] {
    return GradientDescent::applyMemAggressive(g, innerIterNumber, eps, 1, fileName);
  });
}
*/

void degreeDistribution(Graph& g) {
  vector<long> dist;
  for (int i = 0; i < g.n; i++) {
    dist.push_back(g.g[i].size());
  }
  sort(dist.begin(), dist.end());
  int cnt = 8;
  for (int i = 0; i < cnt; i++)
    out << dist[(i+1) * g.n / cnt - 1] << " ";
  int sum = 0;
  for (int i = 0; i < g.n; i++)
    sum += dist[i];
  out << " | avg = " << sum / g.n;
  out << endl;
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
  vector<string> snapFiles = {"facebook", "wikiVote", "twitter", "gplus", "LiveJournal", "orkut"};
  vector<string> bigFiles = {"p_500000_10", "p_500000_20", "p_500000_50", "p_500000_100", "p500k", "p5M"};

//  for (auto& f : darwiniFiles) files.push_back(f);
  for (auto &f : snapFiles) files.push_back(f);
//  for (auto& f : bigFiles) files.push_back(f);

//  vector<string> darwiniLargeFiles = {"Darwini_10M", "Darwini_50M"};
//  for (auto& f : darwiniLargeFiles) files.push_back(f);

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
      out << name << ": ";
      cerr << name << ": " << endl;
      Graph g = Graph::read(getPath(name));
      string fileName = dir + "/" + name;
//      gradientDescent(g, 0.01, 3, 0.0005, fileName);
//      gradientDescentManyParts(g, 0.01, 3, 0.0003, 20, fileName);
      gradientDescentManyPartsSimultanious(g, 0.03, 3, 0.0003, 20, fileName);
      out.flush();
    }
  });
  out.close();
  return 0;
}

#endif //RUNNER_CPP

