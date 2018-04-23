#ifndef COMMON_CPP
#define COMMON_CPP

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

#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <iterator>
#include <cmath>
#include <numeric>

#include <chrono>
#if defined(_WIN32)
  #include <direct.h>
#else
  #include <sys/stat.h>
#endif


using namespace std;

std::default_random_engine generator(static_cast<unsigned long>(time(0)));

struct Rand {
  static double nextRand() { return std::uniform_real_distribution<double>(0.0,1.0)(generator); }
  static unsigned int next(unsigned long bound) {
    assert(bound != 0);
    if (bound == 1)
      return 0;
    return std::uniform_int_distribution<unsigned int>(0, static_cast<unsigned int>(bound - 1))(generator);
  }
  static double check(double p) {return nextRand() <= p; }
};

/// If the element present in map, then return it key.
/// Otherwise, add it into the map with key=size(map) (before adding). Return the key.
inline static int getOrAdd(const string& u, unordered_map<string, int>& map) {
  auto p = map.find(u);
  if (p == map.end()) {
    auto size = static_cast<int>(map.size());
    map[u] = size;
    return size;
  } else {
    return p->second;
  }
}

inline bool fileExist(const std::string &name) {
  ifstream in(name);
  return in.good();
}

void measureTime(const string& message, const std::function<void()> &f) {
  auto startTime = std::chrono::system_clock::now();
  f();
  cerr << message << " time: " << (std::chrono::duration<double>(std::chrono::system_clock::now() - startTime)).count() << endl;
}

void createDir(const string& dir) {
  #if defined(_WIN32)
  _mkdir(dir.c_str());
  #else
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif
}

#endif