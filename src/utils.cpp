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
#include <cstdarg>
#include <string>
#include <iterator>
#include <cmath>
#include <numeric>
#include <future>
#include <thread>
#include <sstream>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <cstring>


#include <chrono>
#if defined(_WIN32)
  #include <direct.h>
#else
  #include <sys/stat.h>
#endif


const int ASSERT_EXIT_CODE = -400;
const int CHECK_EXIT_CODE = 10;

#define stringize(s) #s
#define XSTR(s) stringize(s)
#define CHECK1(condition) \
if (0 == (condition)) { \
  std::cerr << "assertion '" << XSTR(condition) << "' failed [" << __FILE__ << ":" << __LINE__ << "]\n"; \
  throw ASSERT_EXIT_CODE; \
}
#define CHECK2(condition, message) \
  CHECK3(condition, message, CHECK_EXIT_CODE)
#define CHECK3(condition, message, exitCode) \
if (0 == (condition)) { \
  std::cerr << "\033[91m" << message << "\033[0m" << " [" << __FILE__ << ":" << __LINE__ << "]\n"; \
  throw exitCode; \
}
#define GET_MACRO(_1,_2,_3,NAME,...) NAME
#define CHECK(...) GET_MACRO(__VA_ARGS__, CHECK3, CHECK2, CHECK1)(__VA_ARGS__)

#define ERROR(message) \
{ \
  std::cerr << "\033[91m" << message << "\033[0m" << " [" << __FILE__ << ":" << __LINE__ << "]\n"; \
  throw ASSERT_EXIT_CODE; \
}

typedef long long int int64;

inline void LOG(const char* message, va_list args) {
  auto end = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(end);
  auto stime = std::string(std::ctime(&time));
  stime.erase(stime.find_last_not_of(" \n\r\t") + 1);
  stime = "\033[90m" + stime + ":\033[0m";

  char* buffer = new char[1024];

  std::vsprintf(buffer, message, args);

  std::cerr << stime << " " << std::string(buffer) << "\n";
  delete[] buffer;
}

inline void LOG_IF(bool condition, const char* message, ...) {
  if (!condition) return;

  va_list args;
  va_start(args, message);
  LOG(message, args);
  va_end(args);
}

inline void LOG(const char* message, ...) {
  va_list args;
  va_start(args, message);
  LOG(message, args);
  va_end(args);
}

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
  static bool check(double p) {return nextRand() <= p; }
};

/// If the element present in map, then return it key.
/// Otherwise, add it into the map with key=size(map) (before adding). Return the key.
inline static int getOrAdd(const int64& u, unordered_map<int64, int>& map) {
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

inline int sign(double x) {
  return x < -1e-9 ? -1 : (x > 1e-9 ? 1 : 0);
}


size_t numThreads = (size_t)std::thread::hardware_concurrency();;

/**
 * Run a given function in parallel manner; a given interval [begin..end) is
 * split and the function is executed in separate threads
 */
template <class T, class Functor>
void parallel_for(vector<T>& vector, Functor func) {
  assert(vector.size() > 0);
  auto n = vector.size();
  auto stepSize = std::max((size_t)1, (n + numThreads - 1) / numThreads);

  if (n <= stepSize || numThreads == 1) {
    for (size_t i = 0; i < vector.size(); i++) {
      func(vector[i]);
    }
    return;
  }

  auto f = [&](size_t fBegin, size_t fEnd) {
    for (size_t i = fBegin; i < fEnd; i++) {
      func(vector[i]);
    }
  };

  std::vector<std::future<void>> futures;
  size_t i = 0;
  while (i < vector.size()) {
    size_t step = std::min(static_cast<size_t>(vector.size() - i), stepSize);
    auto next = i + step;
    futures.push_back(std::async(std::launch::async, f, i, next));
    i = next;
  }
  for (auto& future : futures) {
    future.wait();
  }
}

/**
 * Run a given function in parallel manner; a given interval [begin..end) is
 * split and the function is executed in separate threads
 */
template <class Res, class T, class Functor, class Merge>
Res parallel_reduce(vector<T>& vector, Functor func, Merge merge) {
  assert(vector.size() > 0);
  auto n = vector.size();
  auto stepSize = std::max((size_t)1, (n + numThreads - 1) / numThreads);

  if (n <= stepSize || numThreads == 1) {
    Res res = func(vector[0]);
    for (size_t i = 1; i < vector.size(); i++) {
      res = merge(res, func(vector[i]));
    }
    return res;
  }

  auto f = [&](size_t fBegin, size_t fEnd) {
    Res res = func(vector[fBegin]);
    for (size_t i = fBegin + 1; i < fEnd; i++) {
      res = merge(res, func(vector[i]));
    }
    return res;
  };

  std::vector<std::future<Res>> futures;
  size_t i = 0;
  while (i < vector.size()) {
    size_t step = std::min(static_cast<size_t>(vector.size() - i), stepSize);
    auto next = i + step;
    futures.push_back(std::async(std::launch::async, f, i, next));
    i = next;
  }
  for (auto& future : futures) {
    future.wait();
  }
  Res res = futures[0].get();
  for (size_t i = 1; i < futures.size(); ++i) {
    res = merge(res, futures[i].get());
  }
  return res;
}

template <class T, class Functor>
double parallel_sum(vector<T>& vector, Functor func) {
  return parallel_reduce<double>(vector, func, [](double a, double b) {return a + b;});
};

template <class T, class Functor>
tuple<double, double> parallel_sum_tuple(vector<T>& vector, Functor func) {
  return parallel_reduce< tuple<double, double> >(vector, func, [](const tuple<double, double>& a, const tuple<double, double>& b) {
    return make_tuple(get<0>(a) + get<0>(b), get<1>(a) + get<1>(b));
  });
};

template <class T, class Functor>
double parallel_sum_not_fixed(vector<T>& vector, Functor func) {
  return parallel_sum<double>(vector, [&](T v) {return v.fixed ? 0. : func(v);});
};

template <class T, class Functor>
double parallel_sum_tuple_not_fixed(vector<T>& vector, Functor func) {
  return parallel_sum<tuple<double, double>>(vector, [&](T v) {return v.fixed ? make_pair(0., 0.) : func(v);});
};

template <typename T>
std::string my_to_string(T const& value) {
  stringstream sstr;
  sstr << value;
  return sstr.str();
}

template<typename T> T getMax(const vector<T>& vector) {
  return *max_element(begin(vector), end(vector));
}

template<typename T> T getMin(const vector<T>& vector) {
  return *min_element(begin(vector), end(vector));
}

struct sysinfo memInfo;

int maxMemory = 0;

int parseLineMem(char* line){
  // This assumes that a digit will be found and the line ends in " Kb".
  int i = strlen(line);
  const char* p = line;
  while (*p <'0' || *p > '9') p++;
  line[i-3] = '\0';
  i = atoi(p);
  return i;
}

int getValueMem(){ //Note: this value is in KB!
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != NULL){
    if (strncmp(line, "VmRSS:", 6) == 0){
      result = parseLineMem(line);
      break;
    }
  }
  fclose(file);
  return result;
}

void updateMem() {
  maxMemory = max(maxMemory, getValueMem());
}

void resetMem() {
  maxMemory = 0;
}
#endif
