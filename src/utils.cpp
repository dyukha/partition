#pragma once

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

#include <ctime>
#include <cstdarg>
#include <random>
#include <future>
#include <thread>
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

std::default_random_engine generator(static_cast<size_t>(time(0)));
//std::default_random_engine generator(static_cast<size_t>(std::default_random_engine::default_seed));

struct Rand {
  static double nextRand() { return std::uniform_real_distribution<double>(0.0, 1.0)(generator); }
  static size_t next(size_t bound) {
    CHECK(bound != 0);
    if (bound == 1)
      return 0;
    return std::uniform_int_distribution<size_t>(0, static_cast<size_t>(bound - 1))(generator);
  }
  static double check(double p) {return nextRand() <= p; }
};

/**
 * Run a given function in parallel manner; a given interval [begin..end) is
 * split and the function is executed in separate threads
 */
template <class Functor>
void parallel_for(size_t numThreads, size_t begin, size_t end, Functor func) {
  CHECK(begin <= end);
  if (begin == end) return;

  auto n = end - begin;
  numThreads = std::min(numThreads, n);
  numThreads = std::min(numThreads,
                        (size_t)std::thread::hardware_concurrency());
  auto stepSize = std::max((size_t)1, (n + numThreads - 1) / numThreads);

  if (n <= stepSize || numThreads == 1) {
    for (size_t i = begin; i < end; i++) {
      func(i);
    }
    return;
  }

  auto f = [&](size_t fBegin, size_t fEnd) {
    for (size_t i = fBegin; i < fEnd; i++) {
      func(i);
    }
  };

  std::vector<std::future<void>> futures;
  size_t i = 0;
  while (i < end) {
    size_t step = std::min(static_cast<size_t>(end - i), stepSize);
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
template <class Functor>
double parallel_sum(size_t numThreads, size_t begin, size_t end, Functor func) {
  typedef double Res;
  CHECK(begin < end);

  auto n = end - begin;
  numThreads = std::min(numThreads, n);
  numThreads = std::min(numThreads,
                        (size_t)std::thread::hardware_concurrency());
  auto stepSize = std::max((size_t)1, (n + numThreads - 1) / numThreads);

  if (n <= stepSize || numThreads == 1) {
    auto res = func(begin);
    for (size_t i = begin+1; i < end; i++) {
      res = res + func(i);
    }
    return res;
  }

  auto f = [&](size_t fBegin, size_t fEnd) {
    auto res = func(fBegin);
    for (size_t i = fBegin+1; i < fEnd; i++) {
      res = res + func(i);
    }
    return res;
  };

  std::vector<std::future<Res>> futures;
  size_t i = 0;
  while (i < end) {
    size_t step = std::min(static_cast<size_t>(end - i), stepSize);
    auto next = i + step;
    futures.push_back(std::async(std::launch::async, f, i, next));
    i = next;
  }
  vector<Res> returns;
  for (auto& future : futures) {
    future.wait();
    returns.push_back(future.get());
  }
  Res res = returns[0];
  for (int i = 1; i < returns.size(); ++i) {
    res = res + returns[i];
  }
  return res;
}

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
