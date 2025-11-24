---
title: C++ 技术栈
date: 2025-11-24 12:44:30
tags:
---

## 手撕Vector

```C++
#include <bits/stdc++.h>
using namespace std;
#define int long long
template <typename T>
class Vector {
  T* _data;
  size_t _size;
  size_t _capacity;
  void expand() {
    size_t new_capacity = (_capacity == 0) ? 1 : _capacity * 2;
    T* new_data = new T[new_capacity];
    for (size_t i = 0; i < _size; i++) new_data[i] = _data[i];
    if (_data) delete[] _data;
    _data = new_data;
    _capacity = new_capacity;
    cout << "容量拓展到" << _capacity << '\n';
  }
 public:
  Vector() : _data(nullptr), _size(0), _capacity(0) {}
  ~Vector() {
    if (_data) {
      delete[] _data;
      _data = nullptr;
    }
  }
  void push_back(const T& value) {
    if (_size == _capacity) {
      expand();
    }
    _data[_size] = value;
    _size++;
  }
  T& operator[](size_t index) { return _data[index]; }
  size_t size() const { return _size; }
  size_t capacity() const { return _capacity; }
};

void solve() {
  Vector<int> v;
  for (int i = 0; i < 6; i++) {
    v.push_back(i);
    cout << "插入" << i << ",size:" << v.size() << "\n";
  }
}
signed main() {
  solve();
  return 0;
}
```