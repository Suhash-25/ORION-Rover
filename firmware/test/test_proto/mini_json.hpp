// Minimal JSON reader for the protocol golden vectors (test-only; no dependency to install on
// the host, in CI or in PlatformIO). Supports objects, arrays, strings without \u escapes,
// numbers, true/false/null. Throws std::runtime_error on malformed input.
#ifndef MINI_JSON_HPP_
#define MINI_JSON_HPP_

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_json
{

struct Value
{
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<Value> array;
  std::map<std::string, Value> object;

  const Value & operator[](const std::string & key) const
  {
    auto it = object.find(key);
    if (kind != Kind::kObject || it == object.end()) {
      throw std::runtime_error("missing key: " + key);
    }
    return it->second;
  }
  const Value & operator[](size_t i) const {return array.at(i);}
  bool has(const std::string & key) const {return object.count(key) != 0;}
  long long as_int() const {return static_cast<long long>(number);}
};

class Parser
{
public:
  explicit Parser(const std::string & text)
  : s_(text) {}

  Value parse()
  {
    Value v = value();
    ws();
    if (i_ != s_.size()) {
      fail("trailing characters");
    }
    return v;
  }

private:
  [[noreturn]] void fail(const std::string & what)
  {
    throw std::runtime_error("json: " + what + " at offset " + std::to_string(i_));
  }
  void ws()
  {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      ++i_;
    }
  }
  char peek()
  {
    ws();
    if (i_ >= s_.size()) {
      fail("unexpected end");
    }
    return s_[i_];
  }
  void expect(char c)
  {
    if (peek() != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++i_;
  }
  bool literal(const char * word)
  {
    const std::string w(word);
    if (s_.compare(i_, w.size(), w) == 0) {
      i_ += w.size();
      return true;
    }
    return false;
  }
  std::string str()
  {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) {
          fail("bad escape");
        }
        const char e = s_[i_++];
        c = e == 'n' ? '\n' : e == 't' ? '\t' : e;
        if (e == 'u') {
          fail("\\u escapes not supported");
        }
      }
      out.push_back(c);
    }
    expect('"');
    return out;
  }
  Value value()
  {
    Value v;
    const char c = peek();
    if (c == '{') {
      v.kind = Value::Kind::kObject;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      while (true) {
        const std::string key = str();
        expect(':');
        v.object[key] = value();
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.kind = Value::Kind::kArray;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      while (true) {
        v.array.push_back(value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.kind = Value::Kind::kString;
      v.string = str();
      return v;
    }
    if (literal("true")) {
      v.kind = Value::Kind::kBool;
      v.boolean = true;
      return v;
    }
    if (literal("false")) {
      v.kind = Value::Kind::kBool;
      return v;
    }
    if (literal("null")) {
      return v;
    }
    const char * begin = s_.c_str() + i_;
    char * end = nullptr;
    v.number = std::strtod(begin, &end);
    if (end == begin) {
      fail("bad value");
    }
    v.kind = Value::Kind::kNumber;
    i_ += static_cast<size_t>(end - begin);
    return v;
  }

  const std::string & s_;
  size_t i_ = 0;
};

inline Value parse_file(const std::string & path)
{
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  return Parser(text).parse();
}

/// "0a1b" -> {0x0a, 0x1b}
inline std::vector<uint8_t> from_hex(const std::string & hex)
{
  if (hex.size() % 2 != 0) {
    throw std::runtime_error("odd hex length");
  }
  std::vector<uint8_t> out;
  for (size_t i = 0; i < hex.size(); i += 2) {
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

}  // namespace mini_json

#endif  // MINI_JSON_HPP_
