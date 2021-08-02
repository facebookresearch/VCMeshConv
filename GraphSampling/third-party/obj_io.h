// Copyright(C) 2018 Tommy Hinks <tommy.hinks@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#pragma once

#include <array>
#include <exception>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace thinks {

template <typename ArithT, std::size_t N>
struct ObjPosition {
  static_assert(std::is_arithmetic<ArithT>::value,
                "position values must be arithmetic");
  static_assert(N == 3 || N == 4, "position value count must be 3 or 4");

  constexpr ObjPosition() noexcept = default;

  constexpr ObjPosition(const ArithT x, const ArithT y, const ArithT z) noexcept
      : values{x, y, z} {
    static_assert(N == 3, "position value count must be 3");
  }

  constexpr ObjPosition(const ArithT x, const ArithT y, const ArithT z,
                        const ArithT w) noexcept
      : values{x, y, z, w} {
    static_assert(N == 4, "position value count must be 4");
  }

  std::array<ArithT, N> values;
};

template <typename FloatT, std::size_t N>
struct ObjTexCoord {
  static_assert(std::is_floating_point<FloatT>::value,
                "texture coordinate values must be floating point");
  static_assert(N == 2 || N == 3,
                "texture coordinate value count must be 2 or 3");

  constexpr ObjTexCoord() noexcept = default;

  constexpr ObjTexCoord(const FloatT u, const FloatT v) noexcept
      : values{u, v} {
    static_assert(N == 2, "texture coordinate value count must be 2");
  }

  constexpr ObjTexCoord(const FloatT u, const FloatT v, const FloatT w) noexcept
      : values{u, v, w} {
    static_assert(N == 3, "texture coordinate value count must be 3");
  }

  std::array<FloatT, N> values;
};

template <typename ArithT>
struct ObjNormal {
  static_assert(std::is_arithmetic<ArithT>::value,
                "normal values must be arithmetic");

  constexpr ObjNormal() noexcept = default;

  constexpr ObjNormal(const ArithT x, const ArithT y, const ArithT z) noexcept
      : values{x, y, z} {}

  std::array<ArithT, 3> values;
};

template <typename IntT>
struct ObjIndex {
  static_assert(std::is_integral<IntT>::value, "index value must be integral");

  constexpr ObjIndex() noexcept = default;

  constexpr explicit ObjIndex(const IntT idx) noexcept : value(idx) {}

  IntT value;
};

template <typename IntT>
struct ObjIndexGroup {
  constexpr ObjIndexGroup() noexcept
      : position_index{},
        tex_coord_index(ObjIndex<IntT>{}, false),
        normal_index(ObjIndex<IntT>{}, false) {}

  constexpr ObjIndexGroup(const IntT position_index_value) noexcept
      : position_index(position_index_value),
        tex_coord_index(ObjIndex<IntT>{}, false),
        normal_index(ObjIndex<IntT>{}, false) {}

  constexpr ObjIndexGroup(const IntT position_index_value,
                          const IntT tex_coord_index_value,
                          const IntT normal_index_value) noexcept
      : position_index(position_index_value),
        tex_coord_index(tex_coord_index_value, true),
        normal_index(normal_index_value, true) {}

  constexpr ObjIndexGroup(
      const IntT position_index_value,
      const std::pair<IntT, bool>& tex_coord_index_value,
      const std::pair<IntT, bool>& normal_index_value) noexcept
      : position_index(ObjIndex<IntT>(position_index_value)),
        tex_coord_index(
            std::make_pair(ObjIndex<IntT>(tex_coord_index_value.first),
                           tex_coord_index_value.second)),
        normal_index(std::make_pair(ObjIndex<IntT>(normal_index_value.first),
                                    normal_index_value.second)) {}

  // Note: Optional would have been nice instead of bool-pairs here.
  ObjIndex<IntT> position_index;
  std::pair<ObjIndex<IntT>, bool> tex_coord_index;
  std::pair<ObjIndex<IntT>, bool> normal_index;
};

namespace obj_io_internal {

template <typename T>
struct IsIndex : std::false_type {};

// Note: Not decaying the type here.
template <typename IntT>
struct IsIndex<ObjIndex<IntT>> : std::true_type {};

template <typename IntT>
struct IsIndex<ObjIndexGroup<IntT>> : std::true_type {};

}  // namespace obj_io_internal

template <typename IndexT>
struct ObjTriangleFace {
  static_assert(obj_io_internal::IsIndex<IndexT>::value,
                "face values must be of index type");

  constexpr ObjTriangleFace() noexcept = default;

  constexpr ObjTriangleFace(const IndexT i0, const IndexT i1,
                            const IndexT i2) noexcept
      : values{i0, i1, i2} {}

  std::array<IndexT, 3> values;
};

template <typename IndexT>
struct ObjQuadFace {
  static_assert(obj_io_internal::IsIndex<IndexT>::value,
                "face values must be of index type");

  constexpr ObjQuadFace() noexcept = default;

  constexpr ObjQuadFace(const IndexT i0, const IndexT i1, const IndexT i2,
                        const IndexT i3) noexcept
      : values{i0, i1, i2, i3} {}

  std::array<IndexT, 4> values;
};

template <typename IndexT>
struct ObjPolygonFace {
  static_assert(obj_io_internal::IsIndex<IndexT>::value,
                "face values must be of index type");

  constexpr ObjPolygonFace() noexcept = default;

  template <typename... Args>
  constexpr ObjPolygonFace(Args&&... args) noexcept
      : values(std::forward<Args>(args)...) {}

  std::vector<IndexT> values;
};

template <typename T>
struct ObjMapResult {
  T value;
  bool is_end;
};

template <typename T>
ObjMapResult<T> ObjMap(const T& value) noexcept {
  return {value, false};
}

template <typename T>
ObjMapResult<T> ObjEnd() noexcept {
  return {T{}, true};
}

template <typename ParseT, typename Func>
struct ObjAddFunc {
  using ParseType = ParseT;

  Func func;
};

template <typename ParseT, typename Func>
ObjAddFunc<ParseT, typename std::decay<Func>::type> MakeObjAddFunc(
    Func&& func) {
  return {std::forward<Func>(func)};
}

namespace obj_io_internal {

template <typename T>
struct IsPositionImpl : std::false_type {};

template <typename T, std::size_t N>
struct IsPositionImpl<ObjPosition<T, N>> : std::true_type {};

template <typename T>
using IsPosition = IsPositionImpl<typename std::decay<T>::type>;

template <typename T>
struct IsObjTexCoordImpl : std::false_type {};

template <typename T, std::size_t N>
struct IsObjTexCoordImpl<ObjTexCoord<T, N>> : std::true_type {};

template <typename T>
using IsObjTexCoord = IsObjTexCoordImpl<typename std::decay<T>::type>;

template <typename T>
struct IsNormalImpl : std::false_type {};

template <typename T>
struct IsNormalImpl<ObjNormal<T>> : std::true_type {};

template <typename T>
using IsNormal = IsNormalImpl<typename std::decay<T>::type>;

template <typename T>
struct IsFaceImpl : std::false_type {};

template <typename IndexT>
struct IsFaceImpl<ObjTriangleFace<IndexT>> : std::true_type {};

template <typename IndexT>
struct IsFaceImpl<ObjQuadFace<IndexT>> : std::true_type {};

template <typename IndexT>
struct IsFaceImpl<ObjPolygonFace<IndexT>> : std::true_type {};

template <typename T>
using IsFace = IsFaceImpl<typename std::decay<T>::type>;

// Face traits.
struct StaticFaceTag {};
struct DynamicFaceTag {};

template <typename T>
struct FaceTraitsImpl;  // Not implemented!

template <typename IndexT>
struct FaceTraitsImpl<ObjTriangleFace<IndexT>> {
  using FaceCategory = StaticFaceTag;
};

template <typename IndexT>
struct FaceTraitsImpl<ObjQuadFace<IndexT>> {
  using FaceCategory = StaticFaceTag;
};

template <typename IndexT>
struct FaceTraitsImpl<ObjPolygonFace<IndexT>> {
  using FaceCategory = DynamicFaceTag;
};

template <typename T>
using FaceTraits = FaceTraitsImpl<typename std::decay<T>::type>;

// Tag dispatch for optional vertex attributes, e.g. tex coords and normals.
struct FuncTag {};
struct NoOpFuncTag {};

template <typename T>
struct FuncTraits {
  using FuncCategory = FuncTag;
};

template <>
struct FuncTraits<std::nullptr_t> {
  using FuncCategory = NoOpFuncTag;
};

template <typename FloatT, std::size_t N>
void ValidateObjTexCoord(const ObjTexCoord<FloatT, N>& tex_coord) {
  using ValueType = typename decltype(tex_coord.values)::value_type;

  for (const auto v : tex_coord.values) {
    if (!(ValueType{0} <= v && v <= ValueType{1})) {
      auto oss = std::ostringstream{};
      oss << "texture coordinate values must be in range [0, 1] (found " << v
          << ")";
      throw std::runtime_error(oss.str());
    }
  }
}

template <typename FaceT>
void ValidateFace(const FaceT& face, DynamicFaceTag) {
  if (!(face.values.size() >= 3)) {
    auto oss = std::ostringstream{};
    oss << "faces must have at least 3 indices (found " << face.values.size()
        << ")";
    throw std::runtime_error(oss.str());
  }
}

// No need to validate non-polygon faces, the number
// of indices for these are enforced in the class templates.
template <typename FaceT>
void ValidateFace(const FaceT& face, StaticFaceTag) {}

constexpr inline const char* CommentPrefix() { return "#"; }
constexpr inline const char* PositionPrefix() { return "v"; }
constexpr inline const char* FacePrefix() { return "f"; }
constexpr inline const char* ObjTexCoordPrefix() { return "vt"; }
constexpr inline const char* NormalPrefix() { return "vn"; }
constexpr inline const char* IndexGroupSeparator() { return "/"; }

namespace read {

inline std::vector<std::string> Tokenize(const std::string& str,
                                         const char* const delimiters) {
  auto tokens = std::vector<std::string>{};
  auto prev = std::size_t{0};
  auto pos = std::size_t{0};
  while ((pos = str.find_first_of(delimiters, prev)) != std::string::npos) {
    if (pos == prev) {
      tokens.push_back(std::string{});
    }
    if (pos > prev) {
      tokens.push_back(str.substr(prev, pos - prev));
    }
    prev = pos + 1;  // Skip delimiter.
  }

  // Check for characters after last delimiter.
  if (prev == str.length()) {
    tokens.push_back(std::string{});
  }
  if (prev < str.length()) {
    tokens.push_back(str.substr(prev, std::string::npos));
  }

  return tokens;
}

inline std::vector<std::string> TokenizeIndexGroup(
    const std::string& index_group_str) {
  return Tokenize(index_group_str, IndexGroupSeparator());
}

template <typename T>
bool ParseValue(std::istream* const is, T* const value) {
  if (*is >> *value || !is->eof()) {
    if (is->fail()) {
      is->clear();  // Clear status bits.
      auto dummy = std::string{};
      *is >> dummy;
      auto oss = std::ostringstream{};
      oss << "failed parsing '" << dummy << "'";
      throw std::runtime_error(oss.str());
    }
    return true;
  }
  return false;
}

template <typename IntT>
std::istream& operator>>(std::istream& is, ObjIndex<IntT>& index) {
  if (ParseValue(&is, &index.value)) {
    // Check for underflow.
    if (!(index.value > 0)) {
      throw std::runtime_error("parsed index must be greater than zero");
    }

    // Convert to zero-based index.
    --index.value;
  }

  return is;
}

template <typename IntT>
std::istream& operator>>(std::istream& is, ObjIndexGroup<IntT>& index_group) {
  // Read index group as the string leading up to the
  // following whitespace/newline.
  auto index_group_str = std::string{};
  if (!ParseValue(&is, &index_group_str)) {
    return is;
  }

  const auto tokens = TokenizeIndexGroup(index_group_str);
  if (tokens.empty()) {
    throw std::runtime_error("empty index group tokens");
  }

  if (tokens.size() > 3) {
    auto oss = std::stringstream{};
    oss << "index group can have at most 3 tokens ('" << index_group_str
        << "')";
    throw std::runtime_error(oss.str());
  }

  // ObjPosition index.
  if (tokens[0].empty()) {
    auto oss = std::stringstream{};
    oss << "empty position index ('" << index_group_str << "')";
    throw std::runtime_error(oss.str());
  }
  auto iss_pos = std::istringstream(tokens[0]);
  iss_pos >> index_group.position_index;

  // Texture coordinate index, may be empty.
  if (tokens.size() > 1 && !tokens[1].empty()) {
    auto iss_tex = std::istringstream(tokens[1]);
    iss_tex >> index_group.tex_coord_index.first;
    index_group.tex_coord_index.second = true;
  }

  // ObjNormal index.
  if (tokens.size() > 2) {
    if (tokens[2].empty()) {
      auto oss = std::stringstream{};
      oss << "empty normal index ('" << index_group_str << "')";
      throw std::runtime_error(oss.str());
    }
    auto iss_nml = std::istringstream(tokens[2]);
    iss_nml >> index_group.normal_index.first;
    index_group.normal_index.second = true;
  }

  return is;
}

template <typename T, std::size_t N>
std::uint32_t ParseValues(std::istringstream* const iss,
                          std::array<T, N>* const values) {
  using ContainerType = typename std::remove_pointer<decltype(values)>::type;
  using ValueType = typename ContainerType::value_type;

  constexpr auto kValueCount = std::tuple_size<ContainerType>::value;
  static_assert(kValueCount > 0, "empty array");

  auto parse_count = std::uint32_t{0};
  auto value = ValueType{};
  while (ParseValue(iss, &value)) {
    if (parse_count >= kValueCount) {
      auto oss = std::ostringstream{};
      oss << "expected to parse at most " << kValueCount << " values";
      throw std::runtime_error(oss.str());
    }
    (*values)[parse_count++] = value;
  }

  return parse_count;
}

template <typename T>
std::uint32_t ParseValues(std::istringstream* const iss,
                          std::vector<T>* const values) {
  using ContainerType = typename std::remove_pointer<decltype(values)>::type;
  using ValueType = typename ContainerType::value_type;

  auto value = ValueType{};
  while (ParseValue(iss, &value)) {
    values->push_back(value);
  }

  return static_cast<std::uint32_t>(values->size());
}

template <typename AddPositionFuncT>
void ParsePosition(std::istringstream* const iss, AddPositionFuncT&& add_position,
                   std::uint32_t* const count) {
  using ParseType = typename std::decay<AddPositionFuncT>::type::ParseType;
  static_assert(IsPosition<ParseType>::value,
                "parse type must be a ObjPosition type");

  auto position = ParseType{};
  const auto parse_count = ParseValues(iss, &position.values);

  if (parse_count < 3) {
    auto oss = std::ostringstream{};
    oss << "positions must have 3 or 4 values (found " << parse_count << ")";
    throw std::runtime_error(oss.str());
  }

  // Fourth position value (if any) defaults to 1.
  using ArrayType = decltype(position.values);
  if (std::tuple_size<ArrayType>::value == 4 && parse_count == 3) {
    position.values[3] = typename ArrayType::value_type{1};
  }

  add_position.func(position);
  ++(*count);
}

template <typename AddFaceFuncT>
void ParseFace(std::istringstream* const iss, 
               AddFaceFuncT&& add_face,
               std::uint32_t* const count) {
  using ParseType = typename std::decay<AddFaceFuncT>::type::ParseType;
  static_assert(IsFace<ParseType>::value, "parse type must be a Face type");

  auto face = ParseType{};
  const auto parse_count = ParseValues(iss, &face.values);

  // Works for both std::array and std::vector.
  // This is never an issue for polygons.
  if (parse_count != face.values.size()) {
    auto oss = std::ostringstream{};
    oss << "expected " << face.values.size() << " face indices (found "
        << parse_count << ")";
    throw std::runtime_error(oss.str());
  }

  ValidateFace(face, typename FaceTraits<ParseType>::FaceCategory{});
  add_face.func(face);
  ++(*count);
}

template <typename AddObjTexCoordFuncT>
void ParseObjTexCoord(std::istringstream* const iss,
                   AddObjTexCoordFuncT&& add_tex_coord, 
                   std::uint32_t* const count,
                   FuncTag) {
  using ParseType = typename std::decay<AddObjTexCoordFuncT>::type::ParseType;
  static_assert(IsObjTexCoord<ParseType>::value,
                "parse type must be a ObjTexCoord type");

  auto tex_coord = ParseType{};
  const auto parse_count = ParseValues(iss, &tex_coord.values);

  if (parse_count < 2) {
    auto oss = std::ostringstream{};
    oss << "texture coordinates must have 2 or 3 values (found " << parse_count
        << ")";
    throw std::runtime_error(oss.str());
  }

  // Third texture coordinate value (if any) defaults to 1.
  using ArrayType = decltype(tex_coord.values);
  if (std::tuple_size<ArrayType>::value == 3 && parse_count == 2) {
    tex_coord.values[2] = typename ArrayType::value_type{1};
  }

  ValidateObjTexCoord(tex_coord);
  add_tex_coord.func(tex_coord);
  ++(*count);
}

// Dummy.
template <typename AddObjTexCoordFuncT>
void ParseObjTexCoord(std::istringstream* const, AddObjTexCoordFuncT&&,
                   std::uint32_t* const, NoOpFuncTag) {}

template <typename AddNormalFuncT>
void ParseNormal(std::istringstream* const iss, 
                 AddNormalFuncT&& add_normal,
                 std::uint32_t* const count, 
                 FuncTag) {
  using ParseType = typename std::decay<AddNormalFuncT>::type::ParseType;
  static_assert(IsNormal<ParseType>::value, "parse type must be a ObjNormal type");

  auto normal = ParseType{};
  const auto parse_count = ParseValues(iss, &normal.values);

  if (parse_count < 3) {
    auto oss = std::ostringstream{};
    oss << "normals must have 3 values (found " << parse_count << ")";
    throw std::runtime_error(oss.str());
  }

  add_normal.func(normal);
  ++(*count);
}

// Dummy.
template <typename AddNormalFuncT>
void ParseNormal(std::istringstream* const, AddNormalFuncT&&,
                 std::uint32_t* const, NoOpFuncTag) {}

template <typename AddPositionFuncT, typename AddObjTexCoordFuncT,
          typename AddNormalFuncT, typename AddFaceFuncT>
void ParseLine(const std::string& line, 
               AddPositionFuncT&& add_position,
               AddFaceFuncT&& add_face, 
               AddObjTexCoordFuncT&& add_tex_coord,
               AddNormalFuncT&& add_normal, 
               std::uint32_t* const position_count,
               std::uint32_t* const face_count,
               std::uint32_t* const tex_coord_count,
               std::uint32_t* const normal_count) {
  auto iss = std::istringstream(line);

  // Prefix is first non-whitespace token.
  auto prefix = std::string{};
  iss >> prefix;

  // Parse the rest of the line depending on prefix.
  if (prefix.empty() || prefix == CommentPrefix()) {
    return;  // Ignore empty lines and comments.
  } else if (prefix == PositionPrefix()) {
    ParsePosition(&iss, std::forward<AddPositionFuncT>(add_position),
                  position_count);
  } else if (prefix == FacePrefix()) {
    ParseFace(&iss, std::forward<AddFaceFuncT>(add_face), face_count);
  } else if (prefix == ObjTexCoordPrefix()) {
    ParseObjTexCoord(&iss, std::forward<AddObjTexCoordFuncT>(add_tex_coord),
                     tex_coord_count,
                     typename FuncTraits<AddObjTexCoordFuncT>::FuncCategory{});
  } else if (prefix == NormalPrefix()) {
    ParseNormal(&iss, std::forward<AddNormalFuncT>(add_normal), normal_count,
                typename FuncTraits<AddNormalFuncT>::FuncCategory{});
  } else {
    auto oss = std::ostringstream{};
    oss << "unrecognized line prefix '" << prefix << "'";
    throw std::runtime_error(oss.str());
  }
}

template <typename AddPositionFuncT, typename AddObjTexCoordFuncT,
          typename AddNormalFuncT, typename AddFaceFuncT>
void ParseLines(std::istream& is, 
                AddPositionFuncT&& add_position,
                AddFaceFuncT&& add_face, 
                AddObjTexCoordFuncT&& add_tex_coord,
                AddNormalFuncT&& add_normal,
                std::uint32_t* const position_count,
                std::uint32_t* const face_count,
                std::uint32_t* const tex_coord_count,
                std::uint32_t* const normal_count) {
  auto line = std::string{};
  while (std::getline(is, line)) {
    obj_io_internal::read::ParseLine(
        line, 
        std::forward<AddPositionFuncT>(add_position),
        std::forward<AddFaceFuncT>(add_face),
        std::forward<AddObjTexCoordFuncT>(add_tex_coord),
        std::forward<AddNormalFuncT>(add_normal), 
        position_count, face_count,
        tex_coord_count, normal_count);
  }
}

}  // namespace read

namespace write {

template <typename IntT>
std::ostream& operator<<(std::ostream& os, const ObjIndex<IntT>& index) {
  using ValueType = decltype(index.value);

  // Note that the valid range allows increment of one.
  if (!(ValueType{0} <= index.value &&
        index.value < std::numeric_limits<ValueType>::max())) {
    auto oss = std::ostringstream{};
    oss << "invalid index: " << static_cast<std::int64_t>(index.value);
    throw std::runtime_error(oss.str());
  }

  // Input indices are assumed to be zero-based.
  // OBJ format uses one-based indexing.
  os << index.value + 1;
  return os;
}

template <typename IntT>
std::ostream& operator<<(std::ostream& os,
                         const ObjIndexGroup<IntT>& index_group) {
  os << index_group.position_index;
  if (index_group.tex_coord_index.second && index_group.normal_index.second) {
    os << IndexGroupSeparator() << index_group.tex_coord_index.first
       << IndexGroupSeparator() << index_group.normal_index.first;
  } else if (index_group.tex_coord_index.second) {
    os << IndexGroupSeparator() << index_group.tex_coord_index.first;
  } else if (index_group.normal_index.second) {
    os << IndexGroupSeparator() << IndexGroupSeparator()
       << index_group.normal_index.first;
  }
  return os;
}

inline void WriteHeader(std::ostream& os, const std::string& newline) {
  os << CommentPrefix() << " Written by https://github.com/thinks/obj-io"
     << newline;
}

template <template <typename> class MappedTypeCheckerT, typename MapperT,
          typename ValidatorT>
std::uint32_t WriteMappedLines(std::ostream& os, const std::string& line_prefix,
                               MapperT&& mapper, ValidatorT validator,
                               const std::string& newline) {
  auto count = std::uint32_t{0};
  auto map_result = mapper();
  while (!map_result.is_end) {
    static_assert(MappedTypeCheckerT<decltype(map_result.value)>::value,
                  "incorrect mapped type");

    validator(map_result.value);

    // Write line.
    os << line_prefix;
    for (const auto& element : map_result.value.values) {
      os << " " << element;
    }
    os << newline;

    ++count;
    map_result = mapper();
  }
  return count;
}

template <typename MapperT>
std::uint32_t WritePositions(std::ostream& os, MapperT&& mapper,
                             const std::string& newline) {
  return WriteMappedLines<IsPosition>(os, PositionPrefix(),
                                      std::forward<MapperT>(mapper),
                                      [](const auto&) {},  // No validation.
                                      newline);
}

template <typename MapperT>
std::uint32_t WriteObjTexCoords(std::ostream& os, MapperT&& mapper,
                                const std::string& newline, FuncTag) {
  return WriteMappedLines<IsObjTexCoord>(
      os, ObjTexCoordPrefix(), std::forward<MapperT>(mapper),
      [](const auto& tex_coord) { ValidateObjTexCoord(tex_coord); }, newline);
}

// Dummy.
template <typename MapperT>
std::uint32_t WriteObjTexCoords(std::ostream&, MapperT&&, const std::string&,
                                NoOpFuncTag) {
  return 0;
}

template <typename MapperT>
std::uint32_t WriteNormals(std::ostream& os, MapperT&& mapper,
                           const std::string& newline, FuncTag) {
  return WriteMappedLines<IsNormal>(os, NormalPrefix(),
                                    std::forward<MapperT>(mapper),
                                    [](const auto&) {},  // No validation.
                                    newline);
}

// Dummy.
template <typename MapperT>
std::uint32_t WriteNormals(std::ostream&, MapperT&&, const std::string&,
                           NoOpFuncTag) {
  return 0;
}

template <typename MapperT>
std::uint32_t WriteFaces(std::ostream& os, MapperT&& mapper,
                         const std::string& newline) {
  return WriteMappedLines<IsFace>(
      os, FacePrefix(), std::forward<MapperT>(mapper),
      [](const auto& face) {
        ValidateFace(face, typename FaceTraits<decltype(face)>::FaceCategory{});
      },
      newline);
}

}  // namespace write
}  // namespace obj_io_internal

struct ObjReadResult {
  std::uint32_t position_count;
  std::uint32_t face_count;
  std::uint32_t tex_coord_count;
  std::uint32_t normal_count;
};

template <typename AddPositionFuncT, typename AddFaceFuncT,
          typename AddObjTexCoordFuncT = std::nullptr_t,
          typename AddNormalFuncT = std::nullptr_t>
ObjReadResult ReadObj(std::istream& is, 
                      AddPositionFuncT&& add_position,
                      AddFaceFuncT&& add_face,
                      AddObjTexCoordFuncT&& add_tex_coord = nullptr,
                      AddNormalFuncT&& add_normal = nullptr) {
  ObjReadResult result = {};
  obj_io_internal::read::ParseLines(
      is, std::forward<AddPositionFuncT>(add_position),
      std::forward<AddFaceFuncT>(add_face),
      std::forward<AddObjTexCoordFuncT>(add_tex_coord),
      std::forward<AddNormalFuncT>(add_normal), &result.position_count,
      &result.face_count, &result.tex_coord_count, &result.normal_count);
  return result;
}

struct ObjWriteResult {
  std::uint32_t position_count;
  std::uint32_t face_count;
  std::uint32_t tex_coord_count;
  std::uint32_t normal_count;
};

template <typename PositionMapperT, typename FaceMapperT,
          typename ObjTexCoordMapperT = std::nullptr_t,
          typename NormalMapperT = std::nullptr_t>
ObjWriteResult WriteObj(std::ostream& os, 
                        PositionMapperT&& position_mapper,
                        FaceMapperT&& face_mapper,
                        ObjTexCoordMapperT&& tex_coord_mapper = nullptr,
                        NormalMapperT&& normal_mapper = nullptr,
                        const std::string& newline = "\n") {
  ObjWriteResult result = {};
  obj_io_internal::write::WriteHeader(os, newline);
  result.position_count += obj_io_internal::write::WritePositions(
      os, std::forward<PositionMapperT>(position_mapper), newline);
  result.tex_coord_count += obj_io_internal::write::WriteObjTexCoords(
      os, std::forward<ObjTexCoordMapperT>(tex_coord_mapper), newline,
      typename obj_io_internal::FuncTraits<ObjTexCoordMapperT>::FuncCategory{});
  result.normal_count += obj_io_internal::write::WriteNormals(
      os, std::forward<NormalMapperT>(normal_mapper), newline,
      typename obj_io_internal::FuncTraits<NormalMapperT>::FuncCategory{});
  result.face_count += obj_io_internal::write::WriteFaces(
      os, std::forward<FaceMapperT>(face_mapper), newline);
  return result;
}

}  // namespace thinks
