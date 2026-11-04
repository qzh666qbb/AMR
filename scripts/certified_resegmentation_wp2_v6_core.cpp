#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kMissingState = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kMaximumFeatureRecords = 2000000;
constexpr uint64_t kMaximumPairScores = 50000000;
constexpr uint32_t kSimMarkDimension = 768;
using Count = unsigned __int128;

void checked_add(Count& destination, Count value) {
  const Count maximum = static_cast<Count>(-1);
  if (maximum - destination < value) {
    throw std::overflow_error("threat-set count exceeds 128 bits");
  }
  destination += value;
}

struct FeatureTable {
  uint32_t mode = 0;
  uint32_t dimension = 0;
  std::unordered_map<uint64_t, uint32_t> index_by_span;
  std::vector<uint64_t> span_codes;
  std::vector<uint8_t> allowed;
  std::vector<uint8_t> regions;
  std::array<uint8_t, 64> relation{};
  std::vector<uint64_t> embedding_offsets;
  std::vector<float> embeddings;
  std::vector<double> embedding_norms;
};

struct DumpRequest {
  std::string path;
  uint32_t mode;
};

struct Problem {
  uint32_t n;
  uint32_t m;
  uint32_t k;
  uint32_t score_mode;
  uint32_t span_mode;
  double normalization;
  double bias;
  double threshold;
  std::vector<uint32_t> original;
  std::string span_bits;
  std::unordered_map<uint64_t, double> external_scores;
  FeatureTable features;
};

struct Slots {
  std::array<uint32_t, 4> state;

  Slots() { state.fill(kMissingState); }
};

struct Layer {
  std::unordered_map<uint64_t, Slots> groups;
  std::vector<double> best;
  std::vector<Count> counts;
  std::vector<uint32_t> path_rank;
  std::vector<uint64_t> pair_code;
  std::vector<uint8_t> moves;
};

struct Result {
  double original_score;
  double certified_score;
  Count threat_set_size;
  std::array<Count, 4> histogram;
  std::vector<uint32_t> witness;
  uint64_t reachable_states;
  uint64_t transitions;
  uint64_t peak_layer_states;
  uint64_t candidate_positions;
  uint64_t unique_spans;
  uint64_t unique_pairs;
};

uint64_t pair_code(uint32_t left, uint32_t middle, uint64_t base) {
  return static_cast<uint64_t>(left) * base + middle;
}

uint64_t triple_code(
    uint32_t left,
    uint32_t middle,
    uint32_t right,
    uint64_t base) {
  return (static_cast<uint64_t>(left) * base + middle) * base + right;
}

void read_exact(
    std::ifstream& input,
    char* destination,
    std::streamsize size) {
  if (!input.read(destination, size)) {
    throw std::runtime_error("truncated feature sidecar");
  }
}

uint8_t read_u8(std::ifstream& input) {
  char value = 0;
  read_exact(input, &value, 1);
  return static_cast<uint8_t>(static_cast<unsigned char>(value));
}

uint32_t read_u32_le(std::ifstream& input) {
  std::array<uint8_t, 4> bytes{};
  read_exact(
      input,
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  return static_cast<uint32_t>(bytes[0])
      | (static_cast<uint32_t>(bytes[1]) << 8)
      | (static_cast<uint32_t>(bytes[2]) << 16)
      | (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t read_u64_le(std::ifstream& input) {
  std::array<uint8_t, 8> bytes{};
  read_exact(
      input,
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  uint64_t value = 0;
  for (uint32_t index = 0; index < bytes.size(); ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
  }
  return value;
}

float read_f32_le(const uint8_t* bytes) {
  const uint32_t bits = static_cast<uint32_t>(bytes[0])
      | (static_cast<uint32_t>(bytes[1]) << 8)
      | (static_cast<uint32_t>(bytes[2]) << 16)
      | (static_cast<uint32_t>(bytes[3]) << 24);
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits), "float32 is required");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

FeatureTable read_feature_table(
    const std::string& path,
    uint32_t expected_mode,
    uint32_t expected_n) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open feature sidecar");
  }
  input.seekg(0, std::ios::end);
  const std::streampos end_position = input.tellg();
  if (end_position < 0) {
    throw std::runtime_error("failed to size feature sidecar");
  }
  const uint64_t file_size = static_cast<uint64_t>(end_position);
  if (file_size > 50ULL * 1024 * 1024 * 1024) {
    throw std::runtime_error("feature sidecar exceeds 50 GiB");
  }
  input.seekg(0, std::ios::beg);

  std::array<char, 8> magic{};
  read_exact(
      input,
      magic.data(),
      static_cast<std::streamsize>(magic.size()));
  const std::array<char, 8> expected_magic{
      'W', 'P', '2', 'V', '6', 'F', '1', '\0'};
  if (magic != expected_magic) {
    throw std::runtime_error("invalid feature sidecar magic");
  }

  FeatureTable table;
  table.mode = read_u32_le(input);
  const uint32_t n = read_u32_le(input);
  const uint64_t record_count = read_u64_le(input);
  table.dimension = read_u32_le(input);
  if (table.mode != expected_mode || n != expected_n) {
    throw std::runtime_error("feature sidecar identity mismatch");
  }
  if (record_count == 0 || record_count > kMaximumFeatureRecords) {
    throw std::runtime_error("invalid feature record count");
  }
  if ((table.mode == 5 && table.dimension != 1)
      || (table.mode == 6 && table.dimension != kSimMarkDimension)) {
    throw std::runtime_error("invalid feature dimension");
  }

  if (table.mode == 5) {
    for (uint8_t& value : table.relation) {
      value = read_u8(input);
      if (value > 1) {
        throw std::runtime_error("invalid SemStamp relation value");
      }
    }
  } else {
    const uint64_t fixed_bytes =
        8 + 4 + 4 + 8 + 4 + record_count * 9;
    if (file_size < fixed_bytes) {
      throw std::runtime_error("truncated feature sidecar");
    }
    const uint64_t embedding_bytes = file_size - fixed_bytes;
    if (embedding_bytes % (table.dimension * sizeof(float)) != 0) {
      throw std::runtime_error("invalid embedding payload size");
    }
    const uint64_t embedding_values =
        embedding_bytes / sizeof(float);
    if (embedding_values > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("embedding payload is too large");
    }
    table.embeddings.reserve(static_cast<size_t>(embedding_values));
  }

  const size_t size = static_cast<size_t>(record_count);
  table.index_by_span.reserve(size);
  table.span_codes.reserve(size);
  table.allowed.reserve(size);
  table.regions.assign(size, 0);
  table.embedding_offsets.assign(
      size,
      std::numeric_limits<uint64_t>::max());
  table.embedding_norms.assign(size, 0.0);
  const uint64_t base = static_cast<uint64_t>(expected_n) + 1;
  uint64_t previous_code = 0;
  std::vector<uint8_t> embedding_bytes(
      static_cast<size_t>(table.dimension) * sizeof(float));
  for (size_t index = 0; index < size; ++index) {
    const uint32_t start = read_u32_le(input);
    const uint32_t end = read_u32_le(input);
    const uint8_t allowed = read_u8(input);
    if (!(start < end && end <= expected_n) || allowed > 1) {
      throw std::runtime_error("invalid feature span record");
    }
    const uint64_t code = pair_code(start, end, base);
    if (index != 0 && code <= previous_code) {
      throw std::runtime_error(
          "feature span records are not strictly sorted");
    }
    previous_code = code;
    table.index_by_span.emplace(code, static_cast<uint32_t>(index));
    table.span_codes.push_back(code);
    table.allowed.push_back(allowed);
    if (!allowed) {
      continue;
    }
    if (table.mode == 5) {
      const uint8_t region = read_u8(input);
      if (region > 7) {
        throw std::runtime_error("invalid SemStamp region");
      }
      table.regions[index] = region;
      continue;
    }

    table.embedding_offsets[index] = table.embeddings.size();
    read_exact(
        input,
        reinterpret_cast<char*>(embedding_bytes.data()),
        static_cast<std::streamsize>(embedding_bytes.size()));
    double squared_norm = 0.0;
    for (uint32_t component = 0;
         component < table.dimension;
         ++component) {
      const float value = read_f32_le(
          embedding_bytes.data() + component * sizeof(float));
      if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite SimMark embedding");
      }
      table.embeddings.push_back(value);
      squared_norm += static_cast<double>(value)
          * static_cast<double>(value);
    }
    if (!std::isfinite(squared_norm) || squared_norm <= 0.0) {
      throw std::runtime_error("zero-norm SimMark embedding");
    }
    table.embedding_norms[index] = std::sqrt(squared_norm);
  }
  char trailing = 0;
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("trailing feature sidecar bytes");
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading feature sidecar");
  }
  return table;
}

void write_u32_le(std::ofstream& output, uint32_t value) {
  const std::array<uint8_t, 4> bytes{
      static_cast<uint8_t>(value),
      static_cast<uint8_t>(value >> 8),
      static_cast<uint8_t>(value >> 16),
      static_cast<uint8_t>(value >> 24),
  };
  output.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
}

void write_u64_le(std::ofstream& output, uint64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (uint32_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (8 * index));
  }
  output.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
}

void write_synthetic_feature_table(
    const DumpRequest& request,
    uint32_t n,
    const std::unordered_set<uint64_t>& span_seen) {
  if (request.mode != 5 && request.mode != 6) {
    throw std::runtime_error("dump mode must be 5 or 6");
  }
  if (span_seen.empty() || span_seen.size() > kMaximumFeatureRecords) {
    throw std::runtime_error("invalid dumped feature record count");
  }
  std::vector<uint64_t> codes(span_seen.begin(), span_seen.end());
  std::sort(codes.begin(), codes.end());

  std::vector<char> output_buffer(1024 * 1024);
  std::ofstream output;
  output.rdbuf()->pubsetbuf(
      output_buffer.data(),
      static_cast<std::streamsize>(output_buffer.size()));
  output.open(request.path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create synthetic sidecar");
  }
  const std::array<char, 8> magic{
      'W', 'P', '2', 'V', '6', 'F', '1', '\0'};
  output.write(
      magic.data(),
      static_cast<std::streamsize>(magic.size()));
  write_u32_le(output, request.mode);
  write_u32_le(output, n);
  write_u64_le(output, codes.size());
  write_u32_le(
      output,
      request.mode == 5 ? 1 : kSimMarkDimension);
  if (request.mode == 5) {
    for (uint32_t previous = 0; previous < 8; ++previous) {
      for (uint32_t current = 0; current < 8; ++current) {
        const uint8_t value = static_cast<uint8_t>(
            ((previous + 1) * 5 + (current + 3) * 3) % 4 == 0);
        output.put(static_cast<char>(value));
      }
    }
  }

  std::vector<uint8_t> embedding(
      static_cast<size_t>(kSimMarkDimension) * sizeof(float),
      0);
  embedding[2] = 128;
  embedding[3] = 63;
  const uint64_t base = static_cast<uint64_t>(n) + 1;
  for (uint64_t code : codes) {
    const uint32_t start = static_cast<uint32_t>(code / base);
    const uint32_t end = static_cast<uint32_t>(code % base);
    write_u32_le(output, start);
    write_u32_le(output, end);
    output.put(1);
    if (request.mode == 5) {
      const uint8_t region = static_cast<uint8_t>(
          ((start + 3) * 5 + (end + 7) * 3) % 8);
      output.put(static_cast<char>(region));
    } else {
      output.write(
          reinterpret_cast<const char*>(embedding.data()),
          static_cast<std::streamsize>(embedding.size()));
    }
  }
  output.flush();
  if (!output) {
    throw std::runtime_error("failed to write synthetic sidecar");
  }
}

double score_pair(
    uint32_t left,
    uint32_t middle,
    uint32_t right,
    uint32_t mode) {
  if (mode == 1) {
    return static_cast<double>(
        ((static_cast<uint64_t>(left) + 3) * 1009
         + (static_cast<uint64_t>(middle) + 5) * 9176
         + (static_cast<uint64_t>(right) + 7) * 65537)
        % 1000003);
  }
  if (mode == 2) {
    return static_cast<double>(
        ((static_cast<uint64_t>(left) + 3) * 31
         + (static_cast<uint64_t>(middle) + 5) * 17
         + (static_cast<uint64_t>(right) + 7) * 13)
        % 97);
  }
  if (mode == 3) {
    return 0.0;
  }
  throw std::runtime_error("unsupported score mode");
}

std::vector<std::vector<uint32_t>> rank_candidates(
    const Problem& problem) {
  std::vector<std::vector<uint32_t>> output;
  output.reserve(problem.m);
  for (uint32_t rank = 1; rank <= problem.m; ++rank) {
    const int64_t lower_rank =
        static_cast<int64_t>(rank) - problem.k;
    const uint32_t upper_rank = rank + problem.k;
    uint32_t lower = lower_rank <= 0
        ? 0
        : problem.original[static_cast<size_t>(lower_rank - 1)];
    uint32_t upper = upper_rank > problem.m
        ? problem.n
        : problem.original[upper_rank - 1];
    lower = std::max(lower, rank);
    upper = std::min(
        upper,
        problem.n - problem.m + rank - 1);
    std::vector<uint32_t> values;
    values.reserve(upper >= lower ? upper - lower + 1 : 0);
    for (uint32_t value = lower; value <= upper; ++value) {
      values.push_back(value);
    }
    output.push_back(std::move(values));
  }
  return output;
}

Result solve(
    const Problem& problem,
    const DumpRequest* dump_request = nullptr) {
  const uint64_t base = static_cast<uint64_t>(problem.n) + 1;
  const auto candidates = rank_candidates(problem);
  uint64_t candidate_positions = 0;
  for (const auto& values : candidates) {
    candidate_positions += values.size();
  }
  std::vector<uint8_t> is_original(base, 0);
  for (uint32_t value : problem.original) {
    is_original[value] = 1;
  }

  std::unordered_set<uint64_t> span_seen;
  span_seen.reserve(static_cast<size_t>(std::min<uint64_t>(
      2000000,
      std::max<uint64_t>(64, candidate_positions * 12))));
  std::vector<uint8_t> feature_records_queried(
      problem.features.span_codes.size(),
      0);
  auto span_allowed = [&](uint32_t start, uint32_t end) {
    const uint64_t code = pair_code(start, end, base);
    span_seen.insert(code);
    if (problem.span_mode == 1) {
      return true;
    }
    if (problem.span_mode == 2) {
      const auto found = problem.features.index_by_span.find(code);
      if (found == problem.features.index_by_span.end()) {
        throw std::runtime_error("missing external span record");
      }
      feature_records_queried[found->second] = 1;
      return problem.features.allowed[found->second] != 0;
    }
    const size_t index = static_cast<size_t>(code);
    if (index >= problem.span_bits.size()) {
      throw std::runtime_error("span table is too short");
    }
    return problem.span_bits[index] == '1';
  };

  std::vector<uint32_t> endpoints;
  endpoints.reserve(problem.m + 2);
  endpoints.push_back(0);
  endpoints.insert(
      endpoints.end(),
      problem.original.begin(),
      problem.original.end());
  endpoints.push_back(problem.n);
  for (uint32_t index = 0; index + 1 < endpoints.size(); ++index) {
    if (!span_allowed(endpoints[index], endpoints[index + 1])) {
      throw std::runtime_error("illegal original span");
    }
  }

  std::unordered_set<uint64_t> external_scores_used;
  if (problem.score_mode == 4) {
    external_scores_used.reserve(problem.external_scores.size());
  }
  auto feature_index = [&](uint32_t start, uint32_t end) {
    const uint64_t code = pair_code(start, end, base);
    const auto found = problem.features.index_by_span.find(code);
    if (found == problem.features.index_by_span.end()
        || problem.features.allowed[found->second] == 0) {
      throw std::runtime_error("missing legal span feature");
    }
    return found->second;
  };
  auto read_score = [&](
      uint32_t left,
      uint32_t middle,
      uint32_t right) {
    if (problem.score_mode <= 3) {
      return score_pair(left, middle, right, problem.score_mode);
    }
    if (problem.score_mode == 4) {
      const uint64_t code = triple_code(left, middle, right, base);
      const auto found = problem.external_scores.find(code);
      if (found == problem.external_scores.end()) {
        throw std::runtime_error("missing external pair score");
      }
      external_scores_used.insert(code);
      return found->second;
    }
    const uint32_t left_index = feature_index(left, middle);
    const uint32_t right_index = feature_index(middle, right);
    if (problem.score_mode == 5) {
      const uint8_t left_region =
          problem.features.regions[left_index];
      const uint8_t right_region =
          problem.features.regions[right_index];
      return static_cast<double>(
          problem.features.relation[
              static_cast<size_t>(left_region) * 8 + right_region]);
    }
    if (problem.score_mode != 6) {
      throw std::runtime_error("unsupported feature score mode");
    }
    const uint64_t left_offset =
        problem.features.embedding_offsets[left_index];
    const uint64_t right_offset =
        problem.features.embedding_offsets[right_index];
    double dot = 0.0;
    for (uint32_t component = 0;
         component < problem.features.dimension;
         ++component) {
      dot += static_cast<double>(
          problem.features.embeddings[left_offset + component])
          * static_cast<double>(
              problem.features.embeddings[right_offset + component]);
    }
    const double cosine = dot
        / (problem.features.embedding_norms[left_index]
           * problem.features.embedding_norms[right_index]);
    const double distance = cosine >= 0.68 && cosine <= 0.76
        ? 0.0
        : std::min(std::abs(cosine - 0.68), std::abs(cosine - 0.76));
    const double value = std::exp(-250.0 * distance);
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite factorized pair score");
    }
    return value;
  };

  std::unordered_map<uint64_t, double> original_scores;
  original_scores.reserve(problem.m * 2);
  double original_sum = 0.0;
  for (uint32_t index = 1; index + 1 < endpoints.size(); ++index) {
    const uint32_t left = endpoints[index - 1];
    const uint32_t middle = endpoints[index];
    const uint32_t right = endpoints[index + 1];
    const double value = read_score(left, middle, right);
    original_scores.emplace(
        triple_code(left, middle, right, base),
        value);
    original_sum += value;
  }

  uint64_t unique_pair_scores = original_scores.size();
  std::unordered_map<uint64_t, double> pair_cache;
  pair_cache.reserve(static_cast<size_t>(std::min<uint64_t>(
      1000000,
      std::max<uint64_t>(128, candidate_positions * 8))));
  std::vector<std::vector<uint64_t>> pair_keys(problem.m + 2);
  auto cached_score = [&](
      uint32_t left,
      uint32_t middle,
      uint32_t right,
      uint32_t layer) {
    const uint64_t code = triple_code(left, middle, right, base);
    auto cached = pair_cache.find(code);
    if (cached != pair_cache.end()) {
      return cached->second;
    }
    double value;
    const auto original = original_scores.find(code);
    if (original != original_scores.end()) {
      value = original->second;
    } else {
      value = read_score(left, middle, right);
      ++unique_pair_scores;
      if ((problem.score_mode == 5 || problem.score_mode == 6)
          && unique_pair_scores > kMaximumPairScores) {
        throw std::runtime_error(
            "factorized pair score count exceeds frozen limit");
      }
    }
    pair_cache.emplace(code, value);
    pair_keys[layer].push_back(code);
    return value;
  };
  auto expire_scores = [&](uint32_t layer) {
    const int64_t expired =
        static_cast<int64_t>(layer) - (2 * problem.k + 1);
    if (expired < 2) {
      return;
    }
    auto& keys = pair_keys[static_cast<size_t>(expired)];
    for (uint64_t code : keys) {
      pair_cache.erase(code);
    }
    std::vector<uint64_t>().swap(keys);
  };

  std::vector<std::vector<uint32_t>> predecessor_history(problem.m + 1);
  std::vector<std::vector<uint32_t>> boundary_history(problem.m + 1);

  Layer previous;
  previous.groups.reserve(candidates[0].size() * 2);
  const uint32_t remaining_after_first = problem.m - 1;
  for (uint32_t end : candidates[0]) {
    const uint32_t moves = !is_original[end];
    const uint32_t original_after = static_cast<uint32_t>(
        problem.original.end()
        - std::upper_bound(
            problem.original.begin(),
            problem.original.end(),
            end));
    const uint32_t minimum_future_new =
        remaining_after_first > original_after
        ? remaining_after_first - original_after
        : 0;
    if (moves + minimum_future_new > problem.k
        || !span_allowed(0, end)) {
      continue;
    }
    const uint32_t state_index =
        static_cast<uint32_t>(previous.best.size());
    auto inserted = previous.groups.emplace(
        pair_code(0, end, base),
        Slots());
    inserted.first->second.state[moves] = state_index;
    previous.best.push_back(0.0);
    previous.counts.emplace_back(1);
    previous.path_rank.push_back(state_index);
    previous.pair_code.push_back(pair_code(0, end, base));
    previous.moves.push_back(static_cast<uint8_t>(moves));
  }
  predecessor_history[1].assign(previous.best.size(), 0);
  boundary_history[1].reserve(previous.best.size());
  for (uint64_t code : previous.pair_code) {
    boundary_history[1].push_back(
        static_cast<uint32_t>(code % base));
  }

  uint64_t reachable_states = previous.best.size();
  uint64_t peak_layer_states = previous.best.size();
  uint64_t transitions = 0;

  for (uint32_t selected = 1; selected < problem.m; ++selected) {
    const uint32_t layer = selected + 1;
    const auto& next_values = candidates[selected];
    std::vector<uint8_t> next_is_new;
    std::vector<uint8_t> next_caps;
    next_is_new.reserve(next_values.size());
    next_caps.reserve(next_values.size());
    const uint32_t remaining = problem.m - layer;
    for (uint32_t value : next_values) {
      next_is_new.push_back(!is_original[value]);
      const uint32_t original_after = static_cast<uint32_t>(
          problem.original.end()
          - std::upper_bound(
              problem.original.begin(),
              problem.original.end(),
              value));
      const uint32_t minimum_future_new =
          remaining > original_after
          ? remaining - original_after
          : 0;
      next_caps.push_back(
          static_cast<uint8_t>(problem.k - minimum_future_new));
    }

    Layer current;
    current.groups.reserve(previous.groups.size() * 2);
    current.best.reserve(previous.best.size());
    current.counts.reserve(previous.counts.size());
    current.path_rank.reserve(previous.path_rank.size());
    current.pair_code.reserve(previous.pair_code.size());
    current.moves.reserve(previous.moves.size());
    std::vector<uint32_t> current_predecessors;
    std::vector<uint32_t> current_predecessor_ranks;
    current_predecessors.reserve(previous.best.size());
    current_predecessor_ranks.reserve(previous.best.size());

    for (const auto& group_entry : previous.groups) {
      const uint64_t code = group_entry.first;
      const uint32_t left = static_cast<uint32_t>(code / base);
      const uint32_t middle = static_cast<uint32_t>(code % base);
      const auto& slots = group_entry.second.state;
      auto value_it = std::upper_bound(
          next_values.begin(),
          next_values.end(),
          middle);
      size_t value_index =
          static_cast<size_t>(value_it - next_values.begin());
      for (; value_index < next_values.size(); ++value_index) {
        const uint32_t end = next_values[value_index];
        const uint32_t added_move = next_is_new[value_index];
        const int32_t eligible_limit =
            static_cast<int32_t>(next_caps[value_index])
            - static_cast<int32_t>(added_move);
        uint32_t eligible_count = 0;
        for (int32_t moves = 0; moves <= eligible_limit; ++moves) {
          if (slots[static_cast<size_t>(moves)] != kMissingState) {
            ++eligible_count;
          }
        }
        if (eligible_count == 0 || !span_allowed(middle, end)) {
          continue;
        }
        const double score =
            cached_score(left, middle, end, layer);
        const uint64_t destination =
            pair_code(middle, end, base);
        auto inserted = current.groups.emplace(destination, Slots());
        auto& destination_slots = inserted.first->second.state;
        transitions += eligible_count;

        for (int32_t moves = 0; moves <= eligible_limit; ++moves) {
          const uint32_t previous_index =
              slots[static_cast<size_t>(moves)];
          if (previous_index == kMissingState) {
            continue;
          }
          const uint32_t new_moves =
              static_cast<uint32_t>(moves) + added_move;
          const double candidate_sum =
              previous.best[previous_index] + score;
          uint32_t destination_index =
              destination_slots[new_moves];
          const uint32_t predecessor_rank =
              previous.path_rank[previous_index];
          if (destination_index == kMissingState) {
            destination_index =
                static_cast<uint32_t>(current.best.size());
            destination_slots[new_moves] = destination_index;
            current.best.push_back(candidate_sum);
            current.counts.push_back(
                previous.counts[previous_index]);
            current_predecessors.push_back(previous_index);
            current_predecessor_ranks.push_back(predecessor_rank);
            current.pair_code.push_back(destination);
            current.moves.push_back(
                static_cast<uint8_t>(new_moves));
            continue;
          }
          checked_add(
              current.counts[destination_index],
              previous.counts[previous_index]);
          if (candidate_sum < current.best[destination_index]
              || (candidate_sum == current.best[destination_index]
                  && predecessor_rank
                      < current_predecessor_ranks[destination_index])) {
            current.best[destination_index] = candidate_sum;
            current_predecessors[destination_index] = previous_index;
            current_predecessor_ranks[destination_index] =
                predecessor_rank;
          }
        }
      }
    }

    std::vector<uint32_t> order(current.best.size());
    for (uint32_t index = 0; index < order.size(); ++index) {
      order[index] = index;
    }
    std::sort(
        order.begin(),
        order.end(),
        [&](uint32_t left_index, uint32_t right_index) {
          const uint32_t left_rank =
              current_predecessor_ranks[left_index];
          const uint32_t right_rank =
              current_predecessor_ranks[right_index];
          if (left_rank != right_rank) {
            return left_rank < right_rank;
          }
          return current.pair_code[left_index] % base
              < current.pair_code[right_index] % base;
        });
    current.path_rank.assign(current.best.size(), 0);
    for (uint32_t rank = 0; rank < order.size(); ++rank) {
      current.path_rank[order[rank]] = rank;
    }

    predecessor_history[layer] = std::move(current_predecessors);
    auto& boundaries = boundary_history[layer];
    boundaries.reserve(current.pair_code.size());
    for (uint64_t state_code : current.pair_code) {
      boundaries.push_back(
          static_cast<uint32_t>(state_code % base));
    }
    reachable_states += current.best.size();
    peak_layer_states = std::max<uint64_t>(
        peak_layer_states,
        current.best.size());
    previous = std::move(current);
    expire_scores(layer);
  }

  std::array<Count, 4> histogram{};
  double worst_sum = std::numeric_limits<double>::infinity();
  uint32_t worst_moves = kMissingState;
  uint32_t worst_rank = kMissingState;
  uint32_t worst_index = kMissingState;
  const uint32_t final_layer = problem.m + 1;
  for (const auto& group_entry : previous.groups) {
    const uint64_t code = group_entry.first;
    const uint32_t left = static_cast<uint32_t>(code / base);
    const uint32_t middle = static_cast<uint32_t>(code % base);
    if (!span_allowed(middle, problem.n)) {
      continue;
    }
    const double score =
        cached_score(left, middle, problem.n, final_layer);
    for (uint32_t moves = 0; moves <= problem.k; ++moves) {
      const uint32_t state_index =
          group_entry.second.state[moves];
      if (state_index == kMissingState) {
        continue;
      }
      ++transitions;
      const double final_sum = previous.best[state_index] + score;
      checked_add(histogram[moves], previous.counts[state_index]);
      const uint32_t path_rank = previous.path_rank[state_index];
      if (final_sum < worst_sum
          || (final_sum == worst_sum
              && (worst_moves == kMissingState
                  || moves < worst_moves
                  || (moves == worst_moves
                      && path_rank < worst_rank)))) {
        worst_sum = final_sum;
        worst_moves = moves;
        worst_rank = path_rank;
        worst_index = state_index;
      }
    }
  }
  if (worst_index == kMissingState) {
    throw std::runtime_error("empty threat set");
  }

  std::vector<uint32_t> reversed;
  reversed.reserve(problem.m);
  uint32_t state_index = worst_index;
  for (uint32_t layer = problem.m; layer >= 1; --layer) {
    reversed.push_back(boundary_history[layer][state_index]);
    if (layer > 1) {
      state_index = predecessor_history[layer][state_index];
    }
  }
  std::reverse(reversed.begin(), reversed.end());

  Count threat_set_size = 0;
  for (uint32_t moves = 0; moves <= problem.k; ++moves) {
    checked_add(threat_set_size, histogram[moves]);
  }
  if (problem.score_mode == 4
      && external_scores_used.size() != problem.external_scores.size()) {
    throw std::runtime_error("unused external pair score");
  }
  if ((problem.score_mode == 5 || problem.score_mode == 6)
      && std::find(
          feature_records_queried.begin(),
          feature_records_queried.end(),
          0) != feature_records_queried.end()) {
    throw std::runtime_error("unused external span record");
  }
  if (dump_request != nullptr) {
    if (problem.score_mode != 3 || problem.span_mode != 1) {
      throw std::runtime_error(
          "synthetic sidecar dump requires score mode 3 and span mode 1");
    }
    write_synthetic_feature_table(
        *dump_request,
        problem.n,
        span_seen);
  }
  return Result{
      problem.bias + original_sum / problem.normalization,
      problem.bias + worst_sum / problem.normalization,
      threat_set_size,
      histogram,
      reversed,
      reachable_states,
      transitions,
      peak_layer_states,
      candidate_positions,
      span_seen.size(),
      unique_pair_scores,
  };
}

Problem read_problem() {
  Problem problem{};
  if (!(std::cin
        >> problem.n
        >> problem.m
        >> problem.k
        >> problem.score_mode
        >> problem.span_mode
        >> problem.normalization
        >> problem.bias
        >> problem.threshold)) {
    throw std::runtime_error("failed to read problem header");
  }
  if (problem.n < 2 || problem.m < 1 || problem.m >= problem.n
      || problem.k > 3 || problem.k > problem.m
      || !std::isfinite(problem.normalization)
      || problem.normalization <= 0
      || !std::isfinite(problem.bias)
      || !std::isfinite(problem.threshold)) {
    throw std::runtime_error("invalid problem header");
  }
  problem.original.resize(problem.m);
  for (uint32_t& boundary : problem.original) {
    std::cin >> boundary;
  }
  if (!std::is_sorted(
          problem.original.begin(),
          problem.original.end())
      || std::adjacent_find(
          problem.original.begin(),
          problem.original.end())
          != problem.original.end()
      || problem.original.front() == 0
      || problem.original.back() >= problem.n) {
    throw std::runtime_error("invalid original boundaries");
  }
  if (problem.span_mode == 0) {
    std::cin >> problem.span_bits;
    const size_t expected =
        static_cast<size_t>(problem.n + 1) * (problem.n + 1);
    if (problem.span_bits.size() != expected) {
      throw std::runtime_error("invalid span table size");
    }
    if (problem.span_bits.find_first_not_of("01")
        != std::string::npos) {
      throw std::runtime_error("span table must contain only 0 or 1");
    }
  } else if (problem.span_mode == 2) {
    if (problem.score_mode != 5 && problem.score_mode != 6) {
      throw std::runtime_error(
          "factorized span mode requires score mode 5 or 6");
    }
    std::string feature_path;
    if (!(std::cin >> feature_path) || feature_path.empty()) {
      throw std::runtime_error("failed to read feature sidecar path");
    }
    problem.features = read_feature_table(
        feature_path,
        problem.score_mode,
        problem.n);
  } else if (problem.span_mode != 1) {
    throw std::runtime_error("unsupported span mode");
  }
  if (problem.score_mode == 4) {
    uint64_t score_count = 0;
    if (!(std::cin >> score_count)) {
      throw std::runtime_error("failed to read external score count");
    }
    if (score_count > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("external score table is too large");
    }
    problem.external_scores.reserve(static_cast<size_t>(score_count));
    const uint64_t base = static_cast<uint64_t>(problem.n) + 1;
    for (uint64_t index = 0; index < score_count; ++index) {
      uint32_t left = 0;
      uint32_t middle = 0;
      uint32_t right = 0;
      double value = 0.0;
      if (!(std::cin >> left >> middle >> right >> value)) {
        throw std::runtime_error("failed to read external pair score");
      }
      if (!(left < middle && middle < right && right <= problem.n)
          || !std::isfinite(value)) {
        throw std::runtime_error("invalid external pair score");
      }
      const auto inserted = problem.external_scores.emplace(
          triple_code(left, middle, right, base),
          value);
      if (!inserted.second) {
        throw std::runtime_error("duplicate external pair score");
      }
    }
  } else if (problem.score_mode == 5 || problem.score_mode == 6) {
    if (problem.span_mode != 2) {
      throw std::runtime_error(
          "factorized score mode requires span mode 2");
    }
  } else if (problem.score_mode < 1 || problem.score_mode > 3) {
    throw std::runtime_error("unsupported score mode");
  }
  return problem;
}

std::string join_uints(const std::vector<uint32_t>& values) {
  std::ostringstream output;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

std::string count_to_string(Count value) {
  if (value == 0) {
    return "0";
  }
  std::string digits;
  while (value != 0) {
    digits.push_back(
        static_cast<char>('0' + static_cast<uint32_t>(value % 10)));
    value /= 10;
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

void write_result(uint32_t case_index, const Result& result) {
  std::cout
      << case_index << '\t'
      << std::setprecision(17) << result.original_score << '\t'
      << std::setprecision(17) << result.certified_score << '\t'
      << count_to_string(result.threat_set_size) << '\t';
  for (uint32_t moves = 0; moves < result.histogram.size(); ++moves) {
    if (moves != 0) {
      std::cout << ',';
    }
    std::cout
        << moves << ':'
        << count_to_string(result.histogram[moves]);
  }
  std::cout
      << '\t' << join_uints(result.witness)
      << '\t' << result.reachable_states
      << '\t' << result.transitions
      << '\t' << result.peak_layer_states
      << '\t' << result.candidate_positions
      << '\t' << result.unique_spans
      << '\t' << result.unique_pairs
      << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  try {
    DumpRequest dump_request{};
    const DumpRequest* dump = nullptr;
    if (argc != 1) {
      if (argc != 4
          || std::string(argv[1]) != "--dump-synthetic-sidecar") {
        throw std::runtime_error("invalid command-line arguments");
      }
      dump_request.path = argv[2];
      const std::string mode_text = argv[3];
      if (mode_text != "5" && mode_text != "6") {
        throw std::runtime_error("dump mode must be 5 or 6");
      }
      dump_request.mode = static_cast<uint32_t>(
          mode_text == "5" ? 5 : 6);
      dump = &dump_request;
    }
    uint32_t case_count = 0;
    if (!(std::cin >> case_count)) {
      throw std::runtime_error("failed to read case count");
    }
    if (dump != nullptr && case_count != 1) {
      throw std::runtime_error(
          "synthetic sidecar dump requires exactly one case");
    }
    for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
      write_result(case_index, solve(read_problem(), dump));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
