#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
using Count = unsigned __int128;

void checked_add(Count& destination, Count value) {
  const Count maximum = static_cast<Count>(-1);
  if (maximum - destination < value) {
    throw std::overflow_error("threat-set count exceeds 128 bits");
  }
  destination += value;
}

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

Result solve(const Problem& problem) {
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
  auto span_allowed = [&](uint32_t start, uint32_t end) {
    const uint64_t code = pair_code(start, end, base);
    span_seen.insert(code);
    if (problem.span_mode == 1) {
      return true;
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
  auto read_score = [&](
      uint32_t left,
      uint32_t middle,
      uint32_t right) {
    if (problem.score_mode != 4) {
      return score_pair(left, middle, right, problem.score_mode);
    }
    const uint64_t code = triple_code(left, middle, right, base);
    const auto found = problem.external_scores.find(code);
    if (found == problem.external_scores.end()) {
      throw std::runtime_error("missing external pair score");
    }
    external_scores_used.insert(code);
    return found->second;
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

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  try {
    uint32_t case_count = 0;
    if (!(std::cin >> case_count)) {
      throw std::runtime_error("failed to read case count");
    }
    for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
      write_result(case_index, solve(read_problem()));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
  return 0;
}
