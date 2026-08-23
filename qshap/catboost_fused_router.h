#ifndef QSHAP_CATBOOST_FUSED_ROUTER_H
#define QSHAP_CATBOOST_FUSED_ROUTER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qshap_catboost_core {

struct TreeRouteSpec {
    std::vector<int> level_feature;
    std::vector<float> level_border;
};

struct QuantizedTreeRoute {
    std::vector<int> level_feature;
    std::vector<std::uint16_t> level_border_bin;
};

struct QuantizedFeatureBins {
    bool wide = false;
    std::vector<std::uint8_t> bins8;
    std::vector<std::uint16_t> bins16;
};

struct QuantizedRoutingPlan {
    // Feature-major compact storage: feature_bins[feature][observation].
    // Unused features stay empty, so memory scales with used features only.
    std::vector<QuantizedFeatureBins> feature_bins;
    std::vector<QuantizedTreeRoute> tree_routes;
};

inline QuantizedRoutingPlan build_quantized_routing_plan(
    const double *x,
    int n_samples,
    int n_features,
    bool column_major,
    const std::vector<TreeRouteSpec> &tree_specs,
    const std::vector<int> &nan_goes_right) {
    if (static_cast<int>(nan_goes_right.size()) != n_features) {
        throw std::runtime_error("CatBoost missing-value metadata is incomplete.");
    }

    std::vector<std::vector<float>> feature_borders(
        static_cast<size_t>(n_features)
    );
    for (const TreeRouteSpec &tree : tree_specs) {
        if (tree.level_feature.size() != tree.level_border.size()) {
            throw std::runtime_error("CatBoost route metadata is inconsistent.");
        }
        for (size_t level = 0; level < tree.level_feature.size(); ++level) {
            const int feature = tree.level_feature[level];
            if (feature < 0 || feature >= n_features ||
                !std::isfinite(tree.level_border[level])) {
                throw std::runtime_error("CatBoost split metadata is invalid.");
            }
            feature_borders[static_cast<size_t>(feature)].push_back(
                tree.level_border[level]
            );
        }
    }

    QuantizedRoutingPlan plan;
    plan.feature_bins.resize(static_cast<size_t>(n_features));
    for (int feature = 0; feature < n_features; ++feature) {
        std::vector<float> &borders = feature_borders[static_cast<size_t>(feature)];
        if (borders.empty()) continue;

        std::sort(borders.begin(), borders.end());
        borders.erase(std::unique(borders.begin(), borders.end()), borders.end());
        if (borders.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error(
                "CatBoost feature has too many distinct split borders."
            );
        }

        QuantizedFeatureBins &quantized =
            plan.feature_bins[static_cast<size_t>(feature)];
        quantized.wide = borders.size() >
            static_cast<size_t>(std::numeric_limits<std::uint8_t>::max());
        if (quantized.wide) {
            quantized.bins16.resize(static_cast<size_t>(n_samples));
        } else {
            quantized.bins8.resize(static_cast<size_t>(n_samples));
        }

        for (int row = 0; row < n_samples; ++row) {
            const size_t source_index = column_major
                ? static_cast<size_t>(feature) * n_samples + row
                : static_cast<size_t>(row) * n_features + feature;
            const float value = static_cast<float>(x[source_index]);
            std::uint16_t bin = 0;
            if (std::isnan(value)) {
                bin = nan_goes_right[static_cast<size_t>(feature)]
                    ? static_cast<std::uint16_t>(borders.size())
                    : static_cast<std::uint16_t>(0);
            } else {
                const auto it = std::lower_bound(
                    borders.begin(), borders.end(), value
                );
                bin = static_cast<std::uint16_t>(
                    std::distance(borders.begin(), it)
                );
            }

            if (quantized.wide) {
                quantized.bins16[static_cast<size_t>(row)] = bin;
            } else {
                quantized.bins8[static_cast<size_t>(row)] =
                    static_cast<std::uint8_t>(bin);
            }
        }
    }

    plan.tree_routes.reserve(tree_specs.size());
    for (const TreeRouteSpec &tree : tree_specs) {
        QuantizedTreeRoute route;
        route.level_feature = tree.level_feature;
        route.level_border_bin.resize(tree.level_border.size());
        for (size_t level = 0; level < tree.level_feature.size(); ++level) {
            const int feature = tree.level_feature[level];
            const float border = tree.level_border[level];
            const std::vector<float> &borders =
                feature_borders[static_cast<size_t>(feature)];
            const auto it = std::lower_bound(borders.begin(), borders.end(), border);
            if (it == borders.end() || *it != border) {
                throw std::runtime_error(
                    "CatBoost split border was not quantized consistently."
                );
            }
            route.level_border_bin[level] = static_cast<std::uint16_t>(
                std::distance(borders.begin(), it)
            );
        }
        plan.tree_routes.push_back(std::move(route));
    }
    return plan;
}

template <typename Consumer>
inline void route_tree_tiled(
    const QuantizedRoutingPlan &routing,
    const QuantizedTreeRoute &tree_route,
    int n_samples,
    Consumer consume) {
    const int depth = static_cast<int>(tree_route.level_feature.size());
    if (depth < 0 || depth >= 30) {
        throw std::runtime_error("Invalid CatBoost symmetric-tree depth.");
    }

    // A 4096-element uint32 leaf block is 16 KiB and remains L1-sized. Routing
    // is level-major and branchless inside the tile; grouped statistics consume
    // the tile immediately, so no n x trees leaf matrix is ever materialized.
    constexpr int routing_block_size = 4096;
    std::vector<std::uint32_t> leaf_block(
        static_cast<size_t>(std::min(n_samples, routing_block_size)), 0
    );

    for (int block_start = 0; block_start < n_samples;
         block_start += routing_block_size) {
        const int block_n = std::min(
            routing_block_size, n_samples - block_start
        );
        std::fill(
            leaf_block.begin(),
            leaf_block.begin() + block_n,
            static_cast<std::uint32_t>(0)
        );

        for (int level = 0; level < depth; ++level) {
            const int feature =
                tree_route.level_feature[static_cast<size_t>(level)];
            const QuantizedFeatureBins &quantized =
                routing.feature_bins[static_cast<size_t>(feature)];
            const std::uint16_t border_bin =
                tree_route.level_border_bin[static_cast<size_t>(level)];
            const std::uint32_t bit =
                static_cast<std::uint32_t>(1) << (depth - 1 - level);

            if (quantized.wide) {
                const std::uint16_t *bin_column =
                    quantized.bins16.data() + block_start;
                for (int offset = 0; offset < block_n; ++offset) {
                    leaf_block[static_cast<size_t>(offset)] +=
                        static_cast<std::uint32_t>(
                            bin_column[offset] > border_bin
                        ) * bit;
                }
            } else {
                const std::uint8_t *bin_column =
                    quantized.bins8.data() + block_start;
                const std::uint8_t border_bin8 =
                    static_cast<std::uint8_t>(border_bin);
                for (int offset = 0; offset < block_n; ++offset) {
                    leaf_block[static_cast<size_t>(offset)] +=
                        static_cast<std::uint32_t>(
                            bin_column[offset] > border_bin8
                        ) * bit;
                }
            }
        }

        for (int offset = 0; offset < block_n; ++offset) {
            consume(
                block_start + offset,
                static_cast<int>(leaf_block[static_cast<size_t>(offset)])
            );
        }
    }
}

}  // namespace qshap_catboost_core

#endif
