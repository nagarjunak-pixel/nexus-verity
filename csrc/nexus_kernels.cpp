#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

// 1. Doubly-Robust (Dudik-Langford-Li) + Self-Normalized IPS Off-Policy Counterfactual Evaluator
void nexus_doubly_robust_ope_f64(
    const int32_t* logged_actions,
    const int32_t* target_actions,
    const double* logged_propensities,
    const double* observed_rewards,
    const double* qhat_logged,
    const double* qhat_target,
    int32_t n,
    double clip_min_propensity,
    double* out_dr_value,
    double* out_dr_stderr,
    double* out_ips_value,
    double* out_snips_value
) {
    if (n <= 0) {
        *out_dr_value = 0.0;
        *out_dr_stderr = 0.0;
        *out_ips_value = 0.0;
        *out_snips_value = 0.0;
        return;
    }

    const double eps = std::max(1e-6, clip_min_propensity);
    std::vector<double> dr_terms(static_cast<std::size_t>(n), 0.0);
    double sum_dr = 0.0;
    double sum_ips = 0.0;
    double sum_weights = 0.0;

    for (int32_t i = 0; i < n; ++i) {
        const double p = std::max(eps, logged_propensities[i]);
        const double match = (logged_actions[i] == target_actions[i]) ? 1.0 : 0.0;
        const double w = match / p;

        const double ips_i = w * observed_rewards[i];
        const double dr_i = qhat_target[i] + w * (observed_rewards[i] - qhat_logged[i]);

        dr_terms[static_cast<std::size_t>(i)] = dr_i;
        sum_dr += dr_i;
        sum_ips += ips_i;
        sum_weights += w;
    }

    const double mean_dr = sum_dr / static_cast<double>(n);
    const double mean_ips = sum_ips / static_cast<double>(n);
    const double snips = (sum_weights > 1e-12) ? (sum_ips / sum_weights) : 0.0;

    double sq_diff = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double d = dr_terms[static_cast<std::size_t>(i)] - mean_dr;
        sq_diff += d * d;
    }
    const double var_dr = (n > 1) ? (sq_diff / static_cast<double>(n - 1)) : 0.0;
    const double se_dr = std::sqrt(var_dr / static_cast<double>(n));

    *out_dr_value = mean_dr;
    *out_dr_stderr = se_dr;
    *out_ips_value = mean_ips;
    *out_snips_value = snips;
}

// 2. Vovk-Lei-Wasserman Finite-Sample Split Conformal Order-Statistic Quantile:
//    k = ceil((n_cal + 1) * (1 - alpha)) (1-indexed, clamped to [1, n_cal])
double nexus_conformal_quantile_f64(
    const double* nonconformity_scores,
    int32_t n_cal,
    double alpha
) {
    if (n_cal <= 0) {
        return 0.0;
    }
    std::vector<double> sorted(
        nonconformity_scores,
        nonconformity_scores + static_cast<std::size_t>(n_cal)
    );
    std::sort(sorted.begin(), sorted.end());

    const double raw_rank = std::ceil(static_cast<double>(n_cal + 1) * (1.0 - alpha));
    int32_t rank_1based = static_cast<int32_t>(raw_rank);
    if (rank_1based < 1) {
        rank_1based = 1;
    }
    if (rank_1based > n_cal) {
        rank_1based = n_cal;
    }
    return sorted[static_cast<std::size_t>(rank_1based - 1)];
}

}  // extern "C"
