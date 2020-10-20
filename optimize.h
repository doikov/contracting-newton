#ifndef OPTIMIZE_H_
#define OPTIMIZE_H_

#include <string>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace optimize {

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SMatrix;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

class History {
public:
    std::vector<int> iteration_;
    std::vector<int> data_accesses_;
    std::vector<double> elapsed_seconds_;
    std::vector<double> func_;
    std::string status_;
    bool dump(const std::string& filename) const;
};

class Parameters {
public:
    std::vector<std::vector<std::pair<int, double>>>* data_;
    SMatrix* A_;
    SMatrix* AT_;
    double R_;
    Vector* x_0_; 
    int n_iters_;
    double max_time_;
    int logging_iter_period_;
    double logging_time_period_;
    double display_time_period_;
    bool display_;
    double inner_eps_;
};

void GradientMethod(
        const Parameters& params, 
        const double L_0, 
        const bool line_search, 
        History* history);

void FastGradientMethod(
        const Parameters& params, 
        const double L_0, 
        const bool line_search, 
        History* history);

void FrankWolfe(
        const Parameters& params, 
        const double c_0, 
        const bool decrease_gamma,
        History* history);

void ContractingNewton(
        const Parameters& params, 
        const double c_0, 
        const bool decrease_gamma,
        History* history);

void AggregatingNewton(
        const Parameters& params, 
        History* history);

void SVRG(
        const Parameters& params, 
        const double L_0, 
        const bool variance_reduction,
        History* history);

void StochasticContractingNewton(
        const Parameters& params,
        const double c_0,
        const bool decrease_gamma,
        const bool variance_reduction,
        const bool hessian_variance_reduction,
        History* history);

}  // namespace optimize

#endif  // OPTIMIZE_H_