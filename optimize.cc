
#include "optimize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "Eigen/Eigenvalues"

namespace optimize {

const double EPS = 1e-9;
const double INF = 1e+20;
const double REGION_TOLERANCE = 1e-3;
const int N_LINE_SEARCH_ITERS = 40;
const int RANDOM_SEED = 31415;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;

inline double Cube(const double x)
{
    return x * x * x;
}

/*
    Computes phi(t) = log(1 + exp(t))
*/
inline double Log_one_exp(double t) 
{
    if (t > 0) 
    {
        return t + log(1 + exp(-t));
    } 
    else 
    {
        return log(1 + exp(t));
    }
}

/*
    Computes phi'(t) = 1 / (1 + exp(-t))
*/
inline double Sigmoid(double t)
{
    return 1.0 / (1 + exp(-t));
}

/* 
    Computes phi''(t) = phi'(t) * (1 - phi'(t)) 
*/
inline double Sigmoid_prime(double t)
{
    const auto s = Sigmoid(t);
    return s * (1 - s);
}

/* 
    Returns the maximal power of two, 
    less or equal than k. 
*/
inline int Pi(int k)
{
    k |= (k >> 1);
    k |= (k >> 2);
    k |= (k >> 4);
    k |= (k >> 8);
    k |= (k >> 16);
    return (k + 1) >> 1;
}

bool History::dump(const std::string& filename) const 
{
    std::ofstream output;
    output.open(filename);
    if (!output) {
        std::cerr << "E: Unable to open file " << filename << std::endl;
        return false;
    }
    for (int i = 0; i < iteration_.size(); ++i) 
    {
        output << iteration_[i];
        output << (i + 1 != iteration_.size() ? ", " : "\n");
    }
    for (int i = 0; i < elapsed_seconds_.size(); ++i)
    {
        output << std::fixed << std::setprecision(5) << elapsed_seconds_[i];
        output << (i + 1 != elapsed_seconds_.size() ? ", " : "\n");
    }
    for (int i = 0; i < func_.size(); ++i)
    {
        output << std::fixed << std::setprecision(10) << func_[i];
        output << (i + 1 != func_.size() ? ", " : "\n");
    }
    for (int i = 0; i < data_accesses_.size(); ++i)
    {
        output << data_accesses_[i];
        output << (i + 1 != data_accesses_.size() ? ", " : "\n");
    }
    output.close();
    return true;
}

void InitDisplay(const Parameters& params, const std::string& method_name)
{
    if (params.display_)
    {
        std::cout << method_name << "\n"
                  << "Iter \t\t : \t Func \t\t\t Time (s) \t\t Data" << "\n"
                  << std::string(80, '-')
                  << std::endl;
    }
}

void FinilizeDisplay(const Parameters& params, const History& history)
{
    if (params.display_)
    {
        std::cout << std::string(80, '-')
                  << "\n" << history.status_ << std::endl;
    }
}

template<class Func>
void UpdateHistory(const Parameters& params, const TimePoint& start_time,
                   const int k, const int data_accesses,
                   Func&& func, TimePoint* last_logging_time,
                   TimePoint* last_display_time, History* history, 
                   bool *to_finish)
{
    *to_finish = false;
    if (k == params.n_iters_ || k % params.logging_iter_period_ == 0)
    {
        auto current_time = std::chrono::system_clock::now();
        if (k == params.n_iters_  || k == 0 ||
            std::chrono::duration<double>(current_time - *last_logging_time)
                .count() > params.logging_time_period_)
        {
            history->iteration_.push_back(k);
            history->data_accesses_.push_back(data_accesses);
            history->elapsed_seconds_.push_back(
                    std::chrono::duration<double>(current_time - start_time)
                    .count());
            history->func_.push_back(func());
            *last_logging_time = current_time;
            if (params.display_ && 
                (k == params.n_iters_ || k == 0 ||
                std::chrono::duration<double>(current_time - *last_display_time)
                .count() > params.display_time_period_))
            {
                std::cout << std::left << std::setw(12)
                          << k << "\t : \t" 
                          << std::left << std::setw(16)
                          << std::fixed << std::setprecision(12)
                          << history->func_.back() << " \t"
                          << std::setprecision(3) << std::setw(10)
                          << history->elapsed_seconds_.back() << " \t\t"
                          << history->data_accesses_.back()
                          << std::endl;
                *last_display_time = current_time;
            }
            if (k == params.n_iters_)
            {
                history->status_ = "iterations_exceeded";
                *to_finish = true;
            }
            if (history->elapsed_seconds_.back() > params.max_time_)
            {
                history->status_ = "time_limit_exceeded";
                *to_finish = true;
            }
        }
    }
}

void GradientMethod(const Parameters& params, const double L_0, 
                    const bool line_search, History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = m;

    Vector x_k = *params.x_0_;
    Vector Ax = (*params.A_) * x_k;
    double func_k = inv_m * Ax.unaryExpr(&Log_one_exp).sum();
    Vector grad_k = inv_m * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
    double L_k = L_0;

    Vector h(n);
    Vector T(n);
    double func_T;

    InitDisplay(params, "Gradient Method");
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 func_k;
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        for (int i = 0; i < N_LINE_SEARCH_ITERS; ++i, L_k *= 2)
        {
            T = x_k - 1.0 / L_k * grad_k;
            double T_norm = T.norm();
            if (T_norm > params.R_ + EPS) 
            {
                // Project onto the ball.
                T *= params.R_ / T_norm;
            }
            Ax = (*params.A_) * T;
            func_T = inv_m * Ax.unaryExpr(&Log_one_exp).sum();

            if (!line_search)
            {
                break;
            }

            h = T - x_k;
            if (func_T <= func_k + grad_k.dot(h) + 0.5 * L_k * h.squaredNorm() 
                          + EPS)
            {
                L_k *= 0.5;
                break;
            }

            if (i == N_LINE_SEARCH_ITERS - 1) 
            {
                std::cerr << "W: N_LINE_SEARCH_ITERS exceeded "
                          << "in Gradient Method, k = "
                          << k << std::endl;
                break; 
            }
        }
        x_k = T;
        func_k = func_T;
        grad_k = inv_m * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
        data_accesses += m;
    }
    FinilizeDisplay(params, *history);
}


void FastGradientMethod(const Parameters& params, const double L_0, 
                        const bool line_search, History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = m;

    Vector x_k = *params.x_0_;
    Vector Ax = (*params.A_) * x_k;
    Vector v_k = x_k;
    Vector Av = Ax;
    double func_k = inv_m * Ax.unaryExpr(&Log_one_exp).sum();

    double L_k = L_0;
    double A_k = 0.0;

    Vector y_k(n);
    Vector Ay(n);
    Vector grad_y_k(n);

    Vector T(n);
    Vector AT(n);
    Vector h(n);
    double func_T;
    double a_k_new;
    double A_k_new;

    InitDisplay(params, "Fast Gradient Method");
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) 
                                 ? INF 
                                 : func_k;
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        for (int i = 0; i < N_LINE_SEARCH_ITERS; ++i, L_k *= 2)
        {
            // Solve quadratic equation a_k_new^2 / (a_k_new + A_k) = 1 / L_k.
            a_k_new = 0.5 * (1 + sqrt(1 + 4 * L_k * A_k)) / L_k;
            A_k_new = A_k + a_k_new;

            double gamma_k = a_k_new / A_k_new;
            y_k = gamma_k * v_k + (1 - gamma_k) * x_k;
            Ay = gamma_k * Av + (1 - gamma_k) * Ax;

            grad_y_k = inv_m * ((*params.AT_) * Ay.unaryExpr(&Sigmoid));
            T = y_k - 1.0 / L_k * grad_y_k;
            double T_norm = T.norm();
            if (T_norm > params.R_ + EPS) 
            {
                // Project onto the ball.
                T *= params.R_ / T_norm;
            }
            AT = (*params.A_) * T;
            func_T = inv_m * AT.unaryExpr(&Log_one_exp).sum();

            if (!line_search)
            {
                break;
            }

            double func_y_k = inv_m * Ay.unaryExpr(&Log_one_exp).sum();
            h = T - y_k;
            if (func_T <= func_y_k + grad_y_k.dot(h) 
                          + 0.5 * L_k * h.squaredNorm() + EPS)
            {
                L_k *= 0.5;
                break;
            }

            if (i == N_LINE_SEARCH_ITERS - 1) 
            {
                std::cerr << "W: N_LINE_SEARCH_ITERS exceeded "
                          << "in Fast Gradient Method, k = "
                          << k << std::endl;
                break; 
            }
        }

        v_k = T + (A_k / a_k_new) * (T - x_k);
        Av = AT + (A_k / a_k_new) * (AT - Ax);

        x_k = T;
        func_k = func_T;
        Ax = AT;

        A_k = A_k_new;
        data_accesses += m;
    }
    FinilizeDisplay(params, *history);
}

void FrankWolfe(const Parameters& params, const double c_0, 
                const bool decrease_gamma, History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = m;

    Vector x_k = *params.x_0_;
    Vector Ax = (*params.A_) * x_k;
    Vector grad_k(n);

    std::string gamma_str = "gamma_k = " + std::to_string(c_0);
    if (decrease_gamma)
    {
        gamma_str += " / (3 + k)"; 
    }
    InitDisplay(params, "Frank-Wolfe Method, " + gamma_str);
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 inv_m * Ax.unaryExpr(&Log_one_exp).sum();
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        double gamma_k = c_0;
        if (decrease_gamma) 
        {
            gamma_k /= 3.0 + k;
        }

        grad_k = inv_m * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
        double grad_k_norm = grad_k.norm();
        if (grad_k_norm < EPS)
        {
            x_k -= gamma_k * x_k;
        }
        else
        {
            x_k -= gamma_k * (grad_k * (params.R_ / grad_k_norm) + x_k);
        }
        Ax = (*params.A_) * x_k;
        data_accesses += m;
    }
    FinilizeDisplay(params, *history);
}

/*
    Solve linear system Ax = b,
    where A = H + tau I, with tridiagonal matrix H
    (given by its diagonal and subdiagonal vectors).
*/
void SolveTridiagonalSystem(const Vector& diag, const Vector& subdiag,
                            const double tau, const Vector& b,
                            Vector* x, Vector* buffer)
{
    const int n = diag.size();
    buffer->resize(n - 1);
    x->resize(n);
    auto& c = *buffer;
    auto& d = *x;

    c[0] = subdiag[0] / (diag[0] + tau);
    d[0] = b[0] / (diag[0] + tau);
    for (int i = 1; i < n - 1; ++i)
    {
        double w = diag[i] + tau - subdiag[i - 1] * c[i - 1];
        c[i] = subdiag[i] / w;
        d[i] = (b[i] - subdiag[i - 1] * d[i - 1]) / w;
    }
    d[n - 1] = (b[n - 1] - subdiag[n - 2] * d[n - 2]) / 
                  (diag[n - 1] + tau - subdiag[n - 2] * c[n - 2]);
    for (int i = n - 2; i >= 0; --i)
    {
        d[i] -= c[i] * d[i + 1];  
    }
}

/*
    Minimize q(s) = <g, s> + 1/2 <Hs, s>,
    s.t. ||s||_2 <= R.
*/
void MinimizeQuadraticOnL2Ball(const Vector& g, const Matrix& H, const double R,
                               const double inner_eps, Vector* s)
{
    const int n = g.size();

    Eigen::Tridiagonalization<Matrix> H_tridiag(H);
    const auto Q = H_tridiag.matrixQ();
    const Vector diag = H_tridiag.diagonal();
    const Vector subdiag = H_tridiag.subDiagonal();
    Vector g_(Q.transpose() * g);

    // Solve nonlinear 1-D equation phi(tau) = 0,
    // where 
    //      phi(tau) := 1 / ||S(tau)||_2 - 1 / R
    // is increasing concave function, with
    //      S(tau) := (H_ + tau * I)^{-1} g_.

    // Initial tau.
    double tau = 1.0;
    Vector S_tau(n);
    Vector buffer(n - 1);
    double S_tau_norm;
    double phi_tau;

    // Find tau s.t. phi(tau) <= 0.
    for (int i = 0; i < N_LINE_SEARCH_ITERS + 1; ++i, tau *= 0.5)
    {
        if (i == N_LINE_SEARCH_ITERS)
        {
            std::cerr << "W: Preliminaty line search iterations exceeded " 
                      << "in MinimizeQuadraticOnL2Ball" << std::endl;
            break;
        }

        SolveTridiagonalSystem(diag, subdiag, tau, g_, &S_tau, &buffer);
        S_tau_norm = S_tau.norm();
        phi_tau = 1.0 / S_tau_norm - 1.0 / R;
        if (phi_tau < inner_eps || tau < inner_eps)
        {
            break;
        }
    }
    if (phi_tau < -inner_eps)
    {
        // Run 1-D Newton method.
        Vector S_tau_grad(n);
        for (int i = 0; i < N_LINE_SEARCH_ITERS + 1; ++i)
        {
            if (i == N_LINE_SEARCH_ITERS)
            {
                std::cerr << "W: 1-D Newton iterations exceeded "
                          << "in MinimizeQuadraticOnL2Ball" << std::endl;
                break;
            }

            SolveTridiagonalSystem(diag, subdiag, tau, S_tau, &S_tau_grad, 
                                   &buffer);
            double phi_tau_prime = 
                (1.0 / Cube(S_tau_norm)) *
                static_cast<double>(S_tau.transpose() * S_tau_grad);
            // Newton step.
            tau -= phi_tau / phi_tau_prime;

            SolveTridiagonalSystem(diag, subdiag, tau, g_, &S_tau, &buffer);
            S_tau_norm = S_tau.norm();
            phi_tau = 1.0 / S_tau_norm - 1.0 / R;

            if (abs(phi_tau) < inner_eps || abs(phi_tau_prime) < inner_eps)
            {
                break;
            }
        }
    }
    *s = -(Q * S_tau);
}

void ContractingNewton(const Parameters& params, const double c_0, 
                       const bool decrease_gamma, History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = m;

    Vector x_k = *params.x_0_;
    Vector Ax = (*params.A_) * x_k;
    Vector g_k(n);
    Matrix H_k(n, n);
    Vector v_k(n);

    std::string gamma_str = "gamma_k = " + std::to_string(c_0);
    if (decrease_gamma)
    {
        gamma_str += " / (3 + k)"; 
    }
    InitDisplay(params, "Contracting Newton Method, " + gamma_str);
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 inv_m * Ax.unaryExpr(&Log_one_exp).sum();
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        double gamma_k = c_0;
        if (decrease_gamma)
        {
            gamma_k /= 3.0 + k;
        }

        g_k = inv_m * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
        H_k = (inv_m * gamma_k) * 
              ((*params.AT_) * Ax.unaryExpr(&Sigmoid_prime).asDiagonal() *
               (*params.A_));
        g_k -= H_k * x_k;


        MinimizeQuadraticOnL2Ball(g_k, H_k, params.R_, params.inner_eps_, &v_k);

        x_k += gamma_k * (v_k - x_k); 
        Ax = (*params.A_) * x_k;
        data_accesses += m;
    }
    FinilizeDisplay(params, *history);
}

void AggregatingNewton(const Parameters& params, History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = m;

    Vector x_k = *params.x_0_;
    Vector Ax = (*params.A_) * x_k;
    Vector g_k(n);
    Matrix H_k(n, n);
    Vector v_k(n);

    Vector l_k = Vector::Constant(n, 0.0);
    Matrix Q_k = Matrix::Constant(n, n, 0.0);

    double A_k = 0;


    InitDisplay(params, "Aggregating Newton Method");
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 inv_m * Ax.unaryExpr(&Log_one_exp).sum();
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        double A_k_new = Cube(k + 1);
        double a_k_new = A_k_new - A_k;
        double gamma_k = a_k_new / A_k_new;

        g_k = (inv_m * a_k_new) * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
        H_k = (inv_m * gamma_k * a_k_new) * 
              ((*params.AT_) * Ax.unaryExpr(&Sigmoid_prime).asDiagonal() *
               (*params.A_));
        g_k -= H_k * x_k;

        // Aggregate new information.
        l_k += g_k;
        Q_k += H_k;

        MinimizeQuadraticOnL2Ball(l_k, Q_k, params.R_, params.inner_eps_, &v_k);

        x_k += gamma_k * (v_k - x_k); 
        Ax = (*params.A_) * x_k;
        A_k = A_k_new;
        data_accesses += m;
    }
    FinilizeDisplay(params, *history);
}

void SVRG(const Parameters& params, const double L_0, 
          const bool variance_reduction,
          History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = 0;

    Vector x_k = *params.x_0_;
    Vector grad_i_k(n);
    Vector T(n);
    Vector full_grad;
    Vector grad_z_i_k;
    Vector z_k;
    double L_k = L_0;

    std::default_random_engine generator;
    generator.seed(RANDOM_SEED);
    std::uniform_int_distribution<int> uniform(0, m - 1);

    InitDisplay(params, variance_reduction ? "SVRG" : "SGD");
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 inv_m * ((*params.A_) * x_k)
                                 .unaryExpr(&Log_one_exp).sum();
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        // Check, whether to recompute the full_gradient.
        if (variance_reduction && Pi(k) == k) 
        {
            full_grad = inv_m * ((*params.AT_) * 
                        ((*params.A_) * x_k).unaryExpr(&Sigmoid));
            z_k = x_k;
            data_accesses += m;
        }

        int i = uniform(generator);
        double a_i_x_k = params.A_->row(i).dot(x_k);
        grad_i_k = Sigmoid(a_i_x_k) * params.A_->row(i);
        data_accesses += 1;

        if (variance_reduction)
        {
            grad_z_i_k = Sigmoid(params.A_->row(i).dot(z_k)) *
                         params.A_->row(i);
            grad_i_k += full_grad - grad_z_i_k;
        }

        T = x_k - 1.0 / L_k * grad_i_k;
        double T_norm = T.norm();
        if (T_norm > params.R_ + EPS) 
        {
            // Project onto the ball.
            T *= params.R_ / T_norm;
        }
        x_k = T;
    }
    FinilizeDisplay(params, *history);
}

void StochasticContractingNewton(const Parameters& params, const double c_0,
                                 const bool decrease_gamma, 
                                 const bool variance_reduction,
                                 const bool hessian_variance_reduction, 
                                 History* history)
{
    const auto start_time = std::chrono::system_clock::now();
    auto last_logging_time = start_time;
    auto last_display_time = start_time;

    const int n = params.A_->cols();
    const int m = params.A_->rows();
    const double inv_m = 1.0 / m;
    int data_accesses = 0;

    Vector x_k = *params.x_0_;

    SMatrix A_batch;
    SMatrix AT_batch;
    std::vector<Eigen::Triplet<double>> A_batch_triplets;
    std::vector<Eigen::Triplet<double>> AT_batch_triplets;
    
    Vector g_k(n);
    Matrix H_k(n, n);
    Vector v_k(n);

    Vector sigma_prime;
    Vector Ax;
    Vector full_grad;
    Matrix full_Hess;
    Vector z_k;    
    
    std::default_random_engine generator;
    generator.seed(RANDOM_SEED);
    std::vector<int> obj_indices(m);
    std::iota(obj_indices.begin(), obj_indices.end(), 0);
    std::vector<int> batch;

    std::string method_name = "Stochastic Newton";
    if (variance_reduction)
    {
        method_name += (hessian_variance_reduction ? " (HVR)" : " (VR)");
    }
    std::string gamma_str = "gamma_k = " + std::to_string(c_0);
    if (decrease_gamma)
    {
        gamma_str += " / (3 + k)"; 
    }
    InitDisplay(params, method_name + ", " + gamma_str);
    for (int k = 0; k < params.n_iters_ + 1; ++k) 
    {
        bool to_finish;
        UpdateHistory(params, start_time, k, data_accesses,
                      [&]() { 
                          return (x_k.norm() > params.R_ + REGION_TOLERANCE) ? 
                                 INF : 
                                 inv_m * ((*params.A_) * x_k)
                                 .unaryExpr(&Log_one_exp).sum();
                      },
                      &last_logging_time,
                      &last_display_time,
                      history,
                      &to_finish);
        if (to_finish)
        {
            break;
        }

        // Check, whether to recompute the full_gradient and full_Hess.
        if (variance_reduction && Pi(k) == k) 
        {
            Ax = (*params.A_) * x_k;
            full_grad = inv_m * ((*params.AT_) * Ax.unaryExpr(&Sigmoid));
            if (hessian_variance_reduction)
            {
                full_Hess = inv_m * ((*params.AT_) * 
                            Ax.unaryExpr(&Sigmoid_prime).asDiagonal() *
                            (*params.A_));
            }
            z_k = x_k;
            data_accesses += m;
        }

        long long k_sqr = static_cast<long long>(k + 1) * (k + 1);
        int batch_size = (k_sqr < m) ? static_cast<int>(k_sqr) : m;
        if (batch_size == m)
        {
            std::cerr << "W: batch_size equals m" << std::endl;
        }
        batch.clear();
        std::sample(obj_indices.begin(), obj_indices.end(), 
                    std::back_inserter(batch), batch_size, generator);

        double gamma_k = c_0;
        if (decrease_gamma)
        {
            gamma_k /= 3.0 + k;
        }

        g_k = variance_reduction ? full_grad : Vector::Zero(n);
        if (variance_reduction && hessian_variance_reduction) 
        {
            H_k = gamma_k * full_Hess;
        }
        else
        {
            H_k = Matrix::Zero(n, n);
        }

        A_batch_triplets.clear();
        AT_batch_triplets.clear();   
        sigma_prime = Vector::Zero(batch_size);
        for (int batch_i = 0; batch_i < batch_size; ++batch_i)
        {
            int i = batch[batch_i];
            double sigma = Sigmoid(params.A_->row(i).dot(x_k));
            g_k += (sigma / batch_size) * params.A_->row(i);
            const auto& data_i = (*params.data_)[i];
            for (auto index_value : data_i)
            {
                A_batch_triplets.push_back(Eigen::Triplet<double>(
                    batch_i, index_value.first, index_value.second));
                AT_batch_triplets.push_back(Eigen::Triplet<double>(
                    index_value.first, batch_i, index_value.second));
            }
            sigma_prime(batch_i) = sigma * (1 - sigma) * gamma_k / batch_size;
            if (variance_reduction)
            {
                double sigma_z = Sigmoid(params.A_->row(i).dot(z_k));
                g_k -= (sigma_z / batch_size) * params.A_->row(i);
                if (hessian_variance_reduction)
                {
                    sigma_prime(batch_i) -= sigma_z * (1 - sigma_z) * 
                                            gamma_k / batch_size;
                }
            }
        }
        data_accesses += batch_size;

        A_batch = SMatrix(batch_size, n);
        AT_batch = SMatrix(n, batch_size);
        A_batch.setFromTriplets(A_batch_triplets.begin(), 
                                A_batch_triplets.end());
        AT_batch.setFromTriplets(AT_batch_triplets.begin(), 
                                 AT_batch_triplets.end());
        H_k += A_batch.transpose() * sigma_prime.asDiagonal() * A_batch;

        g_k -= H_k * x_k;
        MinimizeQuadraticOnL2Ball(g_k, H_k, params.R_, params.inner_eps_, &v_k);
        x_k += gamma_k * (v_k - x_k);

    }
    FinilizeDisplay(params, *history);    
}

}  // namespace optimize
