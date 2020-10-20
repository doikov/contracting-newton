
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "optimize.h"

using namespace std;
using optimize::History;
using optimize::Matrix;
using optimize::Parameters;
using optimize::SMatrix;
using optimize::Vector;

template <class T>
void ExtractArg(int argc, char* argv[], 
                const string& key, const T& default_value, T* value)
{
    *value = default_value;
    for (int i = 1; i < argc; ++i) 
    {
        string s = argv[i];
        size_t pos = s.find("=");
        if (pos != string::npos)
        {
            string key_part = s.substr(0, pos); 
            int key_part_length = key_part.length();
            int j = 0;
            // Skip '-' in prefix, if any.
            while (j < key_part_length)
            {
                if (key_part[j] != '-')
                {
                    break;
                }
                ++j;
            }
            key_part = key_part.substr(j);
            if (key_part == key)
            {
                stringstream ss(s.substr(pos + 1));
                // Record the last found parameter value.
                ss >> *value;
            }
        }
    }
}

bool ReadData(const string& filename, vector<vector<pair<int, double>>>* d,
              SMatrix* A, SMatrix* AT) 
{
    ifstream input;
    input.open(filename);
    if (!input) {
        cerr << "E: Unable to open file " << filename << endl;
        return false;
    }
    int m = 0;  // Number of training examples.
    int n = 0;  // Number of features.
    int nnz = 0;  // Number of nonzero elements.
    auto& data = *d;
    string buffer;
    vector<int> labels;
    while (getline(input, buffer)) 
    {
        ++m;
        data.push_back(vector<pair<int, double>>());
        stringstream ss(buffer);
        double label;
        ss >> label;
        labels.push_back(static_cast<int>(label));
        string key_value;
        while (ss >> key_value) 
        {
            size_t idx;
            int key = stoi(key_value, &idx);
            double value = stod(key_value.substr(idx + 1));
            data.back().push_back({key - 1, value});
            n = max(n, key);
            ++nnz;
        }
    }
    input.close();
    if (m == 0 || n == 0 || nnz == 0) {
        cerr << "E: empty input data" << endl;
        return false;
    } 
    unordered_set<int> merged_labels(labels.begin(), labels.end());
    if (merged_labels.size() != 2) {
        cerr << "W: the number of classes is " << merged_labels.size() << endl;
    }
    int first_label = labels[0];
    // Initialize triplet lists first.
    std::vector<Eigen::Triplet<double>> A_triplets(nnz);
    std::vector<Eigen::Triplet<double>> AT_triplets(nnz);
    for (int i = 0; i < m; ++i) {
        int label = (labels[i] == first_label ? 1 : -1);
        for (auto& key_value : data[i]) {
            key_value.second *= -label;
            A_triplets.push_back(Eigen::Triplet<double>(
                i, key_value.first, key_value.second));
            AT_triplets.push_back(Eigen::Triplet<double>(
                key_value.first, i, key_value.second));
        }
    }
    *A = SMatrix(m, n);
    *AT = SMatrix(n, m);
    A->setFromTriplets(A_triplets.begin(), A_triplets.end());
    AT->setFromTriplets(AT_triplets.begin(), AT_triplets.end());
    return true;
}

int main(int argc, char* argv[])
{
    ios_base::sync_with_stdio(false);

    Parameters params;
    string dataset;
    string output_name;
    string experiment;
    double L_0;
    double c_0;
    ExtractArg<string>(argc, argv, "dataset", "data/w8a.txt", &dataset);
    ExtractArg(argc, argv, "R", 500.0, &params.R_);
    ExtractArg(argc, argv, "n_iters", 10000, &params.n_iters_);
    ExtractArg(argc, argv, "max_time", 10.0, &params.max_time_);
    ExtractArg(argc, argv, "logging_iter_period", 1, 
               &params.logging_iter_period_);
    ExtractArg(argc, argv, "logging_time_period", 0.0, 
               &params.logging_time_period_);
    ExtractArg(argc, argv, "display", true, &params.display_);
    ExtractArg(argc, argv, "display_time_period", 1.0,
               &params.display_time_period_);
    ExtractArg<string>(argc, argv, "output_name", "output/result", 
                       &output_name);
    ExtractArg<string>(argc, argv, "experiment", "basic", &experiment);
    ExtractArg(argc, argv, "L_0", 1.0, &L_0);
    ExtractArg(argc, argv, "c_0", 1.0, &c_0);
    ExtractArg(argc, argv, "inner_eps", 1e-5, &params.inner_eps_);

    // We duplicate input data, for faster performance of the methods.
    vector<vector<pair<int, double>>> data;
    SMatrix A;
    SMatrix AT;
    if (!ReadData(dataset, &data, &A, &AT))
    {
        return 0;
    }

    int m = A.rows();
    int n = A.cols();
    Vector x_0 = Vector::Constant(n, 0.0);
    params.data_ = &data;
    params.A_ = &A;
    params.AT_ = &AT;
    params.x_0_ = &x_0;

    cout << "Dataset: " << dataset << "(m = " << m << ", n = " << n << ").\n"
         << "R = " << params.R_ << endl;

    if (experiment == "basic")
    {
        History gm_history;
        GradientMethod(params, L_0, true, &gm_history);
        gm_history.dump(output_name + "_GM.csv");

        History fgm_history;
        FastGradientMethod(params, L_0, true, &fgm_history);
        fgm_history.dump(output_name + "_FGM.csv");

        History fw_history;
        FrankWolfe(params, c_0, true, &fw_history);
        fw_history.dump(output_name + "_FW.csv");

        History contr_newton_history;
        ContractingNewton(params, c_0, true, &contr_newton_history);
        contr_newton_history.dump(output_name + "_Contr_Newton.csv");

        History aggr_newton_history;
        AggregatingNewton(params, &aggr_newton_history);
        aggr_newton_history.dump(output_name + "_Aggr_Newton.csv");
    }
    else if (experiment == "stochastic")
    {
        History sgd_history;
        SVRG(params, L_0, false, &sgd_history);
        sgd_history.dump(output_name + "_SGD.csv");
    
        History svrg_history;
        SVRG(params, L_0, true, &svrg_history);
        svrg_history.dump(output_name + "_SVRG.csv");

        History stoch_contr_newton_history;
        StochasticContractingNewton(params, c_0, true, false, false,
                                    &stoch_contr_newton_history);
        stoch_contr_newton_history.dump(output_name + 
                                        "_Stoch_Contr_Newton.csv");

        History stoch_vr_contr_newton_history;
        StochasticContractingNewton(params, c_0, true, true, false,
                                    &stoch_vr_contr_newton_history);
        stoch_vr_contr_newton_history.dump(output_name + 
                                           "_Stoch_VR_Contr_Newton.csv");
    }
    else if (experiment == "stochastic_newton_variations")
    {
        // Variance reduction for both gradients and Hessians.
        History stoch_hvr_contr_newton_history;
        StochasticContractingNewton(params, c_0, true, true, true,
                                    &stoch_hvr_contr_newton_history);
        stoch_hvr_contr_newton_history.dump(output_name + 
                                            "_Stoch_HVR_Contr_Newton.csv");

        // gamma_k = 1.0 (classical Newton step)
        History stoch_classic_newton_history;
        StochasticContractingNewton(params, 1.0, false, false, false,
                                    &stoch_classic_newton_history);
        stoch_classic_newton_history.dump(output_name + 
                                  "_Stoch_Classic_Newton.csv");
    }
    else
    {
        cerr << "E: Unknown experiment name " << experiment << endl; 
    }
    
    return 0;
}
