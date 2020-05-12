#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>

extern "C" {
    DLLEXPORT int my_add(int x, int y);
    DLLEXPORT int * linear_model_create(int input_dim);

DLLEXPORT double linear_model_predict_regression(double* model, double* inputs);
DLLEXPORT double linear_model_predict_classification(double* model, double* inputs);
DLLEXPORT void linear_model_train_classification(double* model,
    double** dataset_inputs,
    int dataset_length,
    int inputs_size,
    double* dataset_expected_outputs,
    int outputs_size,
    int iterations_count,
    bool should_plot_results,
    float alpha = 10);
DLLEXPORT void linear_model_train_regression(double* model,
        double* dataset_inputs,
        int dataset_length,
        int inputs_size,
        double* dataset_expected_outputs,
        int outputs_size);
//DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size);
}

int main() {
    std::cout << my_add(3, 7) << std::endl;
    return 0;
}
