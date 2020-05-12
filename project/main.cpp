#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>

extern "C" {
    DLLEXPORT int my_add(int x, int y);
    DLLEXPORT int * linear_model_create(int input_dim);
    DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size);
    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size);
    DLLEXPORT void linear_model_train_classification(double* model,
        double** dataset_inputs,
        int dataset_length,
        int inputs_size,
        double* dataset_expected_outputs,
        int outputs_size,
        int iterations_count,
        bool should_plot_results,
        float alpha = 10);
}

int main() {
    std::cout << my_add(3, 7) << std::endl;
    return 0;
}
