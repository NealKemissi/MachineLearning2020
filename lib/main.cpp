#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

extern "C" {
    DLLEXPORT int my_add(int x, int y) {
        return x + y;
    }

    DLLEXPORT int my_mul(int x, int y) {
        return x * y;
    }

    DLLEXPORT int * linear_model_create(int input_dim) {
        int k = 0;
        auto array = new int[input_dim + 1];
        int random;
        for (int index = 0; index < input_dim+1; index++)
        {
            array[k] = rand()%100+1;
            k++;
        }
        return array;
    }

    DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size) {
        Vector3d v(model);
        Vector3d w(inputs);
        return v.tail(1).dot(w) + model[0];
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size) {
        return linear_model_predict_regression(model, inputs, inputs_size) >=0 ? 1.0 : -1.0;
    }

    DLLEXPORT void linear_model_train_classification(double* model,
            double** dataset_inputs,
            int dataset_length,
            int inputs_size,
            double* dataset_expected_outputs,
            int outputs_size,
            int iterations_count,
            bool should_plot_results,
            float alpha = 10) {
        // TODO : train Rosenbalt
        for (int it = 0; it < iterations_count; it++) {
            int k = rand() % dataset_length;
            double g_x_k = linear_model_predict_classification(model, dataset_inputs[k], dataset_length);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad * 1;
            for (int i = 0; i < dataset_length; i++) {
                model[i + 1] += grad * dataset_inputs[k][i];
            }

        }
    }
//
//    DLLEXPORT double linear_model_train_regression(double* model,
//            double* dataset_inputs,
//            int dataset_length,
//            int inputs_size,
//            double* dataset_expected_outputs,
//            int outputs_size,
//            int iterations_count,
//            float alpha) {
//    // TODO : train PseudoInverse moore
//    }
}