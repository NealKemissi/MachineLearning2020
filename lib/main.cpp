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
        for (int index = 0; index < input_dim+1; index++)
        {
            array[k] = rand()%100+1;
            k++;
        }
        return array;
    }

    DLLEXPORT double linear_model_predict_regression(double* model, double* inputs) {
        Vector3d v(model);
        Vector3d w(inputs);
        return v.tail(1).dot(w) + model[0];
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* inputs) {
        return linear_model_predict_regression(model, inputs) >=0 ? 1.0 : -1.0;
    }

    DLLEXPORT void linear_model_train_classification(double* model,
            double* dataset_inputs,
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
            int position = inputs_size * k;
            double g_x_k = linear_model_predict_classification(model, &dataset_inputs[position]);
            double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
            model[0] += grad * 1;
            for (int i = 0; i < dataset_length; i++) {
                model[i + 1] += grad * dataset_inputs[position + i];
            }
        }
    }

DLLEXPORT void linear_model_train_regression(double* model,
        double* dataset_inputs,
        int dataset_length,
        int inputs_size,
        double* dataset_expected_outputs,
        int outputs_size) {
    // TODO : train PseudoInverse moore
    MatrixXd x(dataset_length, inputs_size+1);
    MatrixXd y(dataset_length, 1);
    for(int li = 0; li < dataset_length; li++) {
        x(li, 0) = 1;
        y(li, 0) = dataset_expected_outputs[li];
        for(int col = 1; col < (inputs_size+1); col++) {
            int i = (li * inputs_size) +(col - 1);
            x(li, col) = dataset_inputs[i];
        }
    }

    MatrixXd w = (((x.transpose() * x).inverse()) * (x.transpose())) * y;

    for(int i = 0; i <= inputs_size; i++) {
        model[i] = w(i, 0);
    }
}

    DLLEXPORT double* lines_sum(double* lines, int lines_count, int line_size) {
        auto sums = new double[lines_count];
        for(auto l = 0; l < lines_count; l++) {
            auto line = lines + l*line_size;
            auto sum = 0.0;
            for(auto i = 0; i < lines_count; i++) {
                sum+= line[i];
            }
            sums[l] = sum;
        }
        return sums;
    }

    struct MLP {
        int* npl;
        int npl_size;
        double*** w; //layer, i, j
        double** x;
        double** deltas;
    };

    // mlp_model_create([2,3, 4, 1], 4)
    DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size) {
        return NULL;
    }
}