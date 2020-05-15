#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <time.h>
#include <ctime>
#include <cstdlib>
#include <Eigen/Dense>


using namespace Eigen;

struct MLP {
    int* npl;
    int npl_size;
    double*** w; //layer, i, j
    double** x;
    double** deltas;
};

void mlp_propagation(MLP* model, double* inputs, bool regression) {
    for(int j = 1; j < model->npl[0] + 1; j++) {
        model->x[0][j] = inputs[j - 1];
    }

    for(int l = 1; l < model->npl_size; l++) {
        for(int j = 1; j < model->npl[l] + 1; j++) {
            auto sum = 0.0;
            for(int i = 0; i < model->npl[l-1]+1; i++) {
                sum += model->w[l][i][j] * model->x[l-1][i];
            }
            model->x[l][j] = ((l == model->npl_size -1) && regression) ? sum : tanh(sum);
        }
    }
}

double* mlp_propagation_and_extract_result(MLP* model, double* inputs, bool regression) {
    mlp_propagation(model, inputs, regression);

    auto res = new double[model->npl[model->npl_size - 1]];
    for(int j = 1; j < model->npl[model->npl_size - 1] + 1; j++) {
        res[j - 1] = model->x[model->npl_size - 1][j];
    }
    return res;
}

extern "C" {
DLLEXPORT int my_add(int x, int y) {
    return x + y;
}
DLLEXPORT int  my_test(int x, int y, int z){
    return x + y + z;
}

DLLEXPORT int my_mul(int x, int y) {
    return x * y;
}

DLLEXPORT double * linear_model_create(int input_dim) {

    auto array = new double[input_dim + 1];
    for (int index = 0; index < input_dim+1; index++)
    {
        array[index] = ((double)rand())/RAND_MAX;
    }
    return array;
}
/*DLLEXPORT double linear_model_predict_regression(double* model, double* inputs) {
    Vector2d v(model);
    Vector2d w(inputs);
    return v.tail(1).dot(w) + model[0];
}
DLLEXPORT double linear_model_predict_classification(double* model, double* inputs) {
    return linear_model_predict_regression(model, inputs) >=0 ? 1.0 : -1.0;
}*/
DLLEXPORT double linear_model_predict_regression(double* model, double* inputs, int inputs_size) {
    double val = 0;
    for (int i = 0; i < inputs_size; i++) {
        val += model[i+1] * inputs[i];
    }
    return val + model[0];
}
DLLEXPORT double linear_model_predict_classification(double* model, double* inputs, int inputs_size) {
    return linear_model_predict_regression(model, inputs, inputs_size) >= 0 ? 1.0 : -1.0;
}

DLLEXPORT void linear_model_train_classification(double* model,
                                                 double* dataset_inputs,
                                                 int dataset_length,
                                                 int inputs_size,
                                                 double* dataset_expected_outputs,
                                                 int outputs_size,
                                                 int iterations_count,
                                                 float alpha) {
    // TODO : train Rosenbalt
    for (int it = 0; it < iterations_count; it++) {
        int k = ((double)rand())/RAND_MAX * dataset_length;
        int position = inputs_size * k;
        double g_x_k = linear_model_predict_classification(model, &dataset_inputs[position], inputs_size);//
        double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
        model[0] += grad * 1;
        for (int i = 0; i < inputs_size; i++) {
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



// mlp_model_create([2,3, 4, 1], 4)
DLLEXPORT MLP* mlp_model_create(int* npl, int npl_size) {
    MLP* mlp = new MLP;
    mlp->npl = new int[npl_size];
    for(int i = 0; i < npl_size;i++) {
        mlp->npl[i] = npl[i];
    }
    mlp->npl_size = npl_size;

    mlp->w = new double**[npl_size];
    for (int l = 1; l < npl_size; l++) {
        mlp->w[l] = new double*[npl[l-1] + 1];

        for (int i = 0; i < npl[l-1] + 1; i++) {
            mlp->w[l][i] = new double[npl[l] + 1];

            for (int j = 1; j < npl[l] + 1; j++) {
                mlp->w[l][i][j] = ((double) rand()) / RAND_MAX * 2.0 - 1.0;
            }
        }
    }
    return mlp;
}

DLLEXPORT double* mlp_model_predict_regression(struct MLP* model, double* inputs) {
    return mlp_propagation_and_extract_result(model, inputs, true);
}

DLLEXPORT double* mlp_model_predict_classification(struct MLP* model, double* inputs) {
    return mlp_propagation_and_extract_result(model, inputs, false);
}

DLLEXPORT void mlp_model_train_classification(struct MLP* model,
                                                double* dataset_inputs,
                                                int dataset_length,
                                                int inputs_size,
                                                double* dataset_expected_outputs,
                                                int outputs_size,
                                                int iterations_count,
                                                double alpha) {
    for(int it = 0; it < iterations_count; it++) {
        auto k = (int)floor(((double)std::min(rand(), RAND_MAX - 1)) / RAND_MAX * dataset_length);

        auto inputs = dataset_inputs + k * inputs_size;
        auto expected_outputs = dataset_expected_outputs + k * outputs_size;

        mlp_propagation(model, inputs, true);

        for(int j = 1; j < model->npl[model->npl_size-1] + 1; j++) {
            model->deltas[model->npl_size -1][j] = (1 - pow(model->x[model->npl_size - 1][j], 2))
                    * (model->x[model->npl_size - 1][j] - expected_outputs[j - 1]);
        }

        for(int l = model->npl_size-1; l >= 2; l--) {
            for(int i = 1; i < model->npl[l - 1] + 1; i++) {
                auto sum = 0.0;
                for(int j = 1; j < model->npl[l] + 1; j++) {
                    sum += model->w[l][i][j] * model->deltas[l][j];
                }
                model->deltas[l-1][i] = (1 - pow(model->x[l-1][i],2)) * sum;
            }
        }

        for(int l = 1; l < model->npl_size; l++) {
            for(int i = 0; i < model->npl[l-1]+1; i++) {
                for(int j = 1; j < model->npl[l] + 1; j++) {
                    model->w[l][i][j] -= alpha * model->x[l-1][i] * model->deltas[l][j];
                }
            }
        }
    }
}


DLLEXPORT void mlp_model_train_regression(struct MLP* model,
                                          double* dataset_inputs,
                                          int dataset_length,
                                          int inputs_size,
                                          double* dataset_expected_outputs,
                                          int outputs_size,
                                          int iterations_count,
                                          double alpha) {
    for(int it = 0; it < iterations_count; it++) {
        auto k = (int)floor(((double)std::min(rand(), RAND_MAX - 1)) / RAND_MAX * dataset_length);

        auto inputs = dataset_inputs + k * inputs_size;
        auto expected_outputs = dataset_expected_outputs + k * outputs_size;

        mlp_propagation(model, inputs, false);

        for(int j = 1; j < model->npl[model->npl_size-1] + 1; j++) {
            model->deltas[model->npl_size -1][j] = (1 - pow(model->x[model->npl_size - 1][j], 2))
                                                   * (model->x[model->npl_size - 1][j] - expected_outputs[j - 1]);
        }

        for(int l = model->npl_size-1; l >= 2; l--) {
            for(int i = 1; i < model->npl[l - 1] + 1; i++) {
                auto sum = 0.0;
                for(int j = 1; j < model->npl[l] + 1; j++) {
                    sum += model->w[l][i][j] * model->deltas[l][j];
                }
                model->deltas[l-1][i] = (1 - pow(model->x[l-1][i],2)) * sum;
            }
        }

        for(int l = 1; l < model->npl_size; l++) {
            for(int i = 0; i < model->npl[l-1]+1; i++) {
                for(int j = 1; j < model->npl[l] + 1; j++) {
                    model->w[l][i][j] -= alpha * model->x[l-1][i] * model->deltas[l][j];
                }
            }
        }
    }
}


}