#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
    DLLEXPORT int my_add(int x, int y) {
        return x + y;
    }

    DLLEXPORT int my_mul(int x, int y) {
        return x * y;
    }

    // TODO : retourner Eigen np.ndarray
    DLLEXPORT int linear_model_create(int input_dim) {
        return 0;
    }

    DLLEXPORT double linear_model_predict_regression() {
        return 0;
    }

    DLLEXPORT double linear_model_predict_classification() {
        return 0;
    }

    DLLEXPORT void linear_model_train_classification() {
    }
}