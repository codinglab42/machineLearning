#ifndef EXCEPTION_MACROS_H
#define EXCEPTION_MACROS_H

#include "dimension_exception.h"
#include "fitting_exception.h"
#include "validation_exception.h"

// Macro per controlli rapidi
#define ML_CHECK_FITTED(condition, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::NotFittedException(model_type); \
    } while(0)

#define ML_CHECK_DIMENSIONS(actual_rows, expected_rows, \
                           actual_cols, expected_cols, \
                           operation, model_type) \
    do { \
        if ((actual_rows) != (expected_rows) || \
            (actual_cols) != (expected_cols)) \
            throw ml_exception::DimensionMismatchException( \
                operation, expected_rows, expected_cols, \
                actual_rows, actual_cols, model_type); \
    } while(0)

/**
 * @brief Check that X and y have compatible dimensions for training
 * @param X_rows Number of rows in X (samples)
 * @param y_size Size of y vector (targets)
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_XY_SIZE(X_rows, y_size, model_type) \
    do { \
        const auto x_rows_val = (X_rows); \
        const auto y_size_val = (y_size); \
        if (x_rows_val != y_size_val) { \
            throw ml_exception::DimensionMismatchException( \
                "X and y rows",                /* operation */ \
                static_cast<int>(y_size_val),  /* expected_rows = y.size() */ \
                1,                             /* expected_cols (dummy) */ \
                static_cast<int>(x_rows_val),  /* actual_rows = X.rows() */ \
                1,                             /* actual_cols (dummy) */ \
                model_type \
            ); \
        } \
    } while (0)
    

/**
 * @brief Check that input matrix has expected number of features
 * @param actual_cols Number of columns in input matrix
 * @param expected_cols Expected number of features
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_FEATURE_DIMENSIONS(actual_cols, expected_cols, model_type) \
    do { \
        const auto actual = (actual_cols); \
        const auto expected = (expected_cols); \
        if (actual != expected) { \
            throw ml_exception::FeatureMismatchException( \
                static_cast<int>(expected), \
                static_cast<int>(actual), \
                model_type \
            ); \
        } \
    } while (0)
#define ML_CHECK_FEATURES(actual_features, expected_features, model_type) \
    do { \
        if ((actual_features) != (expected_features)) \
            throw ml_exception::FeatureMismatchException( \
                expected_features, actual_features, model_type); \
    } while(0)

#define ML_CHECK_PARAM(condition, param_name, requirement, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::InvalidParameterException( \
                param_name, requirement, model_type); \
    } while(0)

#define ML_CHECK_NOT_EMPTY(data, data_name, model_type) \
    do { \
        if ((data).rows() == 0 || (data).cols() == 0) \
            throw ml_exception::EmptyDatasetException(data_name, model_type); \
    } while(0)

#define ML_THROW_IO_ERROR(filename, operation, model_type) \
    throw ml_exception::IOException(filename, operation, model_type)

#endif