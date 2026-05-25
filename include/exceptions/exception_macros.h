#ifndef EXCEPTION_MACROS_H
#define EXCEPTION_MACROS_H

#include "ml_exception.h"

// Macro per controlli rapidi
#define ML_CHECK_FITTED(condition, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::NotFittedException("Model not fitted. Call fit() first.", model_type); \
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
 * @brief Check that X and y have compatible dimensions for training (Matrix version)
 * @param X_rows Number of rows in X (samples)
 * @param y_rows Number of rows in y (samples)
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_XY_SIZE_MATRIX(X_rows, y_rows, model_type) \
    do { \
        const auto x_rows_val = (X_rows); \
        const auto y_rows_val = (y_rows); \
        if (x_rows_val != y_rows_val) { \
            throw ml_exception::DimensionMismatchException( \
                "X and y rows", \
                static_cast<int>(y_rows_val), \
                -1, \
                static_cast<int>(x_rows_val), \
                -1, \
                model_type \
            ); \
        } \
    } while (0)

/**
 * @brief Check that X and y have compatible dimensions (columns for matrix)
 * @param X_cols Number of columns in X
 * @param y_cols Number of columns in y
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_XY_COLS(X_cols, y_cols, model_type) \
    do { \
        const auto x_cols_val = (X_cols); \
        const auto y_cols_val = (y_cols); \
        if (x_cols_val != y_cols_val) { \
            throw ml_exception::DimensionMismatchException( \
                "X and y columns", \
                -1, \
                static_cast<int>(y_cols_val), \
                -1, \
                static_cast<int>(x_cols_val), \
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

/**
 * @brief Check that a value is finite (not NaN or Inf)
 * @param value The value to check
 * @param model_type Model name for error reporting
 * @param function_name Function name for error reporting
 */
#define ML_CHECK_FINITE(value, model_type, function_name) \
    do { \
        const auto val = (value); \
        if (std::isnan(val) || std::isinf(val)) { \
            throw ml_exception::MLException( \
                std::string(model_type) + "::" + function_name + \
                ": loss value is " + (std::isnan(val) ? "NaN" : "Inf")); \
        } \
    } while(0)

/**
 * @brief Check that a matrix has no NaN values
 * @param matrix The matrix to check
 * @param model_type Model name for error reporting
 * @param function_name Function name for error reporting
 */
#define ML_CHECK_NO_NAN(matrix, model_type, function_name) \
    do { \
        const auto& mat = (matrix); \
        if (mat.array().isNaN().any()) { \
            throw ml_exception::MLException( \
                std::string(model_type) + "::" + function_name + \
                ": matrix contains NaN values"); \
        } \
    } while(0)

/**
 * @brief Check that a matrix has no Inf values
 * @param matrix The matrix to check
 * @param model_type Model name for error reporting
 * @param function_name Function name for error reporting
 */
#define ML_CHECK_NO_INF(matrix, model_type, function_name) \
    do { \
        const auto& mat = (matrix); \
        if (mat.array().isInf().any()) { \
            throw ml_exception::MLException( \
                std::string(model_type) + "::" + function_name + \
                ": matrix contains Inf values"); \
        } \
    } while(0)

/**
 * @brief Check that a matrix has no NaN or Inf values
 * @param matrix The matrix to check
 * @param model_type Model name for error reporting
 * @param function_name Function name for error reporting
 */
#define ML_CHECK_VALID_MATRIX(matrix, model_type, function_name) \
    do { \
        ML_CHECK_NO_NAN(matrix, model_type, function_name); \
        ML_CHECK_NO_INF(matrix, model_type, function_name); \
    } while(0)

/**
 * @brief Check that prediction values are in valid range [0,1] for probabilities
 * @param matrix The matrix to check
 * @param model_type Model name for error reporting
 * @param function_name Function name for error reporting
 */
#define ML_CHECK_PROBABILITY_RANGE(matrix, model_type, function_name) \
    do { \
        const auto& mat = (matrix); \
        if ((mat.array() < 0.0).any() || (mat.array() > 1.0).any()) { \
            throw ml_exception::MLException( \
                std::string(model_type) + "::" + function_name + \
                ": probabilities must be in range [0, 1]"); \
        } \
    } while(0)

/**
 * @brief Check that learning rate is valid
 * @param lr Learning rate value
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_LEARNING_RATE(lr, model_type) \
    ML_CHECK_PARAM((lr) > 0.0, "learning_rate", "must be > 0", model_type)

/**
 * @brief Check that batch size is valid
 * @param bs Batch size value
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_BATCH_SIZE(bs, model_type) \
    ML_CHECK_PARAM((bs) > 0, "batch_size", "must be > 0", model_type)

/**
 * @brief Check that epochs is valid
 * @param ep Epochs value
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_EPOCHS(ep, model_type) \
    ML_CHECK_PARAM((ep) > 0, "epochs", "must be > 0", model_type)

/**
 * @brief Check that validation split is valid
 * @param split Validation split value
 * @param model_type Model name for error reporting
 */
#define ML_CHECK_VALIDATION_SPLIT(split, model_type) \
    ML_CHECK_PARAM((split) >= 0.0 && (split) < 1.0, "validation_split", \
                   "must be in [0, 1)", model_type)

// Macro per IO
#define ML_THROW_IO_ERROR(filename, operation, model_type) \
    throw ml_exception::IOException(filename, operation, model_type)

#define ML_THROW_DIMENSION_MISMATCH(operation, expected_rows, expected_cols, \
                                    actual_rows, actual_cols, model_type) \
    throw ml_exception::DimensionMismatchException( \
        operation, expected_rows, expected_cols, \
        actual_rows, actual_cols, model_type)

#define ML_THROW_PARAMETER_ERROR(param_name, requirement, model_type) \
    throw ml_exception::InvalidParameterException( \
        param_name, requirement, model_type)

#define ML_THROW_FITTING_ERROR(model_type, message) \
    throw ml_exception::NotFittedException( \
        std::string(model_type) + ": " + message)

#define ML_THROW_VALIDATION_ERROR(message, model_type) \
    throw ml_exception::ValidationException( \
        std::string(model_type) + ": " + message)

#define ML_THROW_NOT_IMPLEMENTED_ERROR(feature, model_type) \
    throw ml_exception::MLException( \
        std::string(model_type) + ": " + feature + " not implemented")

/**
 * @brief Throw an exception for invalid shape
 * @param operation Operation name
 * @param expected Expected shape description
 * @param actual Actual shape description
 * @param model_type Model name
 */
#define ML_THROW_SHAPE_ERROR(operation, expected, actual, model_type) \
    throw ml_exception::MLException( \
        std::string(model_type) + "::" + operation + \
        ": expected shape " + expected + ", got " + actual)

/**
 * @brief Throw an exception for unsupported operation
 * @param operation Operation name
 * @param reason Reason for not supporting
 * @param model_type Model name
 */
#define ML_THROW_UNSUPPORTED_ERROR(operation, reason, model_type) \
    throw ml_exception::MLException( \
        std::string(model_type) + ": " + operation + " not supported (" + reason + ")")


// Aggiungi alla fine del file exception_macros.h

#define ML_THROW_FILE_NOT_FOUND(filename, model_type) \
    throw ml_exception::FileNotFoundException(filename, model_type)

#define ML_CHECK_FILE_EXISTS(filename, model_type) \
    do { \
        if (!std::filesystem::exists(filename)) \
            throw ml_exception::FileNotFoundException(filename, model_type); \
    } while(0)

#define ML_THROW_SERIALIZATION_ERROR(reason, model_type) \
    throw ml_exception::SerializationException(std::string(reason), std::string(model_type))

#define ML_THROW_SERIALIZATION_ERROR_FILE(filename, reason, model_type) \
    throw ml_exception::SerializationException(std::string(reason), std::string(model_type), std::string(filename))

#define ML_THROW_DESERIALIZATION_ERROR(reason, model_type) \
    throw ml_exception::DeserializationException(std::string(reason), std::string(model_type))

#define ML_THROW_DESERIALIZATION_ERROR_FILE(filename, reason, model_type) \
    throw ml_exception::DeserializationException(std::string(reason), std::string(model_type), std::string(filename))

#define ML_CHECK_DESERIALIZATION(condition, reason, model_type) \
    do { \
        if (!(condition)) \
            throw ml_exception::DeserializationException(std::string(reason), std::string(model_type)); \
    } while(0)

#define ML_THROW_MATH_ERROR(operation, reason) \
    throw ml_exception::MathException(operation, reason, "Loss")

#define ML_CHECK_MATH(condition, operation, reason) \
    do { \
        if (!(condition)) \
            ML_THROW_MATH_ERROR(operation, reason); \
    } while(0)

#define ML_CHECK_STATE(condition, function_name, message) \
    do { \
        if (!(condition)) \
            throw ml_exception::MLException( \
                std::string(function_name) + ": " + std::string(message)); \
    } while(0)

#endif // EXCEPTION_MACROS_H