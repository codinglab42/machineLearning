#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>


#include "components/layers/layer_factory.h"
#include "models/estimator.h"
#include "models/linear_regression.h"
#include "models/logistic_regression.h"
#include "utils/math_utils.h"
#include "utils/scaler_factory.h"
#include "models/neural_network.h"

// Includi le eccezioni
#include "exceptions/ml_exception.h"
#include "exceptions/dimension_exception.h"
#include "exceptions/fitting_exception.h"
#include "exceptions/io_exception.h"
#include "exceptions/validation_exception.h"

namespace py = pybind11;


template<typename Model>
void add_fit_wrapper(py::class_<Model, models::Estimator, std::shared_ptr<Model>>& cls) {
    cls.def("fit", 
        [](Model& model, const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) -> void {
            std::cout << "\n*** DEBUG: fit_wrapper ATTIVATO ***" << std::endl;
            std::cout << "*** Type: " << typeid(Model).name() << std::endl;
            std::cout << "*** X: " << X.rows() << "x" << X.cols() << std::endl;
            std::cout << "*** y: " << y.rows() << "x" << y.cols() << std::endl;
            
            // Verifica che y sia (4,1) o (1,4)
            if (y.cols() == 1) {
                std::cout << "*** Rilevato y come colonna (n,1) = " << y.rows() << "x1" << std::endl;
                
                // Costruisci VectorXd esplicitamente
                Eigen::VectorXd y_vec(y.rows());
                for (int i = 0; i < y.rows(); ++i) {
                    y_vec(i) = y(i, 0);
                    std::cout << "*** y_vec[" << i << "] = " << y_vec(i) << std::endl;
                }
                
                std::cout << "*** Chiamata a model.fit con X:" << X.rows() << "x" << X.cols() 
                          << ", y_vec:" << y_vec.size() << std::endl;
                
                model.fit(X, y_vec);
                std::cout << "*** fit completato con successo!" << std::endl;
            }
            else if (y.rows() == 1) {
                std::cout << "*** Rilevato y come riga (1,n) = 1x" << y.cols() << std::endl;
                
                Eigen::VectorXd y_vec(y.cols());
                for (int i = 0; i < y.cols(); ++i) {
                    y_vec(i) = y(0, i);
                    std::cout << "*** y_vec[" << i << "] = " << y_vec(i) << std::endl;
                }
                
                std::cout << "*** Chiamata a model.fit..." << std::endl;
                model.fit(X, y_vec);
                std::cout << "*** fit completato con successo!" << std::endl;
            }
            else {
                std::cout << "*** ERRORE: y ha dimensioni non supportate!" << std::endl;
                throw std::runtime_error(
                    "y must be 1-dimensional. Got shape (" + 
                    std::to_string(y.rows()) + ", " + 
                    std::to_string(y.cols()) + ")"
                );
            }
            std::cout << "*** DEBUG: fine wrapper ***\n" << std::endl;
        },
        py::arg("X"), py::arg("y"),
        "Fit the model. y can be (n,), (n,1), or (1,n)");
}

template<typename Class>
void add_optimizer_methods(Class& cls) {
    using ModelType = typename Class::type;
    
    cls
        .def("set_learning_rate", &ModelType::set_learning_rate,
             py::arg("learning_rate"),
             "Set the learning rate for the optimizer")
        .def("get_learning_rate", &ModelType::get_learning_rate,
             "Get the current learning rate")
        .def("set_optimizer", &ModelType::set_optimizer,
             py::arg("type"), py::arg("learning_rate") = 0.01,
             "Set the optimizer type and learning rate");
}

PYBIND11_MODULE(machine_learning_module, m) {
    m.doc() = R"pbdoc(
        Machine Learning Library Python Bindings
        =========================================
        
        A comprehensive C++ ML library with Python bindings using pybind11.
        
        Features:
        - Linear Regression with multiple solvers
        - Logistic Regression with regularization
        - Neural Networks with multiple layers
        - Cross-validation support
        - Model serialization
        - Comprehensive exception handling
        
        Examples
        --------
        >>> import numpy as np
        >>> import machine_learning_module as ml
        >>> 
        >>> # Linear Regression
        >>> X = np.random.rand(100, 3)
        >>> y = np.random.rand(100)
        >>> 
        >>> model = ml.LinearRegression()
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> 
        >>> # Logistic Regression
        >>> X_log = np.random.rand(100, 3)
        >>> y_log = np.random.randint(0, 2, 100)
        >>> 
        >>> log_model = ml.LogisticRegression()
        >>> log_model.fit(X_log, y_log)
        >>> probabilities = log_model.predict(X_log)
        >>> 
        >>> # Neural Network
        >>> nn = ml.NeuralNetwork([3, 64, 32, 1])
        >>> nn.fit(X, y)
        >>> nn_predictions = nn.predict(X)
    )pbdoc";
    
    // Version info
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "Your Name";
    m.attr("__email__") = "your.email@example.com";
    
    // Registra il traduttore di eccezioni
    // MODIFICA: Usa std::exception come base per catturare tutto
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const ml_exception::MLException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_Exception, e.what());
        } catch (...) {
            PyErr_SetString(PyExc_Exception, "Unknown C++ exception");
        }
    });

    // registra tutti i layer all'avvio
    layers::LayerFactory::register_all_layers();
    
    // Bind LinearRegression::Solver enum
    py::enum_<models::LinearRegression::Solver> solver_enum(m, "LinearSolver");
    solver_enum.value("GRADIENT_DESCENT", models::LinearRegression::Solver::GRADIENT_DESCENT)
               .value("NORMAL_EQUATION", models::LinearRegression::Solver::NORMAL_EQUATION)
               .value("SVD", models::LinearRegression::Solver::SVD)
               .export_values();

    // Bind models::OptimizerType enum
    py::enum_<models::OptimizerType>(m, "OptimizerType")
        .value("SGD", models::OptimizerType::SGD)
        .value("ADAM", models::OptimizerType::ADAM)
        .export_values();

    // Bind Estimator base class
    py::class_<models::Estimator, std::shared_ptr<models::Estimator>>(m, "Estimator");
    
    // Bind LinearRegression
    py::class_<models::LinearRegression, models::Estimator, 
           std::shared_ptr<models::LinearRegression>> lr_cls(m, "LinearRegression");     
    lr_cls 
        .def(py::init<>())
        .def(py::init<double, int, double, models::LinearRegression::Solver>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("solver") = models::LinearRegression::Solver::GRADIENT_DESCENT)

        .def("fit", 
             py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
             &models::LinearRegression::fit),
             py::arg("X"), py::arg("y"),
             "Fit the linear regression model with 1D target vector y");

     add_fit_wrapper(lr_cls);  // Aggiunge il wrapper per matrice    
     add_optimizer_methods(lr_cls);   // Per LinearRegression
   
     lr_cls
        .def("predict", py::overload_cast<const Eigen::MatrixXd&>(&models::LinearRegression::predict, py::const_),
             py::arg("X"),
             "Predict using the linear model")
        
        .def("predict", py::overload_cast<const Eigen::VectorXd&>(&models::LinearRegression::predict, py::const_),
             py::arg("x"),
             "Predict a single sample")
        
        .def("score", &models::LinearRegression::score,
             py::arg("X"), py::arg("y"),
             "Return the R² score")
        
        .def("mse", &models::LinearRegression::mse,
             py::arg("X"), py::arg("y"),
             "Compute Mean Squared Error")
        
        .def("mae", &models::LinearRegression::mae,
             py::arg("X"), py::arg("y"),
             "Compute Mean Absolute Error")
        
        .def("r2_score", &models::LinearRegression::r2_score,
             py::arg("X"), py::arg("y"),
             "Compute R² score")
        
        .def("save", &models::LinearRegression::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &models::LinearRegression::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &models::LinearRegression::to_string,
             "String representation of the model")
        
        .def_static("cross_val_score", &models::LinearRegression::cross_val_score,
                    py::arg("X"), py::arg("y"),
                    py::arg("cv") = 5,
                    py::arg("solver") = models::LinearRegression::Solver::GRADIENT_DESCENT,
                    "Cross-validation scores")
        
        .def_property_readonly("coefficients", &models::LinearRegression::coefficients,
                               "Model coefficients (theta)")
        
        .def_property_readonly("intercept", &models::LinearRegression::intercept,
                               "Model intercept")
        
        .def_property_readonly("cost_history", &models::LinearRegression::cost_history,
                               "History of cost values during training")
        
        // Setters
        .def("set_learning_rate", &models::LinearRegression::set_learning_rate,
             py::arg("rate"),
             "Set learning rate")
        
        .def("set_max_iterations", &models::LinearRegression::set_max_iterations,
             py::arg("max_iter"),
             "Set maximum iterations")
        
        .def("set_lambda", &models::LinearRegression::set_lambda,
             py::arg("lambda"),
             "Set regularization parameter")
        
        .def("__repr__", &models::LinearRegression::to_string)
        .def("__str__", &models::LinearRegression::to_string);
    
    
    // Bind LogisticRegression
    py::class_<models::LogisticRegression, models::Estimator,
           std::shared_ptr<models::LogisticRegression>> log_cls(m, "LogisticRegression");
    log_cls
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.1,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)

        .def("fit", 
              py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
              &models::LogisticRegression::fit),
              py::arg("X"), py::arg("y"),
              "Fit the logistic regression model with 1D target vector y");
    add_fit_wrapper(log_cls);  // Aggiunge il wrapper per matrice  
    add_optimizer_methods(log_cls);   // Per LogisticRegression

    log_cls
        .def("predict", &models::LogisticRegression::predict,
             py::arg("X"),
             "Predict probabilities")
        
        .def("predict_class", &models::LogisticRegression::predict_class,
             py::arg("X"), py::arg("threshold") = 0.5,
             "Predict class labels")
        
        .def("score", &models::LogisticRegression::score,
             py::arg("X"), py::arg("y"),
             "Return the accuracy score")
        
        .def("precision_recall_f1", &models::LogisticRegression::precision_recall_f1,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Compute precision, recall and F1 score")
        
        .def("confusion_matrix", &models::LogisticRegression::confusion_matrix,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Compute confusion matrix")
        
        .def("save", &models::LogisticRegression::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &models::LogisticRegression::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &models::LogisticRegression::to_string,
             "String representation of the model")
        
        .def_property_readonly("coefficients", &models::LogisticRegression::coefficients,
                               "Model coefficients (theta)")
        
        .def_property_readonly("intercept", &models::LogisticRegression::intercept,
                               "Model intercept")
        
        .def_property_readonly("cost_history", &models::LogisticRegression::cost_history,
                               "History of cost values during training")
        
        .def_property_readonly("accuracy_history", &models::LogisticRegression::accuracy_history,
                               "History of accuracy values during training")
        
        // Setters
        .def("set_learning_rate", &models::LogisticRegression::set_learning_rate,
             py::arg("rate"),
             "Set learning rate")
        
        .def("set_max_iterations", &models::LogisticRegression::set_max_iterations,
             py::arg("max_iter"),
             "Set maximum iterations")
        
        .def("set_lambda", &models::LogisticRegression::set_lambda,
             py::arg("lambda"),
             "Set regularization parameter")
        
        .def("set_tolerance", &models::LogisticRegression::set_tolerance,
             py::arg("tolerance"),
             "Set convergence tolerance")
        
        .def("set_verbose", &models::LogisticRegression::set_verbose,
             py::arg("verbose"),
             "Set verbose mode")
        
        .def("__repr__", &models::LogisticRegression::to_string)
        .def("__str__", &models::LogisticRegression::to_string);
    
    
    // Bind NeuralNetwork
    py::class_<models::NeuralNetwork, models::Estimator,
           std::shared_ptr<models::NeuralNetwork>> nn_cls(m, "NeuralNetwork");
    nn_cls
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::string&, const std::string&,
                      models::OptimizerType, double>(),
             py::arg("layer_sizes"),
             py::arg("activation") = "relu",
             py::arg("output_activation") = "sigmoid",
             py::arg("optimizer_type") = models::OptimizerType::ADAM,
             py::arg("learning_rate") = 0.001,
             "Create a neural network with specified architecture")
        
        .def("fit", 
              py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
              &models::NeuralNetwork::fit),
              py::arg("X"), py::arg("y"),
              "Fit the neural network with 1D target vector y");
    add_fit_wrapper(nn_cls);  // Aggiunge il wrapper per matrice
    add_optimizer_methods(nn_cls);    // Per NeuralNetwork  
 
    nn_cls    
        .def("predict", &models::NeuralNetwork::predict,
             py::arg("X"),
             "Make predictions")
        
        .def("predict_proba", &models::NeuralNetwork::predict_proba,
             py::arg("X"),
             "Predict probabilities")

        .def("summary", &models::NeuralNetwork::summary, 
          "Print a summary of the network architecture")

        .def("get_training_history", &models::NeuralNetwork::get_training_history,
          "Returns tuple of (loss_history, val_loss_history, accuracy_history)")  
        
        .def("score", &models::NeuralNetwork::score,
             py::arg("X"), py::arg("y"),
             "Compute model score")
        
        .def("save", &models::NeuralNetwork::save,
             py::arg("filename"),
             "Save model to file")
        
        .def("load", &models::NeuralNetwork::load,
             py::arg("filename"),
             "Load model from file")
        
        .def("to_string", &models::NeuralNetwork::to_string,
             "String representation of the model")
        
        // Configuration methods
        .def("set_batch_size", &models::NeuralNetwork::set_batch_size,
             py::arg("batch_size"),
             "Set batch size for training")
        
        .def("set_epochs", &models::NeuralNetwork::set_epochs,
             py::arg("epochs"),
             "Set number of training epochs")
        
        .def("set_validation_split", &models::NeuralNetwork::set_validation_split,
             py::arg("split"),
             "Set validation split ratio")
        
        .def("set_verbose", &models::NeuralNetwork::set_verbose,
             py::arg("verbose"),
             "Set verbose mode")
        
        .def("set_loss_function", &models::NeuralNetwork::set_loss_function,
             py::arg("loss"),
             "Set loss function")
        
        // Getters
        .def_property_readonly("loss_history", &models::NeuralNetwork::get_loss_history,
                               "History of loss values during training")
        
        .def_property_readonly("val_loss_history", &models::NeuralNetwork::get_val_loss_history,
                               "History of validation loss values")
        
        .def_property_readonly("accuracy_history", &models::NeuralNetwork::get_accuracy_history,
                               "History of accuracy values")
        
        .def_property_readonly("num_layers", &models::NeuralNetwork::get_num_layers,
                               "Number of layers in the network")
        
        .def_property_readonly("num_parameters", &models::NeuralNetwork::get_num_parameters,
                               "Total number of parameters in the network")
        
        .def("__repr__", &models::NeuralNetwork::to_string)
        .def("__str__", &models::NeuralNetwork::to_string);
    
    
    // Classe base Scaler
    py::class_<ml::Scaler, std::shared_ptr<ml::Scaler>> scaler(m, "Scaler");
    scaler
        .def("fit", &ml::Scaler::fit, "Fit scaler to data")
        .def("transform", &ml::Scaler::transform, "Transform data")
        .def("fit_transform", &ml::Scaler::fit_transform, "Fit and transform data")
        .def("inverse_transform", &ml::Scaler::inverse_transform, 
             "Inverse transform to original scale")
        .def("get_type", &ml::Scaler::get_type, "Get scaler type name");
    
    // StandardScaler
    py::class_<ml::StandardScaler, ml::Scaler, std::shared_ptr<ml::StandardScaler>>(m, "StandardScaler")
        .def(py::init<>(), "Create StandardScaler with default epsilon=1e-8")
        .def(py::init<double>(), py::arg("epsilon"), 
             "Create StandardScaler with custom epsilon")
        .def("get_mean", &ml::StandardScaler::get_mean, "Get mean values")
        .def("get_std", &ml::StandardScaler::get_std, "Get standard deviation values")
        .def("set_params", &ml::StandardScaler::set_params, 
             "Set scaler parameters", py::arg("mean"), py::arg("std"));
    
    // MinMaxScaler
    py::class_<ml::MinMaxScaler, ml::Scaler, std::shared_ptr<ml::MinMaxScaler>>(m, "MinMaxScaler")
        .def(py::init<>(), "Create MinMaxScaler with default range [0, 1]")
        .def(py::init<double, double>(), 
             py::arg("feature_range_min") = 0.0, 
             py::arg("feature_range_max") = 1.0,
             "Create MinMaxScaler with custom range")
        .def("get_min", &ml::MinMaxScaler::get_min, "Get min values")
        .def("get_max", &ml::MinMaxScaler::get_max, "Get max values")
        .def("set_params", &ml::MinMaxScaler::set_params, 
             "Set scaler parameters", py::arg("min"), py::arg("max"));
    
    // Factory function con lambda
    m.def("create_scaler", 
          [](const std::string& type_name) -> std::shared_ptr<ml::Scaler> {
              return ml::ScalerFactory::create_scaler(type_name);
          },
          py::arg("scaler_type") = "standard",
          "Create a scaler\n\n"
          "Args:\n"
          "    scaler_type (str): 'standard' or 'minmax'\n\n"
          "Returns:\n"
          "    Scaler: A new scaler instance");
    
    // Bind MathUtils as a utility module
    py::class_<utils::MathUtils>(m, "MathUtils")
        .def_static("sigmoid", py::overload_cast<double>(&utils::MathUtils::sigmoid),
                    py::arg("z"),
                    "Compute sigmoid function")
        
        .def_static("sigmoid_vec", &utils::MathUtils::sigmoid_vec,
                    py::arg("z"),
                    "Compute sigmoid for a vector")
        
        .def_static("add_intercept", &utils::MathUtils::add_intercept,
                    py::arg("X"),
                    "Add intercept column to matrix")
        
        .def_static("train_test_split", &utils::MathUtils::train_test_split,
                    py::arg("X"), py::arg("y"),
                    py::arg("test_size") = 0.2,
                    py::arg("random_state") = 42,
                    py::arg("model_type") = "",
                    "Split data into train and test sets")
        
        .def_static("accuracy_score", &utils::MathUtils::accuracy_score,
                    py::arg("y_true"), py::arg("y_pred"),
                    py::arg("model_type") = "",
                    "Compute accuracy score")
        
        .def_static("one_hot_encode", &utils::MathUtils::one_hot_encode,
                    py::arg("labels"), py::arg("num_classes"),
                    "One-hot encode labels");
    
    // Convenience functions
    m.def("test_library", []() {
        return "Machine Learning library v2.0 is working correctly!";
    });
    
    // Version check
    m.def("check_version", []() {
        return std::string("v2.0 - Neural Networks, Exceptions, Serialization");
    });
    
    // Register Eigen matrix converters
    // py::implicitly_convertible<py::array_t<double>, Eigen::MatrixXd>();
    // py::implicitly_convertible<py::array_t<double>, Eigen::VectorXd>();



    
}