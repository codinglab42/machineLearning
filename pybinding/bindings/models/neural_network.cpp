#include "neural_network.h"
#include "models/neural_network.h"
#include "components/layers/layer.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_neural_network(py::module_& m) {
    py::class_<models::NeuralNetwork, models::Estimator,
               std::shared_ptr<models::NeuralNetwork>> cls(m, "NeuralNetwork");
    
    cls
        // ====================================================================
        // COSTRUTTORI
        // ====================================================================
        .def(py::init<>())
        .def(py::init<const std::vector<int>&, const std::string&, const std::string&,
                      models::OptimizerType, double>(),
             py::arg("layer_sizes"),
             py::arg("activation") = "relu",
             py::arg("output_activation") = "sigmoid",
             py::arg("optimizer_type") = models::OptimizerType::ADAM,
             py::arg("learning_rate") = 0.001)
        
        // ====================================================================
        // LAYER CONSTRUCTION
        // ====================================================================
        .def("add_dense_layer", &models::NeuralNetwork::add_dense_layer,
             py::arg("units"), py::arg("activation") = "relu", py::arg("use_bias") = true)
        .def("add_conv2d_layer", &models::NeuralNetwork::add_conv2d_layer,
             py::arg("filters"), py::arg("kernel_size"), py::arg("strides") = 1,
             py::arg("padding") = "valid", py::arg("activation") = "relu")
        .def("add_pooling_layer", &models::NeuralNetwork::add_pooling_layer,
             py::arg("pool_size") = 2, py::arg("strides") = 2, py::arg("pool_type") = "max")
        .def("add_flatten_layer", &models::NeuralNetwork::add_flatten_layer)
        .def("add_dropout_layer", &models::NeuralNetwork::add_dropout_layer,
             py::arg("rate"))
        .def("add_batch_norm_layer", &models::NeuralNetwork::add_batch_norm_layer)
        .def("add_recurrent_layer", &models::NeuralNetwork::add_recurrent_layer,
             py::arg("type"), py::arg("units"), py::arg("return_sequences") = false,
             py::arg("activation") = "tanh")
        // ⭐ CORRETTO - usa lambda per risolvere l'overload
        .def("add_layer_type", 
             [](models::NeuralNetwork& nn, layers::LayerType type, 
                const std::unordered_map<std::string, double>& params) {
                 nn.add_layer(type, params);
             },
             py::arg("type"), py::arg("params") = std::unordered_map<std::string, double>())
        
        // ====================================================================
        // BUILD
        // ====================================================================
        .def("build", &models::NeuralNetwork::build,
             py::arg("n_features"), py::arg("n_classes"))
        
        // ====================================================================
        // TRAINING
        // ====================================================================
        .def("fit", 
             py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
                 &models::NeuralNetwork::fit),
             py::arg("X"), py::arg("y"))
        .def("fit_matrix", 
             py::overload_cast<const Eigen::MatrixXd&, const Eigen::MatrixXd&>(
                 &models::NeuralNetwork::fit),
             py::arg("X"), py::arg("y"))
        .def("fit_advanced",
             [](models::NeuralNetwork& nn, const Eigen::MatrixXd& X,
                const Eigen::VectorXd& y, int epochs, int batch_size, bool verbose) {
                 nn.fit(X, y, epochs, batch_size, verbose);
             },
             py::arg("X"), py::arg("y"), py::arg("epochs"),
             py::arg("batch_size"), py::arg("verbose") = false)
        .def("fit_advanced_matrix",
             [](models::NeuralNetwork& nn, const Eigen::MatrixXd& X,
                const Eigen::MatrixXd& y, int epochs, int batch_size, bool verbose) {
                 nn.fit(X, y, epochs, batch_size, verbose);
             },
             py::arg("X"), py::arg("y"), py::arg("epochs"),
             py::arg("batch_size"), py::arg("verbose") = false)
        
        // ====================================================================
        // PREDICTION
        // ====================================================================
        .def("predict", &models::NeuralNetwork::predict, py::arg("X"))
        .def("predict_proba", &models::NeuralNetwork::predict_proba, py::arg("X"))
        .def("score", &models::NeuralNetwork::score, py::arg("X"), py::arg("y"))
        
        // ====================================================================
        // NETWORK INFO
        // ====================================================================
        .def("summary", &models::NeuralNetwork::summary)
        .def("get_summary", &models::NeuralNetwork::to_string)
        .def("get_training_history", &models::NeuralNetwork::get_training_history)
        .def_property_readonly("num_layers", &models::NeuralNetwork::get_num_layers)
        .def_property_readonly("num_parameters", &models::NeuralNetwork::get_num_parameters)
        .def_property_readonly("loss_history", &models::NeuralNetwork::get_loss_history)
        .def_property_readonly("val_loss_history", &models::NeuralNetwork::get_val_loss_history)
        .def_property_readonly("accuracy_history", &models::NeuralNetwork::get_accuracy_history)
        .def_property_readonly("input_size", &models::NeuralNetwork::get_input_size)
        .def_property_readonly("output_size", &models::NeuralNetwork::get_output_size)
        .def_property_readonly("is_fitted", &models::NeuralNetwork::is_fitted)
        
        // ====================================================================
        // CONFIGURATION
        // ====================================================================
        .def("set_batch_size", &models::NeuralNetwork::set_batch_size, py::arg("batch_size"))
        .def("set_epochs", &models::NeuralNetwork::set_epochs, py::arg("epochs"))
        .def("set_validation_split", &models::NeuralNetwork::set_validation_split, py::arg("split"))
        .def("set_verbose", &models::NeuralNetwork::set_verbose, py::arg("verbose"))
        .def("set_loss_function", &models::NeuralNetwork::set_loss_function, py::arg("loss"))
        .def("set_regularizer", &models::NeuralNetwork::set_regularizer,
             py::arg("type"), py::arg("strength") = 0.01,
             py::arg("params") = std::unordered_map<std::string, double>())
        .def("set_optimizer", 
             [](models::NeuralNetwork& nn, models::OptimizerType type, double learning_rate) {
                 nn.set_optimizer(type, learning_rate);
             },
             py::arg("type"), py::arg("learning_rate") = 0.01)
        
        // ====================================================================
        // LAYER ACCESS
        // ====================================================================
        .def("get_layer", [](const models::NeuralNetwork& nn, int idx) -> layers::Layer* {
            const auto& layers = nn.get_layers();
            if (idx < 0 || idx >= static_cast<int>(layers.size())) {
                throw std::out_of_range("Layer index out of range");
            }
            return layers[idx].get();
        }, py::arg("idx"))
        .def("get_weights_for_layer", [](const models::NeuralNetwork& nn, int idx) {
            const auto& layers = nn.get_layers();
            if (idx < 0 || idx >= static_cast<int>(layers.size())) {
                throw std::out_of_range("Layer index out of range");
            }
            return layers[idx]->get_weights();
        }, py::arg("idx"))
        .def("set_weights_for_layer", [](models::NeuralNetwork& nn, int idx,
                                          const Eigen::MatrixXd& weights) {
            const auto& layers = nn.get_layers();
            if (idx < 0 || idx >= static_cast<int>(layers.size())) {
                throw std::out_of_range("Layer index out of range");
            }
            layers[idx]->set_weights(weights);
        }, py::arg("idx"), py::arg("weights"))
        .def("get_layer_type", [](const models::NeuralNetwork& nn, int idx) {
            const auto& layers = nn.get_layers();
            if (idx < 0 || idx >= static_cast<int>(layers.size())) {
                throw std::out_of_range("Layer index out of range");
            }
            return layers[idx]->get_type();
        }, py::arg("idx"))
        .def("get_layer_config", [](const models::NeuralNetwork& nn, int idx) {
            const auto& layers = nn.get_layers();
            if (idx < 0 || idx >= static_cast<int>(layers.size())) {
                throw std::out_of_range("Layer index out of range");
            }
            return layers[idx]->get_config();
        }, py::arg("idx"))
        
        // ====================================================================
        // RESET
        // ====================================================================
        .def("reset", &models::NeuralNetwork::reset)
        .def("reset_history", &models::NeuralNetwork::reset_history)
        
        // ====================================================================
        // SERIALIZATION
        // ====================================================================
        .def("save", &models::NeuralNetwork::save, py::arg("filename"))
        .def("load", &models::NeuralNetwork::load, py::arg("filename"))
        .def("to_string", &models::NeuralNetwork::to_string)
        
        .def("__repr__", &models::NeuralNetwork::to_string);
}