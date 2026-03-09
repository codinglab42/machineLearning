#include "components/layers/dense_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

    DenseLayer::DenseLayer(int units, const std::string& activation, bool use_bias)
        : units_(units), activation_(activation), use_bias_(use_bias), input_size_(0) {
        
        ML_CHECK_PARAM(units > 0, "units", "must be > 0", "DenseLayer");
        cache_ = nullptr;
    }

    void DenseLayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "DenseLayer");
        
        input_size_ = input_size;
        
        double scale = std::sqrt(2.0 / (input_size + units_));
        
        weights_.resize(input_size, units_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, scale);
        
        for (int i = 0; i < weights_.rows(); ++i) {
            for (int j = 0; j < weights_.cols(); ++j) {
                weights_(i, j) = dist(gen);
            }
        }
        
        if (use_bias_) {
            bias_.setZero(units_);
        }
    }

    Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input, bool training) {
        ML_CHECK_NOT_EMPTY(input, "input", "DenseLayer");
        
        if (input.cols() != input_size_) {
            ML_THROW_DIMENSION_MISMATCH("forward input", 
                input.rows(), input_size_,
                input.rows(), input.cols(), "DenseLayer");
        }
        
        if (!cache_) {
            cache_ = std::make_shared<DenseCache>();
        }
        
        Eigen::MatrixXd z = input * weights_;
        
        if (use_bias_) {
            z.rowwise() += bias_.transpose();
        }
        
        Eigen::MatrixXd output = apply_activation(z);
        
        cache_->input_cache = input;
        cache_->z_cache = z;
        cache_->output_cache = output;
        
        return output;
    }

    Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("DenseLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "DenseLayer");
        
        auto dense_cache = get_specific_cache();
        
        const Eigen::MatrixXd& input = dense_cache->input_cache;
        const Eigen::MatrixXd& z = dense_cache->z_cache;
        
        if (gradient.rows() != input.rows() || gradient.cols() != units_) {
            ML_THROW_DIMENSION_MISMATCH("backward gradient",
                input.rows(), units_,
                gradient.rows(), gradient.cols(), "DenseLayer");
        }
        
        Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(z).array();
        
        Eigen::MatrixXd dW = input.transpose() * dZ;
        
        Eigen::MatrixXd dX = dZ * weights_.transpose();
        
        weights_ -= learning_rate * dW / input.rows();
        
        if (use_bias_) {
            Eigen::VectorXd db = dZ.colwise().sum();
            bias_ -= learning_rate * db / input.rows();
        }
        
        return dX;
    }

    Eigen::MatrixXd DenseLayer::apply_activation(const Eigen::MatrixXd& z) const {
        if (activation_ == "relu") {
            return z.cwiseMax(0.0);
        } else if (activation_ == "sigmoid") {
            return 1.0 / (1.0 + (-z).array().exp());
        } else if (activation_ == "tanh") {
            return z.array().tanh();
        } else if (activation_ == "softmax") {
            Eigen::MatrixXd exp = z.array().exp();
            Eigen::VectorXd sum = exp.rowwise().sum();
            return exp.array().colwise() / sum.array();
        } else if (activation_ == "linear") {
            return z;
        }
        return z.cwiseMax(0.0);
    }

    Eigen::MatrixXd DenseLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
        if (activation_ == "relu") {
            return (z.array() > 0.0).cast<double>();
        } else if (activation_ == "sigmoid") {
            Eigen::MatrixXd sig = 1.0 / (1.0 + (-z).array().exp());
            return sig.array() * (1.0 - sig.array());
        } else if (activation_ == "tanh") {
            Eigen::MatrixXd tanh = z.array().tanh();
            return 1.0 - tanh.array().square();
        } else if (activation_ == "linear") {
            return Eigen::MatrixXd::Ones(z.rows(), z.cols());
        } else if (activation_ == "softmax") {
            return Eigen::MatrixXd::Ones(z.rows(), z.cols());
        }
        return (z.array() > 0.0).cast<double>();
    }

    Eigen::MatrixXd DenseLayer::get_weights() const {
        if (use_bias_) {
            Eigen::MatrixXd w(weights_.rows(), weights_.cols() + 1);
            w.leftCols(weights_.cols()) = weights_;
            w.col(weights_.cols()) = bias_;
            return w;
        }
        return weights_;
    }

    void DenseLayer::set_weights(const Eigen::MatrixXd& weights) {
        if (use_bias_) {
            if (weights.cols() != weights_.cols() + 1) {
                ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "DenseLayer");
            }
            weights_ = weights.leftCols(weights_.cols());
            bias_ = weights.col(weights_.cols());
        } else {
            if (weights.cols() != weights_.cols()) {
                ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "DenseLayer");
            }
            weights_ = weights;
        }
    }

    int DenseLayer::get_parameter_count() const {
        return weights_.size() + (use_bias_ ? bias_.size() : 0);
    }

    void DenseLayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
        
        bool has_bias = use_bias_;
        out.write(reinterpret_cast<const char*>(&has_bias), sizeof(bool));
        
        for (int i = 0; i < weights_.rows(); ++i) {
            for (int j = 0; j < weights_.cols(); ++j) {
                out.write(reinterpret_cast<const char*>(&weights_(i, j)), sizeof(double));
            }
        }
        
        if (use_bias_) {
            for (int i = 0; i < bias_.size(); ++i) {
                out.write(reinterpret_cast<const char*>(&bias_(i)), sizeof(double));
            }
        }
    }

    void DenseLayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        in.read(reinterpret_cast<char*>(&units_), sizeof(int));
        
        bool has_bias;
        in.read(reinterpret_cast<char*>(&has_bias), sizeof(bool));
        use_bias_ = has_bias;
        
        weights_.resize(input_size_, units_);
        for (int i = 0; i < weights_.rows(); ++i) {
            for (int j = 0; j < weights_.cols(); ++j) {
                in.read(reinterpret_cast<char*>(&weights_(i, j)), sizeof(double));
            }
        }
        
        if (use_bias_) {
            bias_.resize(units_);
            for (int i = 0; i < bias_.size(); ++i) {
                in.read(reinterpret_cast<char*>(&bias_(i)), sizeof(double));
            }
        }
    }

    std::string DenseLayer::get_config() const {
        std::ostringstream oss;
        oss << "DenseLayer(units=" << units_ 
            << ", activation=" << activation_
            << ", use_bias=" << (use_bias_ ? "true" : "false")
            << ", input_size=" << input_size_ << ")";
        return oss.str();
    }

} // namespace layers
