#include "components/layers/dropout_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>

namespace layers {

    DropoutLayer::DropoutLayer(double rate) 
        : rate_(rate), scale_(1.0 / (1.0 - rate)), input_size_(0), rng_(std::random_device{}()) {
        
        ML_CHECK_PARAM(rate >= 0.0 && rate < 1.0, "rate", "must be in [0, 1)", "DropoutLayer");
        cache_ = nullptr;
    }

    Eigen::MatrixXd DropoutLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    Eigen::MatrixXd DropoutLayer::forward(const Eigen::MatrixXd& input, bool training) {
        ML_CHECK_NOT_EMPTY(input, "input", "DropoutLayer");
        
        if (!cache_) {
            cache_ = std::make_shared<DropoutCache>();
        }
        
        input_size_ = input.cols();
        cache_->input_cache = input;
        cache_->training = training;
        
        if (!training) {
            cache_->output_cache = input;
            return input;
        }
        
        cache_->mask.resize(input.rows(), input.cols());
        
        std::bernoulli_distribution dist(1.0 - rate_);
        
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                cache_->mask(i, j) = dist(rng_) ? scale_ : 0.0;
            }
        }
        
        cache_->output_cache = input.cwiseProduct(cache_->mask);
        return cache_->output_cache;
    }

    Eigen::MatrixXd DropoutLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("DropoutLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate >= 0, "learning_rate", "must be >= 0", "DropoutLayer");
        
        if (!cache_->training) {
            return gradient;
        }
        
        if (gradient.rows() != cache_->mask.rows() || gradient.cols() != cache_->mask.cols()) {
            ML_THROW_DIMENSION_MISMATCH("backward gradient",
                cache_->mask.rows(), cache_->mask.cols(),
                gradient.rows(), gradient.cols(), "DropoutLayer");
        }
        
        return gradient.cwiseProduct(cache_->mask);
    }

    void DropoutLayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&rate_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    }

    void DropoutLayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        in.read(reinterpret_cast<char*>(&rate_), sizeof(double));
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        scale_ = 1.0 / (1.0 - rate_);
    }

    std::string DropoutLayer::get_config() const {
        std::ostringstream oss;
        oss << "DropoutLayer(rate=" << rate_ << ", input_size=" << input_size_ << ")";
        return oss.str();
    }

    void DropoutLayer::set_input_shape(int input_size) { 
        input_size_ = input_size; 
    }

} // namespace layers

