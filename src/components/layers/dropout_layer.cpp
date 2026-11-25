#include <random>
#include <memory>
#include <Eigen/Dense>
#include <cstdint>
#include "components/layers/dropout_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>

namespace layers {

DropoutLayer::DropoutLayer(double rate) 
    : rate_(rate), scale_(1.0 / (1.0 - rate)), input_size_(0), rng_(std::random_device{}()) {
    
    ML_CHECK_PARAM(rate >= 0.0 && rate < 1.0, "rate", "must be in [0, 1)", "DropoutLayer");
    cache_ = nullptr;
    
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
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

Eigen::MatrixXd DropoutLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("DropoutLayer", "cache not initialized. Call forward first.");
    }
    
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
    std::ostringstream state;
    state << rng_;
    const std::string serialized_state = state.str();
    const std::uint64_t state_size = serialized_state.size();
    out.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
    out.write(serialized_state.data(), static_cast<std::streamsize>(serialized_state.size()));
}

void DropoutLayer::deserialize(std::istream& in) {
    deserialize_versioned(in, 0);
}

void DropoutLayer::deserialize_versioned(std::istream& in, uint32_t version) {
    if (version > get_version()) {
        ML_THROW_DESERIALIZATION_ERROR("Unsupported DropoutLayer version", "DropoutLayer");
    }
    std::string config;
    std::getline(in, config);
    in.read(reinterpret_cast<char*>(&rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    if (!in || rate_ < 0.0 || rate_ >= 1.0 || input_size_ < 0) {
        ML_THROW_DESERIALIZATION_ERROR("Invalid DropoutLayer data", "DropoutLayer");
    }
    if (version >= 2 || (version == 0 && in.peek() != std::char_traits<char>::eof())) {
        std::uint64_t state_size = 0;
        in.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
        if (!in || state_size == 0 || state_size > 16384) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid DropoutLayer RNG state size", "DropoutLayer");
        }
        std::string serialized_state(static_cast<std::size_t>(state_size), '\0');
        if (!in.read(serialized_state.data(), static_cast<std::streamsize>(state_size))) {
            ML_THROW_DESERIALIZATION_ERROR("Truncated DropoutLayer RNG state", "DropoutLayer");
        }
        std::istringstream state(serialized_state);
        if (!(state >> rng_)) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid DropoutLayer RNG state", "DropoutLayer");
        }
    } else {
        rng_.seed(42);
    }
    scale_ = 1.0 / (1.0 - rate_);
    cache_ = std::make_shared<DropoutCache>();
}

std::string DropoutLayer::get_config() const {
    std::ostringstream oss;
    oss << "DropoutLayer(rate=" << rate_ << ", input_size=" << input_size_ << ")";
    return oss.str();
}

void DropoutLayer::set_input_shape(int input_size) { 
    input_size_ = input_size; 
}

void DropoutLayer::initialize_weights() {
}

void DropoutLayer::initialize_weights(std::mt19937& rng) {
    rng_.seed(rng());
}

} // namespace layers