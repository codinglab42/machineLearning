#include "components/layers/flatten_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>

namespace layers {

    FlattenLayer::FlattenLayer() : input_size_(0) {
        cache_ = nullptr;
    }

    Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd& input, bool training) {
        ML_CHECK_NOT_EMPTY(input, "input", "FlattenLayer");
        
        if (!cache_) {
            cache_ = std::make_shared<FlattenCache>();
        }
        
        input_size_ = input.cols();
        
        cache_->original_shape.clear();
        cache_->original_shape.push_back(input.rows());
        cache_->original_shape.push_back(input.cols());
        
        cache_->input_cache = input;
        cache_->output_cache = input;
        
        return input;
    }

    Eigen::MatrixXd FlattenLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("FlattenLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate >= 0, "learning_rate", "must be >= 0", "FlattenLayer");
        
        return gradient;
    }

    void FlattenLayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    }

    void FlattenLayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    }

    std::string FlattenLayer::get_config() const {
        std::ostringstream oss;
        oss << "FlattenLayer(input_size=" << input_size_ << ")";
        return oss.str();
    }

    void FlattenLayer::set_input_shape(int input_size) { 
        input_size_ = input_size; 
    }

} // namespace layers
