#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/conv2d_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

// ============================================================================
// COSTRUTTORI
// ============================================================================

Conv2DLayer::Conv2DLayer() 
    : filters_(0), kernel_size_(0), strides_(1), padding_("valid"),
      activation_("relu"), use_bias_(true), input_size_(0),
      input_height_(0), input_width_(0), input_channels_(0),
      output_height_(0), output_width_(0), kernel_elements_(0) {
    cache_ = nullptr;
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

Conv2DLayer::Conv2DLayer(int filters, int kernel_size, int strides,
                       const std::string& padding,
                       const std::string& activation)
    : filters_(filters), kernel_size_(kernel_size), strides_(strides),
      padding_(padding), activation_(activation), use_bias_(true),
      input_size_(0), input_height_(0), input_width_(0), input_channels_(0),
      output_height_(0), output_width_(0), kernel_elements_(0) {
    
    ML_CHECK_PARAM(filters > 0, "filters", "must be > 0", "Conv2DLayer");
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", "Conv2DLayer");
    ML_CHECK_PARAM(strides > 0, "strides", "must be > 0", "Conv2DLayer");
    
    cache_ = nullptr;
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

// ============================================================================
// DIMENSIONI
// ============================================================================

void Conv2DLayer::compute_output_dimensions() {
    if (padding_ == "same") {
        output_height_ = static_cast<int>(std::ceil(static_cast<double>(input_height_) / strides_));
        output_width_ = static_cast<int>(std::ceil(static_cast<double>(input_width_) / strides_));
    } else {  // "valid"
        output_height_ = (input_height_ - kernel_size_) / strides_ + 1;
        output_width_ = (input_width_ - kernel_size_) / strides_ + 1;
    }
    
    if (output_height_ <= 0 || output_width_ <= 0) {
        ML_THROW_PARAMETER_ERROR("dimensions", "output size would be <= 0", "Conv2DLayer");
    }
}

void Conv2DLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "Conv2DLayer");
    
    if (input_size_ == input_size && kernels_.size() > 0) {
        return;
    }
    
    input_size_ = input_size;
    
    // Calcola dimensioni input (assumendo input quadrato)
    input_height_ = static_cast<int>(std::sqrt(input_size));
    input_width_ = input_height_;
    input_channels_ = 1;
    
    // Calcola dimensioni output
    compute_output_dimensions();
    
    // Calcola elementi del kernel
    kernel_elements_ = kernel_size_ * kernel_size_ * input_channels_;
    
    // Ridimensiona kernels
    kernels_.resize(filters_, kernel_elements_);
    
    if (use_bias_) {
        bias_.resize(filters_);
        bias_.setZero();
    }
    
    // Ridimensiona gradienti
    weights_gradient_.resize(filters_, kernel_elements_);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(filters_);
        bias_gradient_.setZero();
    }
    
    // Ricrea cache (usa ConvCache, non Conv2DCache!)
    if (!cache_) {
        cache_ = std::make_shared<ConvCache>();
    }
}

// ============================================================================
// INIZIALIZZAZIONE PESI
// ============================================================================

void Conv2DLayer::initialize_weights() {
    ML_CHECK_PARAM(kernels_.rows() > 0 && kernels_.cols() > 0, 
                   "kernels", "not allocated. Call set_input_shape() first", "Conv2DLayer");
    
    double scale = std::sqrt(2.0 / kernel_elements_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    for (int i = 0; i < kernels_.rows(); ++i) {
        for (int j = 0; j < kernels_.cols(); ++j) {
            kernels_(i, j) = dist(gen);
        }
    }
    
    if (use_bias_) {
        bias_.setZero();
    }
}

int Conv2DLayer::get_output_size() const {
    return output_height_ * output_width_ * filters_;
}

// ============================================================================
// FUNZIONI DI ATTIVAZIONE
// ============================================================================

Eigen::MatrixXd Conv2DLayer::apply_activation(const Eigen::MatrixXd& z) const {
    if (activation_ == "relu") {
        return z.cwiseMax(0.0);
    } else if (activation_ == "sigmoid") {
        return 1.0 / (1.0 + (-z).array().exp());
    } else if (activation_ == "tanh") {
        return z.array().tanh();
    }
    return z;
}

Eigen::MatrixXd Conv2DLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
    if (activation_ == "relu") {
        return (z.array() > 0.0).cast<double>();
    } else if (activation_ == "sigmoid") {
        Eigen::MatrixXd sig = 1.0 / (1.0 + (-z).array().exp());
        return sig.array() * (1.0 - sig.array());
    } else if (activation_ == "tanh") {
        return 1.0 - z.array().tanh().square();
    }
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}

// ============================================================================
// IM2COL E COL2IM
// ============================================================================

Eigen::MatrixXd Conv2DLayer::im2col(const Eigen::MatrixXd& input, int batch_idx, int start_idx) const {
    int input_offset = batch_idx * input_size_ + start_idx;
    int patches = output_height_ * output_width_;
    int patch_size = kernel_elements_;
    
    Eigen::MatrixXd cols(patches, patch_size);
    cols.setZero();
    
    for (int ph = 0; ph < output_height_; ++ph) {
        for (int pw = 0; pw < output_width_; ++pw) {
            int patch_idx = ph * output_width_ + pw;
            int h_start = ph * strides_;
            int w_start = pw * strides_;
            
            for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                    for (int c = 0; c < input_channels_; ++c) {
                        int h = h_start + kh;
                        int w = w_start + kw;
                        
                        if (h < input_height_ && w < input_width_) {
                            int input_idx = input_offset + (h * input_width_ + w) * input_channels_ + c;
                            int col_idx = (kh * kernel_size_ + kw) * input_channels_ + c;
                            cols(patch_idx, col_idx) = input(input_idx);
                        }
                    }
                }
            }
        }
    }
    
    return cols;
}

Eigen::MatrixXd Conv2DLayer::col2im(const Eigen::MatrixXd& col, int batch_idx, int start_idx) const {
    Eigen::MatrixXd im = Eigen::MatrixXd::Zero(input_size_, 1);
    
    for (int ph = 0; ph < output_height_; ++ph) {
        for (int pw = 0; pw < output_width_; ++pw) {
            int patch_idx = ph * output_width_ + pw;
            int h_start = ph * strides_;
            int w_start = pw * strides_;
            
            for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                    for (int c = 0; c < input_channels_; ++c) {
                        int h = h_start + kh;
                        int w = w_start + kw;
                        
                        if (h < input_height_ && w < input_width_) {
                            int input_idx = (h * input_width_ + w) * input_channels_ + c;
                            int col_idx = (kh * kernel_size_ + kw) * input_channels_ + c;
                            im(input_idx) += col(patch_idx, col_idx);
                        }
                    }
                }
            }
        }
    }
    
    return im;
}

Eigen::MatrixXd Conv2DLayer::col2im_from_cols(const Eigen::MatrixXd& dCol, int batch_idx) const {
    Eigen::MatrixXd im = Eigen::MatrixXd::Zero(input_size_, 1);
    int spatial_size = output_height_ * output_width_;
    
    for (int ph = 0; ph < output_height_; ++ph) {
        for (int pw = 0; pw < output_width_; ++pw) {
            int patch_idx = ph * output_width_ + pw;
            int h_start = ph * strides_;
            int w_start = pw * strides_;
            
            for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                    for (int c = 0; c < input_channels_; ++c) {
                        int h = h_start + kh;
                        int w = w_start + kw;
                        
                        if (h < input_height_ && w < input_width_) {
                            int input_idx = (h * input_width_ + w) * input_channels_ + c;
                            int col_idx = (kh * kernel_size_ + kw) * input_channels_ + c;
                            if (patch_idx < dCol.rows() && col_idx < dCol.cols()) {
                                im(input_idx) += dCol(patch_idx, col_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return im;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "Conv2DLayer");
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "layer not initialized", "Conv2DLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "Conv2DLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<ConvCache>();
    }
    
    int batch_size = input.rows();
    int output_size = get_output_size();
    int spatial_size = output_height_ * output_width_;
    int col_size = spatial_size * kernel_elements_;
    
    cache_->set_input_shape(input_height_, input_width_, input_channels_);
    cache_->set_output_shape(output_height_, output_width_, filters_);
    cache_->set_batch_size(batch_size);
    cache_->set_kernel_info(kernel_size_, strides_, padding_);
    
    cache_->input_cache = input;
    cache_->output_cache.resize(batch_size, output_size);
    cache_->z_cache.resize(batch_size, output_size);
    cache_->col_cache.resize(batch_size, col_size);
    
    Eigen::MatrixXd output(batch_size, output_size);
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::MatrixXd sample = input.row(b).transpose();
        Eigen::MatrixXd cols = im2col(sample, b, 0);
        
        Eigen::Map<Eigen::RowVectorXd> col_cache_row(cache_->col_cache.row(b).data(), col_size);
        col_cache_row = Eigen::Map<Eigen::RowVectorXd>(cols.data(), col_size);
        
        Eigen::MatrixXd conv = cols * kernels_.transpose();
        
        if (use_bias_) {
            conv.rowwise() += bias_.transpose();
        }
        
        Eigen::Map<Eigen::RowVectorXd> output_row(conv.data(), output_size);
        output.row(b) = output_row;
        
        if (training) {
            cache_->z_cache.row(b) = output_row;
        }
    }
    
    cache_->output_cache = (activation_ == "linear") ? output : apply_activation(output);
    
    return cache_->output_cache;
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

Eigen::MatrixXd Conv2DLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("Conv2DLayer", "cache not initialized. Call forward first.");
    }
    
    auto conv_cache = get_specific_cache();
    
    int batch_size = conv_cache->batch_size;
    int output_size = get_output_size();
    int spatial_size = output_height_ * output_width_;
    
    if (gradient.rows() != batch_size || gradient.cols() != output_size) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient",
            batch_size, output_size,
            gradient.rows(), gradient.cols(), "Conv2DLayer");
    }
    
    Eigen::MatrixXd dOut = (activation_ == "linear") ? gradient : 
                           apply_activation_derivative(conv_cache->output_cache).array() * gradient.array();
    
    Eigen::MatrixXd dZ(batch_size * spatial_size, filters_);
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<Eigen::MatrixXd>(&dZ(b * spatial_size, 0), spatial_size, filters_) = 
            Eigen::Map<Eigen::MatrixXd>(dOut.row(b).data(), spatial_size, filters_);
    }
    
    Eigen::MatrixXd dKernels = Eigen::MatrixXd::Zero(kernels_.rows(), kernels_.cols());
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<Eigen::MatrixXd> cols(
            conv_cache->col_cache.row(b).data(),
            spatial_size,
            kernel_elements_);
        
        dKernels += dZ.block(b * spatial_size, 0, spatial_size, filters_).transpose() * cols;
    }
    
    Eigen::VectorXd dbias = dZ.colwise().sum();
    
    if (use_bias_) {
        weights_gradient_.resize(dKernels.rows(), dKernels.cols() + 1);
        weights_gradient_.leftCols(dKernels.cols()) = dKernels;
        weights_gradient_.col(dKernels.cols()) = dbias;
        bias_gradient_.resize(0);
    } else {
        weights_gradient_ = dKernels;
    }
    
    Eigen::MatrixXd dX(batch_size, input_size_);
    dX.setZero();
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<Eigen::MatrixXd> cols(
            conv_cache->col_cache.row(b).data(),
            spatial_size,
            kernel_elements_);
        
        Eigen::MatrixXd dZ_b = dZ.block(b * spatial_size, 0, spatial_size, filters_);
        Eigen::MatrixXd dCol = dZ_b * kernels_;
        Eigen::MatrixXd dSample = col2im(dCol, b, 0);
        dX.row(b) = dSample.transpose();
    }
    
    return dX;
}

// ============================================================================
// GETTER/SETTER PESI
// ============================================================================

Eigen::MatrixXd Conv2DLayer::get_weights() const {
    if (use_bias_) {
        Eigen::MatrixXd weights_with_bias(kernels_.rows(), kernels_.cols() + 1);
        weights_with_bias.leftCols(kernels_.cols()) = kernels_;
        weights_with_bias.col(kernels_.cols()) = bias_;
        return weights_with_bias;
    }
    return kernels_;
}

void Conv2DLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (use_bias_) {
        if (weights.cols() != kernels_.cols() + 1) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimension", "Conv2DLayer");
        }
        kernels_ = weights.leftCols(kernels_.cols());
        bias_ = weights.col(kernels_.cols());
    } else {
        if (weights.cols() != kernels_.cols()) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimension", "Conv2DLayer");
        }
        kernels_ = weights;
    }
}

int Conv2DLayer::get_parameter_count() const {
    return kernels_.size() + (use_bias_ ? bias_.size() : 0);
}

// ============================================================================
// SERIALIZZAZIONE
// ============================================================================

void Conv2DLayer::serialize(std::ostream& out) const {
    int32_t filters = filters_;
    int32_t kernel_size = kernel_size_;
    int32_t strides = strides_;
    int32_t input_size = input_size_;
    int32_t input_height = input_height_;
    int32_t input_width = input_width_;
    int32_t input_channels = input_channels_;
    int32_t output_height = output_height_;
    int32_t output_width = output_width_;
    int32_t kernel_elements = kernel_elements_;
    
    out.write(reinterpret_cast<const char*>(&filters), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&kernel_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&strides), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&input_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&input_height), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&input_width), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&input_channels), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&output_height), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&output_width), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&kernel_elements), sizeof(int32_t));
    
    auto write_str = [&](const std::string& s) {
        int32_t len = static_cast<int32_t>(s.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
        out.write(s.c_str(), len);
    };
    
    write_str(padding_);
    write_str(activation_);
    
    int8_t use_bias_flag = use_bias_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_bias_flag), sizeof(int8_t));
    
    auto write_mat = [&](const Eigen::MatrixXd& m) {
        int32_t rows = static_cast<int32_t>(m.rows());
        int32_t cols = static_cast<int32_t>(m.cols());
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
    };
    
    write_mat(kernels_);
    
    if (use_bias_) {
        write_mat(bias_);
    }
    
    if (!out.good()) {
        throw ml_exception::SerializationException("Failed to write Conv2DLayer", "Conv2DLayer");
    }
}

void Conv2DLayer::deserialize(std::istream& in) {
    int32_t filters, kernel_size, strides, input_size, input_height, input_width;
    int32_t input_channels, output_height, output_width, kernel_elements;
    
    in.read(reinterpret_cast<char*>(&filters), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&kernel_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&strides), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&input_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&input_height), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&input_width), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&input_channels), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&output_height), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&output_width), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&kernel_elements), sizeof(int32_t));
    
    filters_ = filters;
    kernel_size_ = kernel_size;
    strides_ = strides;
    input_size_ = input_size;
    input_height_ = input_height;
    input_width_ = input_width;
    input_channels_ = input_channels;
    output_height_ = output_height;
    output_width_ = output_width;
    kernel_elements_ = kernel_elements;
    
    auto read_str = [&]() {
        int32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        std::vector<char> buf(len + 1, '\0');
        in.read(buf.data(), len);
        return std::string(buf.data());
    };
    
    padding_ = read_str();
    activation_ = read_str();
    
    int8_t use_bias_flag;
    in.read(reinterpret_cast<char*>(&use_bias_flag), sizeof(int8_t));
    use_bias_ = (use_bias_flag != 0);
    
    auto read_mat = [&]() {
        int32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        Eigen::MatrixXd m(rows, cols);
        in.read(reinterpret_cast<char*>(m.data()), rows * cols * sizeof(double));
        return m;
    };
    
    kernels_ = read_mat();
    
    if (use_bias_) {
        bias_ = read_mat();
    }
    
    weights_gradient_.resize(kernels_.rows(), kernels_.cols() + (use_bias_ ? 1 : 0));
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(filters_);
        bias_gradient_.setZero();
    }
    
    cache_ = std::make_shared<ConvCache>();
    
    if (!in.good() && !in.eof()) {
        throw ml_exception::DeserializationException("Stream error", "Conv2DLayer");
    }
}

// ============================================================================
// CONFIG
// ============================================================================

std::string Conv2DLayer::get_config() const {
    std::ostringstream oss;
    oss << "Conv2DLayer(filters=" << filters_
        << ", kernel_size=" << kernel_size_
        << ", strides=" << strides_
        << ", padding=" << padding_
        << ", activation=" << activation_
        << ", input_size=" << input_size_ << ")";
    return oss.str();
}

} // namespace layers