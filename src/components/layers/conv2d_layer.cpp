// src/components/layers/conv2d_layer.cpp
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/conv2d_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

Conv2DLayer::Conv2DLayer(int filters, int kernel_size, int strides,
                       const std::string& padding,
                       const std::string& activation)
    : filters_(filters), kernel_size_(kernel_size), strides_(strides),
      padding_(padding), activation_(activation), input_size_(0),
      input_height_(0), input_width_(0), input_channels_(0),
      output_height_(0), output_width_(0), kernel_elements_(0), 
      use_bias_(true) {
    
    ML_CHECK_PARAM(filters > 0, "filters", "must be > 0", "Conv2DLayer");
    ML_CHECK_PARAM(kernel_size > 0, "kernel_size", "must be > 0", "Conv2DLayer");
    ML_CHECK_PARAM(strides > 0, "strides", "must be > 0", "Conv2DLayer");
    
    cache_ = nullptr;
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

void Conv2DLayer::compute_output_dimensions() {
    if (padding_ == "same") {
        output_height_ = static_cast<int>(std::ceil(static_cast<double>(input_height_) / strides_));
        output_width_ = static_cast<int>(std::ceil(static_cast<double>(input_width_) / strides_));
    } else {
        output_height_ = (input_height_ - kernel_size_) / strides_ + 1;
        output_width_ = (input_width_ - kernel_size_) / strides_ + 1;
    }
    
    if (output_height_ <= 0 || output_width_ <= 0) {
        ML_THROW_PARAMETER_ERROR("dimensions", "output size would be <= 0", "Conv2DLayer");
    }
}

void Conv2DLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "Conv2DLayer");
    
    input_size_ = input_size;
    
    input_height_ = static_cast<int>(std::sqrt(input_size));
    input_width_ = input_height_;
    input_channels_ = 1;
    
    compute_output_dimensions();
    
    kernel_elements_ = kernel_size_ * kernel_size_ * input_channels_;
    
    double scale = std::sqrt(2.0 / (kernel_elements_ * filters_));
    
    kernels_.resize(filters_, kernel_elements_);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    for (int i = 0; i < kernels_.rows(); ++i) {
        for (int j = 0; j < kernels_.cols(); ++j) {
            kernels_(i, j) = dist(gen);
        }
    }
    
    bias_.setZero(filters_);
    
    weights_gradient_.resize(filters_, kernel_elements_);
    weights_gradient_.setZero();
    bias_gradient_.resize(filters_);
    bias_gradient_.setZero();
}

int Conv2DLayer::get_output_size() const {
    return output_height_ * output_width_ * filters_;
}

// ============================================================================
// FORWARD - Input: [batch_size, input_size], Output: [batch_size, output_size]
// ============================================================================
Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "Conv2DLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "Conv2DLayer");
    }
    
    // Se la cache non esiste OPPURE non è configurata, ricreala e configurala
    if (!cache_) {
        std::cout << "Creating new cache" << std::endl;
        cache_ = std::make_shared<ConvCache>();
    }
    
    // IMPORTANTE: Configura sempre la cache con i parametri attuali del layer
    // Anche se la cache esiste già, potrebbe essere stata creata ma non configurata
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
        conv.rowwise() += bias_.transpose();
        
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
// BACKWARD - Input: gradient [batch_size, output_size], Output: dX [batch_size, input_size]
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
    
    // Gradiente post-attivazione
    Eigen::MatrixXd dOut = (activation_ == "linear") ? gradient : 
                           apply_activation_derivative(conv_cache->output_cache).array() * gradient.array();
    
    // Reshape dOut in formato [batch_size * spatial_size, filters]
    Eigen::MatrixXd dZ(batch_size * spatial_size, filters_);
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<Eigen::MatrixXd>(&dZ(b * spatial_size, 0), spatial_size, filters_) = 
            Eigen::Map<Eigen::MatrixXd>(dOut.row(b).data(), spatial_size, filters_);
    }
    
    // Calcola gradienti per kernels
    Eigen::MatrixXd dKernels = Eigen::MatrixXd::Zero(kernels_.rows(), kernels_.cols());
    
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<Eigen::MatrixXd> cols(
            conv_cache->col_cache.row(b).data(),
            spatial_size,
            kernel_elements_);
        
        dKernels += dZ.block(b * spatial_size, 0, spatial_size, filters_).transpose() * cols;
    }
    
    // Calcola gradienti per bias
    Eigen::VectorXd dbias = dZ.colwise().sum();
    
    if (use_bias_) {
        // UNIFICA: get_weights() restituisce [filters, kernel_elements + 1]
        // Quindi weights_gradient_ deve avere le stesse dimensioni
        weights_gradient_.resize(dKernels.rows(), dKernels.cols() + 1);
        weights_gradient_.leftCols(dKernels.cols()) = dKernels;
        weights_gradient_.col(dKernels.cols()) = dbias;
        bias_gradient_.resize(0);  // Non serve più separatamente
    } else {
        weights_gradient_ = dKernels;
    }
    
    // Calcola dX per l'input usando le colonne salvate
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
// im2col per un singolo campione - formato vettore colonna
// ============================================================================
Eigen::MatrixXd Conv2DLayer::im2col(const Eigen::MatrixXd& input, int batch_idx, int start_idx) const {
    int input_offset = batch_idx * input_size_ + start_idx;
    int patches = output_height_ * output_width_;
    int patch_size = kernel_elements_;
    
    Eigen::MatrixXd cols(patches, patch_size);
    
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
                            cols(patch_idx, (kh * kernel_size_ + kw) * input_channels_ + c) = input(input_idx);
                        } else {
                            cols(patch_idx, (kh * kernel_size_ + kw) * input_channels_ + c) = 0.0;
                        }
                    }
                }
            }
        }
    }
    
    return cols;
}

// ============================================================================
// col2im per un singolo campione - formato vettore colonna
// ============================================================================
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

// ============================================================================
// col2im usa le colonne già calcolate
// ============================================================================
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
                            // dCol ha dimensioni [spatial_size, kernel_elements]
                            // ma noi abbiamo la colonna specifica
                            int col_idx = (kh * kernel_size_ + kw) * input_channels_ + c;
                            im(input_idx) += dCol(patch_idx, col_idx);
                        }
                    }
                }
            }
        }
    }
    
    return im;
}

// ============================================================================
// Attivazioni
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
// Gestione pesi
// ============================================================================
Eigen::MatrixXd Conv2DLayer::get_weights() const {
    if (use_bias_) {
        // kernels_ [filters, kernel_elements], bias_ [filters]
        // Restituisce [filters, kernel_elements + 1]
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
    return kernels_.size() + bias_.size();
}

// ============================================================================
// SERIALIZZAZIONE
// ============================================================================
void Conv2DLayer::serialize(std::ostream& out) const {
    //std::cout << "\n=== Conv2DLayer::serialize ===" << std::endl;
    //std::cout << "input_size_: " << input_size_ << std::endl;
    //std::cout << "filters_: " << filters_ << std::endl;
    //std::cout << "kernel_elements_: " << kernel_elements_ << std::endl;
    //std::cout << "use_bias_: " << use_bias_ << std::endl;
    
    uint32_t version = get_version();
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_height_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_width_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_channels_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&output_height_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&output_width_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&kernel_elements_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&filters_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&strides_), sizeof(int));
    
    size_t pad_len = padding_.size();
    out.write(reinterpret_cast<const char*>(&pad_len), sizeof(size_t));
    out.write(padding_.data(), pad_len);
    
    size_t act_len = activation_.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(activation_.data(), act_len);
    
    out.write(reinterpret_cast<const char*>(&use_bias_), sizeof(bool));
    
    // USIAMO ROW-MAJOR PER LA MASSIMA COMPATIBILITÀ DI MEMORIA
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights_to_save = get_weights();
    int rows = weights_to_save.rows();
    int cols = weights_to_save.cols();
    //std::cout << "weights_to_save rows: " << rows << ", cols: " << cols << std::endl;
    //std::cout << "weights_to_save sum: " << weights_to_save.sum() << std::endl;
    
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(weights_to_save.data()), rows * cols * sizeof(double));
}

void Conv2DLayer::deserialize(std::istream& in) {
    //std::cout << "\n=== Conv2DLayer::deserialize ===" << std::endl;
    
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    //std::cout << "version: " << version << std::endl;
    
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_height_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_width_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_channels_), sizeof(int));
    in.read(reinterpret_cast<char*>(&output_height_), sizeof(int));
    in.read(reinterpret_cast<char*>(&output_width_), sizeof(int));
    in.read(reinterpret_cast<char*>(&kernel_elements_), sizeof(int));
    in.read(reinterpret_cast<char*>(&filters_), sizeof(int));
    in.read(reinterpret_cast<char*>(&kernel_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&strides_), sizeof(int));
    
    //std::cout << "DEBUG POST-LOAD:" << std::endl;
    //std::cout << "input_channels_: " << input_channels_ << std::endl;
    //std::cout << "input_height_: " << input_height_ << std::endl;
    //std::cout << "input_width_: " << input_width_ << std::endl;
    //std::cout << "strides_: " << strides_ << std::endl;
    //std::cout << "padding_: " << padding_ << std::endl;
    //std::cout << "Letto input_size_: " << input_size_ << std::endl;
    //std::cout << "Letto filters_: " << filters_ << std::endl;
    //std::cout << "Letto kernel_elements_: " << kernel_elements_ << std::endl;
    
    size_t pad_len;
    in.read(reinterpret_cast<char*>(&pad_len), sizeof(size_t));
    std::vector<char> pad_buf(pad_len);
    in.read(pad_buf.data(), pad_len);
    padding_.assign(pad_buf.data(), pad_len);
    //std::cout << "padding_: " << padding_ << std::endl;
    
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    std::vector<char> act_buf(act_len);
    in.read(act_buf.data(), act_len);
    activation_.assign(act_buf.data(), act_len);
    //std::cout << "activation_: " << activation_ << std::endl;
    
    in.read(reinterpret_cast<char*>(&use_bias_), sizeof(bool));
    //std::cout << "use_bias_: " << use_bias_ << std::endl;
    
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    //std::cout << "Letto weights rows: " << rows << ", cols: " << cols << std::endl;
    
    // LEGGI FORZANDO ROW-MAJOR
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    //std::cout << "Loaded weights sum: " << loaded_weights.sum() << std::endl;
    
    // Inizializza kernels_ e bias_ prima di chiamare set_weights
    kernels_.resize(filters_, kernel_elements_);
    if (use_bias_) bias_.resize(filters_);
    
    set_weights(loaded_weights);
    
    // Verifica che i pesi siano stati impostati correttamente
    Eigen::MatrixXd check_weights = get_weights();
    //std::cout << "After set_weights - check_weights sum: " << check_weights.sum() << std::endl;
    
    // Re-inizializza i gradienti con la dimensione corretta
    int grad_cols = kernel_elements_ + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(filters_, grad_cols);
    weights_gradient_.setZero();
    bias_gradient_.resize(0);
    
    // Ricalcola le dimensioni di output
    compute_output_dimensions();
    //std::cout << "output_height_: " << output_height_ << ", output_width_: " << output_width_ << std::endl;
    
    //cache_ = std::make_shared<ConvCache>();
    cache_ = nullptr;
}


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