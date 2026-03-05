#include "components/layers/conv2d_layer.h"

namespace layers {

    Conv2DLayer::Conv2DLayer(int filters, int kernel_size, int strides,
                           const std::string& padding,
                           const std::string& activation)
        : filters_(filters),
          kernel_size_(kernel_size),
          strides_(strides),
          padding_(padding),
          activation_(activation),
          input_size_(0),
          input_height_(0),
          input_width_(0),
          input_channels_(0),
          output_height_(0),
          output_width_(0),
          kernel_elements_(0) {
        
        if (filters <= 0) throw std::invalid_argument("filters must be > 0");
        if (kernel_size <= 0) throw std::invalid_argument("kernel_size must be > 0");
        if (strides <= 0) throw std::invalid_argument("strides must be > 0");
        
        // Inizializza cache come nullptr
        cache_ = nullptr;
    }

    // Versione base di forward - chiama quella con training=false
    Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    void Conv2DLayer::compute_output_dimensions() {
        if (padding_ == "same") {
            output_height_ = static_cast<int>(std::ceil(static_cast<double>(input_height_) / strides_));
            output_width_ = static_cast<int>(std::ceil(static_cast<double>(input_width_) / strides_));
        } else { // "valid"
            output_height_ = (input_height_ - kernel_size_) / strides_ + 1;
            output_width_ = (input_width_ - kernel_size_) / strides_ + 1;
        }
        
        if (output_height_ <= 0 || output_width_ <= 0) {
            throw std::runtime_error("Invalid convolution dimensions: output size would be <= 0");
        }
    }

    void Conv2DLayer::set_input_shape(int input_size) {
        input_size_ = input_size;
        
        // Calcola dimensioni (assumiamo input quadrato con 1 canale per semplicità)
        // In una implementazione reale, queste info dovrebbero venire dal layer precedente
        input_height_ = static_cast<int>(std::sqrt(input_size));
        input_width_ = input_height_;
        input_channels_ = 1;
        
        compute_output_dimensions();
        
        kernel_elements_ = kernel_size_ * kernel_size_ * input_channels_;
        
        // Inizializza pesi (Xavier/Glorot initialization)
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
    }

    int Conv2DLayer::get_output_size() const {
        return output_height_ * output_width_ * filters_;
    }

    Eigen::MatrixXd Conv2DLayer::im2col(const Eigen::MatrixXd& input, int batch_idx, int start_idx) const {
        int batch_size = input.rows() / input_size_;
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

    Eigen::MatrixXd Conv2DLayer::col2im(const Eigen::MatrixXd& col, int batch_idx, int start_idx) const {
        int batch_size = col.rows() / (output_height_ * output_width_);
        int input_offset = batch_idx * input_size_ + start_idx;
        
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

    Eigen::MatrixXd Conv2DLayer::apply_activation(const Eigen::MatrixXd& z) const {
        if (activation_ == "relu") {
            return z.cwiseMax(0.0);
        } else if (activation_ == "sigmoid") {
            return 1.0 / (1.0 + (-z).array().exp());
        } else if (activation_ == "tanh") {
            return z.array().tanh();
        } else if (activation_ == "linear") {
            return z;
        } else {
            return z.cwiseMax(0.0); // default relu
        }
    }

    Eigen::MatrixXd Conv2DLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
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
        } else {
            return (z.array() > 0.0).cast<double>();
        }
    }

    Eigen::MatrixXd Conv2DLayer::forward(const Eigen::MatrixXd& input, bool training) {
        // Crea cache se non esiste
        if (!cache_) {
            cache_ = std::make_shared<ConvCache>();
        }
        
        int batch_size = input.rows() / input_size_;
        int output_size = get_output_size();
        
        // Inizializza cache con le dimensioni
        cache_->set_input_shape(input_height_, input_width_, input_channels_);
        cache_->set_output_shape(output_height_, output_width_, filters_);
        cache_->set_batch_size(batch_size);
        cache_->set_kernel_info(kernel_size_, strides_, padding_);
        
        // Alloca spazio nelle cache
        cache_->input_cache = input;
        cache_->z_cache.resize(batch_size * output_size, 1);
        cache_->output_cache.resize(batch_size * output_size, 1);
        cache_->col_cache.resize(batch_size, output_height_ * output_width_ * kernel_elements_);
        
        Eigen::MatrixXd z_output(batch_size * output_size, 1);
        
        for (int b = 0; b < batch_size; ++b) {
            // Trasforma input in colonne
            Eigen::MatrixXd cols = im2col(input, b, 0);
            
            // Salva le colonne nella cache (appiattite)
            Eigen::Map<Eigen::RowVectorXd> col_map(
                cache_->col_cache.row(b).data(), 
                cols.rows() * cols.cols());
            col_map = Eigen::Map<Eigen::RowVectorXd>(cols.data(), cols.rows() * cols.cols());
            
            // Convoluzione: output = cols * kernels_.transpose() + bias
            Eigen::MatrixXd conv_output = cols * kernels_.transpose();
            
            // Aggiungi bias e rimodella
            int output_offset = b * output_size;
            for (int f = 0; f < filters_; ++f) {
                for (int p = 0; p < conv_output.rows(); ++p) {
                    int output_idx = output_offset + p * filters_ + f;
                    conv_output(p, f) += bias_(f);
                    z_output(output_idx) = conv_output(p, f);
                }
            }
        }
        
        // Salva nella cache
        cache_->z_cache = z_output;
        cache_->output_cache = apply_activation(z_output);
        
        return cache_->output_cache;
    }

    Eigen::MatrixXd Conv2DLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            throw std::runtime_error("Conv2DLayer: cache not initialized. Call forward first.");
        }
        
        auto conv_cache = get_specific_cache();
        if (!conv_cache) {
            throw std::runtime_error("Conv2DLayer: invalid cache type");
        }
        
        int batch_size = conv_cache->batch_size;
        int output_size = get_output_size();
        
        // Recupera dati dalla cache
        const Eigen::MatrixXd& input = conv_cache->input_cache;
        const Eigen::MatrixXd& z = conv_cache->z_cache;
        
        // Gradiente rispetto all'output pre-attivazione
        Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(z).array();
        dZ.resize(batch_size * output_height_ * output_width_, filters_);
        
        // Gradiente per bias
        Eigen::VectorXd dbias = dZ.colwise().sum();
        
        // Gradiente per kernels
        Eigen::MatrixXd dKernels = Eigen::MatrixXd::Zero(kernels_.rows(), kernels_.cols());
        
        for (int b = 0; b < batch_size; ++b) {
            // Recupera le colonne dalla cache
            Eigen::Map<Eigen::MatrixXd> cols(
                conv_cache->col_cache.row(b).data(),
                output_height_ * output_width_,
                kernel_elements_);
            
            dKernels += dZ.block(b * output_height_ * output_width_, 0,
                                 output_height_ * output_width_, filters_).transpose() * cols;
        }
        
        // Gradiente rispetto all'input (da propagare indietro)
        Eigen::MatrixXd dX = Eigen::MatrixXd::Zero(input.rows(), 1);
        
        for (int b = 0; b < batch_size; ++b) {
            Eigen::Map<Eigen::MatrixXd> cols(
                conv_cache->col_cache.row(b).data(),
                output_height_ * output_width_,
                kernel_elements_);
            
            Eigen::MatrixXd dZ_b = dZ.block(b * output_height_ * output_width_, 0,
                                            output_height_ * output_width_, filters_);
            
            Eigen::MatrixXd dCol = dZ_b * kernels_;
            dX.block(b * input_size_, 0, input_size_, 1) += col2im(dCol, b, 0);
        }
        
        // Aggiorna pesi con learning rate
        kernels_ -= learning_rate * dKernels / batch_size;
        bias_ -= learning_rate * dbias / batch_size;
        
        return dX;
    }

    Eigen::MatrixXd Conv2DLayer::get_weights() const {
        Eigen::MatrixXd weights(kernels_.rows(), kernels_.cols() + 1);
        weights.leftCols(kernels_.cols()) = kernels_;
        weights.col(kernels_.cols()) = bias_;
        return weights;
    }

    void Conv2DLayer::set_weights(const Eigen::MatrixXd& weights) {
        if (weights.cols() != kernels_.cols() + 1) {
            throw std::invalid_argument("Conv2DLayer: invalid weights dimension");
        }
        kernels_ = weights.leftCols(kernels_.cols());
        bias_ = weights.col(kernels_.cols());
    }

    int Conv2DLayer::get_parameter_count() const {
        return kernels_.size() + bias_.size();
    }

    void Conv2DLayer::serialize(std::ostream& out) const {
        // Salva configurazione
        out << get_config() << std::endl;
        
        // Salva dimensioni
        out << input_size_ << " " << input_height_ << " " << input_width_ << " " << input_channels_ << std::endl;
        out << output_height_ << " " << output_width_ << std::endl;
        out << kernel_elements_ << std::endl;
        
        // Salva pesi
        for (int i = 0; i < kernels_.rows(); ++i) {
            for (int j = 0; j < kernels_.cols(); ++j) {
                out << kernels_(i, j) << " ";
            }
            out << std::endl;
        }
        
        // Salva bias
        for (int i = 0; i < bias_.size(); ++i) {
            out << bias_(i) << " ";
        }
        out << std::endl;
    }

    void Conv2DLayer::deserialize(std::istream& in) {
        // Carica dimensioni
        in >> input_size_ >> input_height_ >> input_width_ >> input_channels_;
        in >> output_height_ >> output_width_;
        in >> kernel_elements_;
        
        // Ridimensiona matrici
        kernels_.resize(filters_, kernel_elements_);
        bias_.resize(filters_);
        
        // Carica pesi
        for (int i = 0; i < kernels_.rows(); ++i) {
            for (int j = 0; j < kernels_.cols(); ++j) {
                in >> kernels_(i, j);
            }
        }
        
        // Carica bias
        for (int i = 0; i < bias_.size(); ++i) {
            in >> bias_(i);
        }
    }

    std::string Conv2DLayer::get_config() const {
        return "Conv2DLayer(filters=" + std::to_string(filters_) +
               ", kernel_size=" + std::to_string(kernel_size_) +
               ", strides=" + std::to_string(strides_) +
               ", padding=" + padding_ +
               ", activation=" + activation_ + ")";
    }

} // namespace layers