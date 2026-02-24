#include "components/layers/pooling.h"
#include "utils/serializable.h"
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>

using namespace Eigen;
using namespace layers;

// Costruttore
Pooling::Pooling(int pool_size, int stride, PoolType type, int channels)
    : pool_size_(pool_size), stride_(stride), channels_(channels), 
      pool_type_(type), input_height_(0), input_width_(0) {
    
    ML_CHECK_PARAM(pool_size > 0, "pool_size", "must be > 0", "Pooling");
    ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", "Pooling");
    ML_CHECK_PARAM(channels > 0, "channels", "must be > 0", "Pooling");
    
    clear_cache();
}

// Validazione dimensioni input
void Pooling::validate_input_shape(int total_elements) const {
    if (input_height_ <= 0 || input_width_ <= 0) {
        throw std::runtime_error("Pooling layer: input dimensions not set. Call set_input_shape() first or ensure forward receives valid dimensions.");
    }
    
    int expected_elements = channels_ * input_height_ * input_width_;
    if (total_elements != expected_elements) {
        throw ml_exception::DimensionMismatchException(
            "input dimensions",
            expected_elements, 1,
            total_elements, 1,
            "Pooling");
    }
}

// Estrae un canale specifico dall'input flatten
MatrixXd Pooling::extract_channel(const MatrixXd& input, int batch, int channel) const {
    MatrixXd channel_data(input_height_, input_width_);
    
    int channel_offset = channel * input_height_ * input_width_;
    for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
            int idx = channel_offset + h * input_width_ + w;
            channel_data(h, w) = input(batch, idx);
        }
    }
    
    return channel_data;
}

// Inserisce un canale nell'output flatten
void Pooling::insert_channel(MatrixXd& output, int batch, int channel, const MatrixXd& channel_data) const {
    int channel_offset = channel * input_height_ * input_width_;
    for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
            int idx = channel_offset + h * input_width_ + w;
            output(batch, idx) = channel_data(h, w);
        }
    }
}

// Calcola dimensioni output
int Pooling::get_output_height() const {
    if (input_height_ <= 0) return -1;
    return (input_height_ - pool_size_) / stride_ + 1;
}

int Pooling::get_output_width() const {
    if (input_width_ <= 0) return -1;
    return (input_width_ - pool_size_) / stride_ + 1;
}


int Pooling::get_output_size() const {
    if (input_height_ <= 0 || input_width_ <= 0) return -1;
    int output_height = (input_height_ - pool_size_) / stride_ + 1;
    int output_width = (input_width_ - pool_size_) / stride_ + 1;
    return channels_ * output_height * output_width;
}

std::vector<int> Pooling::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() >= 3) {
        // Formato [channels, height, width]
        int oh = (input_shape[1] - pool_size_) / stride_ + 1;
        int ow = (input_shape[2] - pool_size_) / stride_ + 1;
        return {input_shape[0], oh, ow};
    } else if (input_shape.size() == 1 && input_height_ > 0 && input_width_ > 0) {
        // Formato flatten con dimensioni memorizzate
        int oh = get_output_height();
        int ow = get_output_width();
        return {channels_ * oh * ow};
    }
    return {};
}

// Forward pass - Versione migliorata
MatrixXd Pooling::forward(const MatrixXd& input) {
    int batch_size = input.rows();
    int total_input_elements = input.cols();
    
    // Se non abbiamo ancora le dimensioni, cerchiamo di derivarle
    if (input_height_ <= 0 || input_width_ <= 0) {
        // Prova a derivare dimensioni quadrate (per retrocompatibilità)
        int spatial_elements = total_input_elements / channels_;
        int spatial_size = static_cast<int>(std::sqrt(spatial_elements));
        
        if (spatial_size * spatial_size == spatial_elements) {
            // È quadrato, possiamo usare questo
            input_height_ = spatial_size;
            input_width_ = spatial_size;
        } else {
            throw std::runtime_error(
                "Pooling layer: cannot determine input dimensions. "
                "Please call set_input_shape(height, width) before forward pass.");
        }
    }
    
    // Valida che le dimensioni corrispondano
    validate_input_shape(total_input_elements);
    
    // Calcola dimensioni output
    int output_height = get_output_height();
    int output_width = get_output_width();
    int output_size = channels_ * output_height * output_width;
    
    // Salva input nella cache e aggiorna shape
    cache_.input = input;
    cache_.has_activation = false;
    input_shape_ = InputShape(batch_size, channels_, input_height_, input_width_);
    
    // Pooling
    MatrixXd output(batch_size, output_size);
    max_indices_.clear();
    max_indices_.resize(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        max_indices_[b].resize(channels_ * output_height * output_width, -1);
        
        for (int c = 0; c < channels_; ++c) {
            // Estrai canale corrente
            MatrixXd channel_input = extract_channel(input, b, c);
            
            // Applica pooling
            MatrixXd pooled = pool_2d(channel_input, c, b);
            
            // Inserisci nell'output
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = c * output_height * output_width + oh * output_width + ow;
                    output(b, idx) = pooled(oh, ow);
                }
            }
        }
    }
    
    cache_.output = output;
    return output;
}

// Pooling 2D
MatrixXd Pooling::pool_2d(const MatrixXd& input, int channel, int batch) {
    int input_height = input.rows();
    int input_width = input.cols();
    int output_height = (input_height - pool_size_) / stride_ + 1;
    int output_width = (input_width - pool_size_) / stride_ + 1;
    
    MatrixXd output(output_height, output_width);
    
    for (int oh = 0; oh < output_height; ++oh) {
        for (int ow = 0; ow < output_width; ++ow) {
            int start_h = oh * stride_;
            int start_w = ow * stride_;
            int end_h = std::min(start_h + pool_size_, input_height);
            int end_w = std::min(start_w + pool_size_, input_width);
            
            if (pool_type_ == MAX) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_idx = -1;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        if (input(h, w) > max_val) {
                            max_val = input(h, w);
                            max_idx = h * input_width + w;
                        }
                    }
                }
                
                output(oh, ow) = max_val;
                
                // Salva indice per backward
                int output_idx = channel * output_height * output_width + oh * output_width + ow;
                if (batch < static_cast<int>(max_indices_.size()) && 
                    output_idx < static_cast<int>(max_indices_[batch].size())) {
                    max_indices_[batch][output_idx] = max_idx;
                }
            } else if (pool_type_ == AVERAGE) {
                double sum = 0.0;
                int count = 0;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        sum += input(h, w);
                        count++;
                    }
                }
                
                output(oh, ow) = sum / count;
            }
        }
    }
    
    return output;
}

// Backward pass
MatrixXd Pooling::backward(const MatrixXd& gradient, double learning_rate) {
    if (gradient.rows() != cache_.input.rows()) {
        throw ml_exception::DimensionMismatchException(
            "gradient rows",
            cache_.input.rows(), 1,
            gradient.rows(), 1,
            "Pooling");
    }
    
    int batch_size = gradient.rows();
    
    // Usa le dimensioni memorizzate
    int output_height = get_output_height();
    int output_width = get_output_width();
    int total_input_elements = channels_ * input_height_ * input_width_;
    
    // Gradiente rispetto all'input
    MatrixXd dInput = MatrixXd::Zero(batch_size, total_input_elements);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_; ++c) {
            // Estrai gradiente per questo canale
            MatrixXd grad_2d(output_height, output_width);
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    int idx = c * output_height * output_width + oh * output_width + ow;
                    grad_2d(oh, ow) = gradient(b, idx);
                }
            }
            
            // Calcola gradiente backward
            MatrixXd dInput_2d = pool_backward_2d(grad_2d, c, b);
            
            // Inserisci nell'output
            insert_channel(dInput, b, c, dInput_2d);
        }
    }
    
    return dInput;
}

// Backward 2D
MatrixXd Pooling::pool_backward_2d(const MatrixXd& gradient, int channel, int batch) {
    int output_height = gradient.rows();
    int output_width = gradient.cols();
    
    MatrixXd dInput = MatrixXd::Zero(input_height_, input_width_);
    
    if (pool_type_ == MAX) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int output_idx = channel * output_height * output_width + oh * output_width + ow;
                
                if (batch < static_cast<int>(max_indices_.size()) &&
                    output_idx < static_cast<int>(max_indices_[batch].size())) {
                    int max_idx = max_indices_[batch][output_idx];
                    
                    if (max_idx >= 0) {
                        int h = max_idx / input_width_;
                        int w = max_idx % input_width_;
                        if (h < input_height_ && w < input_width_) {
                            dInput(h, w) += gradient(oh, ow);
                        }
                    }
                }
            }
        }
    } else if (pool_type_ == AVERAGE) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int start_h = oh * stride_;
                int start_w = ow * stride_;
                int end_h = std::min(start_h + pool_size_, input_height_);
                int end_w = std::min(start_w + pool_size_, input_width_);
                
                double grad_val = gradient(oh, ow);
                int count = (end_h - start_h) * (end_w - start_w);
                double avg_grad = grad_val / count;
                
                for (int h = start_h; h < end_h; ++h) {
                    for (int w = start_w; w < end_w; ++w) {
                        dInput(h, w) += avg_grad;
                    }
                }
            }
        }
    }
    
    return dInput;
}

// Informazioni
int Pooling::get_input_size() const {

    // DEBUG: stampa i valori
    std::cout << "get_input_size: height=" << input_height_ << ", width=" << input_width_ << std::endl;

    // Usa input_height_ e input_width_ direttamente
    if (input_height_ <= 0 || input_width_ <= 0) return -1;
    return channels_ * input_height_ * input_width_;
}

std::string Pooling::get_config() const {
    std::ostringstream oss;
    oss << "Pooling(pool_size=" << pool_size_
        << ", stride=" << stride_
        << ", type=" << (pool_type_ == MAX ? "max" : "average")
        << ", channels=" << channels_;
    
    // DEBUG: stampa i valori
    std::cout << "get_config: height=" << input_height_ << ", width=" << input_width_ << std::endl;

    // Usa input_height_ e input_width_ direttamente
    if (input_height_ > 0 && input_width_ > 0) {
        oss << ", input_shape=" << input_height_ << "x" << input_width_;
    } else {
        oss << ", input_shape=not_set";
    }
    oss << ")";
    return oss.str();
}

// Cache management
void Pooling::clear_cache() {
    cache_.input = MatrixXd();
    cache_.output = MatrixXd();
    cache_.has_activation = false;
    max_indices_.clear();

    // input_height_ = 0;
    // input_width_ = 0;
    // input_shape_ = InputShape();
}

// Serializzazione
void Pooling::serialize(std::ostream& out) const {
    using namespace utils;
    
    out.write(reinterpret_cast<const char*>(&pool_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&channels_), sizeof(int));
    
    int type_int = static_cast<int>(pool_type_);
    out.write(reinterpret_cast<const char*>(&type_int), sizeof(int));
    
    // Salva anche le dimensioni se sono state impostate
    out.write(reinterpret_cast<const char*>(&input_height_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_width_), sizeof(int));

    // DEBUG: stampa cosa stiamo salvando
    std::cout << "Serializing: height=" << input_height_ << ", width=" << input_width_ << std::endl;

}

void Pooling::deserialize(std::istream& in) {
    using namespace utils;

    clear_cache();
    
    in.read(reinterpret_cast<char*>(&pool_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
    in.read(reinterpret_cast<char*>(&channels_), sizeof(int));
    
    int type_int;
    in.read(reinterpret_cast<char*>(&type_int), sizeof(int));
    pool_type_ = static_cast<PoolType>(type_int);
    
    // Carica le dimensioni
    in.read(reinterpret_cast<char*>(&input_height_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_width_), sizeof(int));
    
       
    // Aggiorna input_shape_ con gli stessi valori
    input_shape_ = InputShape(0, channels_, input_height_, input_width_);

    
    
    
}