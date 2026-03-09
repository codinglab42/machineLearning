#include "components/layers/pooling_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>
#include <limits>

namespace layers {

    PoolingLayer::PoolingLayer(int pool_size, int stride, PoolType type, int channels)
        : pool_size_(pool_size), stride_(stride), channels_(channels), 
          pool_type_(type), input_height_(0), input_width_(0), input_size_(0) {
        
        ML_CHECK_PARAM(pool_size > 0, "pool_size", "must be > 0", "Pooling");
        ML_CHECK_PARAM(stride > 0, "stride", "must be > 0", "Pooling");
        ML_CHECK_PARAM(channels > 0, "channels", "must be > 0", "Pooling");
        
        cache_ = nullptr;
    }

    void PoolingLayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "Pooling");
        
        input_size_ = input_size;
        int spatial_elements = input_size / channels_;
        input_height_ = static_cast<int>(std::sqrt(spatial_elements));
        input_width_ = input_height_;
    }

    int PoolingLayer::get_output_height() const {
        if (input_height_ <= 0) return -1;
        return (input_height_ - pool_size_) / stride_ + 1;
    }

    int PoolingLayer::get_output_width() const {
        if (input_width_ <= 0) return -1;
        return (input_width_ - pool_size_) / stride_ + 1;
    }

    int PoolingLayer::get_input_size() const {
        return input_size_;
    }

    int PoolingLayer::get_output_size() const {
        if (input_height_ <= 0 || input_width_ <= 0) return -1;
        int oh = get_output_height();
        int ow = get_output_width();
        return channels_ * oh * ow;
    }

    Eigen::MatrixXd PoolingLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    Eigen::MatrixXd PoolingLayer::forward(const Eigen::MatrixXd& input, bool training) {
        ML_CHECK_NOT_EMPTY(input, "input", "Pooling");
        
        if (!cache_) {
            cache_ = std::make_shared<PoolingCache>();
        }
        
        int batch_size = input.rows();
        input_size_ = input.cols();
        
        if (input_height_ <= 0 || input_width_ <= 0) {
            int spatial_elements = input_size_ / channels_;
            input_height_ = static_cast<int>(std::sqrt(spatial_elements));
            input_width_ = input_height_;
        }
        
        if (input_size_ != channels_ * input_height_ * input_width_) {
            ML_THROW_DIMENSION_MISMATCH("forward input",
                batch_size, channels_ * input_height_ * input_width_,
                batch_size, input_size_, "Pooling");
        }
        
        cache_->set_input(input);
        cache_->set_training(training);
        cache_->set_input_shape(input_height_, input_width_, channels_);
        
        int oh = get_output_height();
        int ow = get_output_width();
        int output_size = channels_ * oh * ow;
        
        Eigen::MatrixXd output(batch_size, output_size);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels_; ++c) {
                Eigen::MatrixXd channel = extract_channel(input, b, c);
                
                for (int i = 0; i < oh; ++i) {
                    for (int j = 0; j < ow; ++j) {
                        int h_start = i * stride_;
                        int w_start = j * stride_;
                        
                        if (pool_type_ == MAX) {
                            double max_val = -std::numeric_limits<double>::infinity();
                            int max_h = 0, max_w = 0;
                            
                            for (int h = h_start; h < h_start + pool_size_ && h < input_height_; ++h) {
                                for (int w = w_start; w < w_start + pool_size_ && w < input_width_; ++w) {
                                    if (channel(h, w) > max_val) {
                                        max_val = channel(h, w);
                                        max_h = h;
                                        max_w = w;
                                    }
                                }
                            }
                            
                            output(b, c * oh * ow + i * ow + j) = max_val;
                            if (training) {
                                cache_->add_max_index(b, c, i, j, max_h, max_w);
                            }
                        } else { // AVERAGE
                            double sum = 0.0;
                            int count = 0;
                            
                            for (int h = h_start; h < h_start + pool_size_ && h < input_height_; ++h) {
                                for (int w = w_start; w < w_start + pool_size_ && w < input_width_; ++w) {
                                    sum += channel(h, w);
                                    count++;
                                }
                            }
                            
                            output(b, c * oh * ow + i * ow + j) = sum / count;
                        }
                    }
                }
            }
        }
        
        cache_->set_output(output);
        return output;
    }

    Eigen::MatrixXd PoolingLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("Pooling", "cache not initialized. Call forward first.");
        }
        
        if (!cache_->get_training()) {
            return Eigen::MatrixXd::Zero(cache_->get_input().rows(), cache_->get_input().cols());
        }
        
        int batch_size = gradient.rows();
        int oh = get_output_height();
        int ow = get_output_width();
        
        if (gradient.cols() != channels_ * oh * ow) {
            ML_THROW_DIMENSION_MISMATCH("backward gradient",
                batch_size, channels_ * oh * ow,
                batch_size, gradient.cols(), "Pooling");
        }
        
        Eigen::MatrixXd dInput = Eigen::MatrixXd::Zero(batch_size, input_size_);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels_; ++c) {
                Eigen::MatrixXd dChannel = Eigen::MatrixXd::Zero(input_height_, input_width_);
                
                if (pool_type_ == MAX) {
                    const auto& indices = cache_->get_max_indices();
                    for (const auto& idx : indices) {
                        if (idx.batch == b && idx.channel == c) {
                            int grad_idx = c * oh * ow + idx.output_row * ow + idx.output_col;
                            dChannel(idx.input_h, idx.input_w) += gradient(b, grad_idx);
                        }
                    }
                } else { // AVERAGE
                    for (int i = 0; i < oh; ++i) {
                        for (int j = 0; j < ow; ++j) {
                            int h_start = i * stride_;
                            int w_start = j * stride_;
                            int grad_idx = c * oh * ow + i * ow + j;
                            double grad_val = gradient(b, grad_idx);
                            
                            int count = 0;
                            for (int h = h_start; h < h_start + pool_size_ && h < input_height_; ++h) {
                                for (int w = w_start; w < w_start + pool_size_ && w < input_width_; ++w) {
                                    count++;
                                }
                            }
                            
                            double avg_grad = grad_val / count;
                            for (int h = h_start; h < h_start + pool_size_ && h < input_height_; ++h) {
                                for (int w = w_start; w < w_start + pool_size_ && w < input_width_; ++w) {
                                    dChannel(h, w) += avg_grad;
                                }
                            }
                        }
                    }
                }
                
                for (int h = 0; h < input_height_; ++h) {
                    for (int w = 0; w < input_width_; ++w) {
                        int idx = c * input_height_ * input_width_ + h * input_width_ + w;
                        dInput(b, idx) = dChannel(h, w);
                    }
                }
            }
        }
        
        return dInput;
    }

    Eigen::MatrixXd PoolingLayer::extract_channel(const Eigen::MatrixXd& input, int batch, int channel) const {
        Eigen::MatrixXd channel_data(input_height_, input_width_);
        
        int offset = channel * input_height_ * input_width_;
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                channel_data(h, w) = input(batch, offset + h * input_width_ + w);
            }
        }
        
        return channel_data;
    }

    void PoolingLayer::insert_channel(Eigen::MatrixXd& output, int batch, int channel, const Eigen::MatrixXd& channel_data) const {
        int offset = channel * input_height_ * input_width_;
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                output(batch, offset + h * input_width_ + w) = channel_data(h, w);
            }
        }
    }

    void PoolingLayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&pool_size_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&stride_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&channels_), sizeof(int));
        int type = static_cast<int>(pool_type_);
        out.write(reinterpret_cast<const char*>(&type), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_height_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_width_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    }

    void PoolingLayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        
        in.read(reinterpret_cast<char*>(&pool_size_), sizeof(int));
        in.read(reinterpret_cast<char*>(&stride_), sizeof(int));
        in.read(reinterpret_cast<char*>(&channels_), sizeof(int));
        int type;
        in.read(reinterpret_cast<char*>(&type), sizeof(int));
        pool_type_ = static_cast<PoolType>(type);
        in.read(reinterpret_cast<char*>(&input_height_), sizeof(int));
        in.read(reinterpret_cast<char*>(&input_width_), sizeof(int));
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    }

    std::string PoolingLayer::get_config() const {
        std::ostringstream oss;
        oss << "Pooling(pool_size=" << pool_size_
            << ", stride=" << stride_
            << ", type=" << (pool_type_ == MAX ? "max" : "average")
            << ", channels=" << channels_
            << ", input_size=" << input_size_ << ")";
        return oss.str();
    }

} // namespace layers
