#include "components/layers/recurrent_layer.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace layers {

void SequenceTensor::validate() const {
    if (batch_size <= 0 || time_steps <= 0 || features <= 0 ||
        (!is_sequence && time_steps != 1)) {
        throw std::invalid_argument("Invalid sequence tensor dimensions");
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto steps = static_cast<std::size_t>(time_steps);
    const auto width = static_cast<std::size_t>(features);
    if (steps > std::numeric_limits<std::size_t>::max() / batch ||
        width > std::numeric_limits<std::size_t>::max() / (batch * steps) ||
        data.size() != batch * steps * width) {
        throw std::invalid_argument("Sequence tensor buffer size does not match dimensions");
    }
}

namespace {

Eigen::MatrixXd read_step(const SequenceTensor& tensor, int step) {
    Eigen::MatrixXd result(tensor.batch_size, tensor.features);
    for (int b = 0; b < tensor.batch_size; ++b) {
        for (int f = 0; f < tensor.features; ++f) {
            const auto offset = (static_cast<std::size_t>(b) * tensor.time_steps + step)
                                * tensor.features + f;
            result(b, f) = tensor.data[offset];
        }
    }
    return result;
}

void write_step(SequenceTensor& tensor, int step, const Eigen::MatrixXd& values) {
    for (int b = 0; b < tensor.batch_size; ++b) {
        for (int f = 0; f < tensor.features; ++f) {
            const auto offset = (static_cast<std::size_t>(b) * tensor.time_steps + step)
                                * tensor.features + f;
            tensor.data[offset] = values(b, f);
        }
    }
}

} // namespace

SequenceTensor RecurrentLayer::forward_sequence(const SequenceTensor& input, bool training) {
    input.validate();
    if (!input.is_sequence || input.features != get_input_size()) {
        throw std::invalid_argument("Recurrent sequence input must be 3D with the expected feature count");
    }
    reset_state();
    std::vector<std::shared_ptr<LayerCache>> step_caches;
    if (training) step_caches.reserve(static_cast<std::size_t>(input.time_steps));
    const int output_steps = return_sequences_ ? input.time_steps : 1;
    SequenceTensor output{{}, input.batch_size, output_steps, get_output_size(), return_sequences_};
    output.data.resize(static_cast<std::size_t>(output.batch_size) * output_steps * output.features);
    for (int t = 0; t < input.time_steps; ++t) {
        const Eigen::MatrixXd value = forward(read_step(input, t), training);
        if (training) {
            step_caches.push_back(get_cache());
            set_cache(nullptr);
        }
        if (return_sequences_ || t == input.time_steps - 1) {
            write_step(output, return_sequences_ ? t : 0, value);
        }
    }
    if (training) set_cache(step_caches.back());
    sequence_caches_ = std::move(step_caches);
    sequence_batch_size_ = input.batch_size;
    sequence_time_steps_ = input.time_steps;
    sequence_input_size_ = input.features;
    sequence_returns_all_ = return_sequences_;
    last_forward_was_sequence_ = true;
    return output;
}

SequenceTensor RecurrentLayer::backward_sequence(const SequenceTensor& gradient) {
    gradient.validate();
    if (sequence_caches_.size() != static_cast<std::size_t>(sequence_time_steps_) ||
        sequence_time_steps_ <= 0 ||
        gradient.batch_size != sequence_batch_size_ ||
        gradient.features != get_output_size() ||
        gradient.is_sequence != sequence_returns_all_ ||
        gradient.time_steps != (sequence_returns_all_ ? sequence_time_steps_ : 1)) {
        throw std::invalid_argument("Sequence gradient does not match a training forward pass");
    }
    SequenceTensor result{{}, sequence_batch_size_, sequence_time_steps_, sequence_input_size_, true};
    result.data.resize(static_cast<std::size_t>(result.batch_size) * result.time_steps * result.features);
    const Eigen::MatrixXd weights = get_weights();
    Eigen::MatrixXd weight_gradient = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    Eigen::VectorXd bias_gradient = Eigen::VectorXd::Zero(get_use_bias() ? get_biases().size() : 0);
    Eigen::MatrixXd hidden_gradient = Eigen::MatrixXd::Zero(sequence_batch_size_, get_output_size());
    Eigen::MatrixXd cell_gradient = Eigen::MatrixXd::Zero(sequence_batch_size_, get_output_size());
    for (int t = sequence_time_steps_ - 1; t >= 0; --t) {
        set_cache(sequence_caches_[static_cast<std::size_t>(t)]);
        if (sequence_returns_all_) {
            hidden_gradient += read_step(gradient, t);
        } else if (t == sequence_time_steps_ - 1) {
            hidden_gradient += read_step(gradient, 0);
        }
        Eigen::MatrixXd previous_hidden_gradient;
        Eigen::MatrixXd previous_cell_gradient;
        const Eigen::MatrixXd input_gradient = backward_step(
            hidden_gradient, cell_gradient, previous_hidden_gradient, previous_cell_gradient);
        weight_gradient += get_weights_gradient();
        if (get_use_bias()) {
            bias_gradient += get_bias_gradient();
        }
        write_step(result, t, input_gradient);
        hidden_gradient = std::move(previous_hidden_gradient);
        if (previous_cell_gradient.size() != 0) {
            cell_gradient = std::move(previous_cell_gradient);
        }
    }
    set_weights_gradient(weight_gradient);
    if (get_use_bias()) {
        set_bias_gradient(bias_gradient);
    }
    set_cache(sequence_caches_.back());
    return result;
}

} // namespace layers
