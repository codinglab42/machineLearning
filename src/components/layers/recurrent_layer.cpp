#include "components/layers/recurrent_layer.h"
#include "utils/serializable.h"

namespace layers {

    RecurrentLayer::RecurrentLayer(int hidden_size, int input_size)
        : hidden_size_(hidden_size), 
          input_size_(input_size),
          sequence_length_(0) {
        
        ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", "RecurrentLayer");
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "RecurrentLayer");
        
        // Stato iniziale azzerato
        h0_ = Eigen::MatrixXd();
    }

    void RecurrentLayer::set_initial_state(const Eigen::MatrixXd& h0) {
        ML_CHECK_DIMENSIONS(h0.rows(), 1, h0.cols(), hidden_size_,
                           "initial state", "RecurrentLayer");
        h0_ = h0;
    }

    void RecurrentLayer::initialize_cache(int batch_size) {
        // Metodo virtuale puro - deve essere implementato dalle derivate
    }

    Eigen::MatrixXd RecurrentLayer::extract_timestep(const Eigen::MatrixXd& input, int t) const {
        // Assume input formato [batch * seq_len, features] oppure [batch, seq_len * features]
        // Dipende da come organizzi i dati - questo è un esempio
        int batch_size = input.rows();
        return input.block(0, t * input_size_, batch_size, input_size_);
    }

    int RecurrentLayer::get_parameter_count() const {
        // Pesi: Wx (input_size * hidden_size) + Wh (hidden_size * hidden_size) + b (hidden_size)
        return input_size_ * hidden_size_ + hidden_size_ * hidden_size_ + hidden_size_;
    }

    void RecurrentLayer::serialize(std::ostream& out) const {
        using namespace utils;
        
        // Serializza parametri comuni
        out.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        
        // Serializza stato iniziale (se presente)
        bool has_initial = h0_.size() > 0;
        out.write(reinterpret_cast<const char*>(&has_initial), sizeof(bool));
        if (has_initial) {
            serialize_matrix(out, h0_);
        }
    }

    void RecurrentLayer::deserialize(std::istream& in) {
        using namespace utils;
        
        in.read(reinterpret_cast<char*>(&hidden_size_), sizeof(int));
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        
        bool has_initial;
        in.read(reinterpret_cast<char*>(&has_initial), sizeof(bool));
        if (has_initial) {
            h0_ = deserialize_matrix(in);
        } else {
            h0_ = Eigen::MatrixXd();
        }
    }

} // namespace layers
