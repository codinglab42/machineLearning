#include "components/optimizers/optimizer.h"
#include "utils/serializable.h"

namespace models {

    Optimizer::Optimizer(double learning_rate, double decay)
        : learning_rate_(learning_rate), decay_(decay), iterations_(0) {}

    double Optimizer::get_current_learning_rate() const {
        if (decay_ == 0.0) {
            return learning_rate_;
        }
        return learning_rate_ / (1.0 + decay_ * iterations_);
    }

    void Optimizer::serialize(std::ostream& out) const {
        using namespace utils;
        
        out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&decay_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&iterations_), sizeof(int));
    }

    void Optimizer::deserialize(std::istream& in) {
        using namespace utils;
        
        in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
        in.read(reinterpret_cast<char*>(&decay_), sizeof(double));
        in.read(reinterpret_cast<char*>(&iterations_), sizeof(int));
    }

} // namespace models
