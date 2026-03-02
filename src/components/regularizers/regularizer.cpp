#include "components/regularizers/regularizer.h"
#include "utils/serializable.h"

namespace models {

    Regularizer::Regularizer(double strength)
        : strength_(strength) {}

    void Regularizer::serialize(std::ostream& out) const {
        using namespace utils;
        out.write(reinterpret_cast<const char*>(&strength_), sizeof(double));
    }

    void Regularizer::deserialize(std::istream& in) {
        using namespace utils;
        in.read(reinterpret_cast<char*>(&strength_), sizeof(double));
    }

} // namespace models
