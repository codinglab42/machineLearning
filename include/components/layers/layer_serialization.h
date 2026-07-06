#ifndef LAYER_SERIALIZATION_H
#define LAYER_SERIALIZATION_H

#include <iostream>
#include "layer.h"
#include "layer_factory.h"
#include "utils/serializable.h"
#include "exceptions/exception_macros.h"

namespace layers {

// Serializza un layer completo (tipo + dati)
inline void serialize_layer(std::ostream& out, const Layer& layer) {
    using namespace utils;
    
    // Scrivi il tipo
    LayerFactory::serialize_type(out, layer.get_layer_type());
    
    // Scrivi la versione del layer
    write_scalar(out, layer.get_version());
    
    // Scrivi i dati specifici del layer
    layer.serialize(out);
    
    if (!out.good()) {
        ML_THROW_IO_ERROR("stream", "write", layer.get_type());
    }
}

// Deserializza un layer completo
inline std::unique_ptr<Layer> deserialize_layer(std::istream& in) {
    using namespace utils;
    
    // Leggi il tipo
    LayerType type = LayerFactory::deserialize_type(in);
    
    // Leggi la versione
    uint32_t version;
    read_scalar(in, version);
    
    // Crea il layer
    auto layer = LayerFactory::create(type);
    
    // Deserializza i dati
    layer->deserialize(in);
    
    if (!in.good()) {
        ML_THROW_IO_ERROR("stream", "read", layer->get_type());
    }
    
    return layer;
}

} // namespace layers

#endif // LAYER_SERIALIZATION_H