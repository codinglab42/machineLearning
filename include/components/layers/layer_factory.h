#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include <memory>
#include <unordered_map>
#include <functional>
#include "layer.h"
#include "utils/serializable.h"
#include "exceptions/exception_macros.h"

namespace layers {

class LayerFactory {
public:
    using Creator = std::function<std::unique_ptr<Layer>()>;
    
    // Registrazione
    static void register_layer(LayerType type, Creator creator, const std::string& name);
    
    // Creazione
    static std::unique_ptr<Layer> create(LayerType type);
    static std::unique_ptr<Layer> create(const std::string& name);
    
    // Serializzazione del tipo
    static void serialize_type(std::ostream& out, LayerType type);
    static LayerType deserialize_type(std::istream& in);
    
    // Utility
    static std::string get_name(LayerType type);
    static void register_all_layers();

private:
    static std::unordered_map<LayerType, Creator>& get_creators();
    static std::unordered_map<LayerType, std::string>& get_names();
    static std::unordered_map<std::string, LayerType>& get_type_map();
    static bool& is_registered();
};

} // namespace layers

#endif // LAYER_FACTORY_H