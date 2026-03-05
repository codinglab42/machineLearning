#include "components/cache/batchnorm_cache.h"
#include <stdexcept>

namespace layers {

    BatchNormCache::BatchNormCache() : training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        x_centered.resize(0, 0);
        x_norm.resize(0, 0);
        batch_mean.resize(0);
        batch_var.resize(0);
        inv_std.resize(0);
    }

    void BatchNormCache::clear() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        x_centered.resize(0, 0);
        x_norm.resize(0, 0);
        batch_mean.resize(0);
        batch_var.resize(0);
        inv_std.resize(0);
        training = false;
    }

    bool BatchNormCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        
        return validate_dimensions();
    }

    bool BatchNormCache::validate_dimensions() const {
        // Verifica dimensioni di input e output
        if (input_cache.rows() != output_cache.rows() || 
            input_cache.cols() != output_cache.cols()) {
            return false;
        }
        
        int feature_size = input_cache.cols();
        
        // In training mode, verifica tutti i dati necessari
        if (training) {
            if (x_centered.rows() != input_cache.rows() || 
                x_centered.cols() != feature_size) {
                return false;
            }
            
            if (x_norm.rows() != input_cache.rows() || 
                x_norm.cols() != feature_size) {
                return false;
            }
            
            if (batch_mean.size() != feature_size) {
                return false;
            }
            
            if (batch_var.size() != feature_size) {
                return false;
            }
            
            if (inv_std.size() != feature_size) {
                return false;
            }
        }
        
        return true;
    }

} // namespace layers
