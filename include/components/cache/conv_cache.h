#ifndef CONV_CACHE_H
#define CONV_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class ConvCache : public LayerCache {
    public:
        ConvCache();
        ~ConvCache() override = default;
        
        // Implementazione interfaccia
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "ConvCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return true; }
        
        // Dati specifici della convoluzione
        Eigen::MatrixXd input_cache;      // Input originale
        Eigen::MatrixXd z_cache;          // Output pre-attivazione
        Eigen::MatrixXd output_cache;      // Output post-attivazione
        Eigen::MatrixXd col_cache;         // Matrice im2col
        
        // Dimensioni
        int input_height;
        int input_width;
        int input_channels;
        int output_height;
        int output_width;
        int filters;
        int batch_size;
        
        // Parametri convoluzione
        int kernel_size;
        int strides;
        std::string padding;
        
        // Metodi per impostare le dimensioni
        void set_input_shape(int h, int w, int c);
        void set_output_shape(int h, int w, int f);
        void set_batch_size(int bs);
        void set_kernel_info(int k_size, int str, const std::string& pad);
        
        // Accesso modificabile
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_z() { return z_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        Eigen::MatrixXd& mutable_col() { return col_cache; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif
