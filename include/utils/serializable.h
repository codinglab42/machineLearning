#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <Eigen/Dense>
#include "exceptions/exception_macros.h"

namespace utils {

constexpr uint32_t SERIALIZATION_MAGIC = 0x4C4D4C42;
constexpr uint32_t SERIALIZATION_VERSION = 2;

// ============================================================================
// FUNZIONI DI BASE
// ============================================================================

inline void write_string(std::ostream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(str.data(), len);
}

inline std::string read_string(std::istream& in) {
    uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!in.good()) {
        ML_THROW_IO_ERROR("stream", "read", "Serialization");
    }
    std::string str(len, ' ');
    in.read(&str[0], len);
    return str;
}

template<typename T>
void write_scalar(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
void read_scalar(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

// ============================================================================
// FUNZIONI PER EIGEN
// ============================================================================

// Per matrici (Eigen::MatrixXd)
inline void write_eigen_matrix(std::ostream& out, const Eigen::MatrixXd& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    write_scalar(out, rows);
    write_scalar(out, cols);
    if (rows > 0 && cols > 0) {
        out.write(reinterpret_cast<const char*>(mat.data()), 
                  rows * cols * sizeof(double));
    }
}

inline void read_eigen_matrix(std::istream& in, Eigen::MatrixXd& mat) {
    int rows, cols;
    read_scalar(in, rows);
    read_scalar(in, cols);
    mat.resize(rows, cols);
    if (rows > 0 && cols > 0) {
        in.read(reinterpret_cast<char*>(mat.data()), 
                rows * cols * sizeof(double));
    }
}

// Per vettori (Eigen::VectorXd)
inline void write_eigen_vector(std::ostream& out, const Eigen::VectorXd& vec) {
    int size = vec.size();
    write_scalar(out, size);
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(vec.data()), 
                  size * sizeof(double));
    }
}

inline void read_eigen_vector(std::istream& in, Eigen::VectorXd& vec) {
    int size;
    read_scalar(in, size);
    vec.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(vec.data()), 
                size * sizeof(double));
    }
}

// ============================================================================
// CLASSI PER LA SERIALIZZAZIONE DEI MODELLI
// ============================================================================

class ISerializable {
public:
    virtual ~ISerializable() = default;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
    virtual std::string to_string() const = 0;
    virtual void serialize_binary(std::ostream& out) const = 0;
    virtual void deserialize_binary(std::istream& in) = 0;
    virtual std::string get_model_type() const = 0;
};

template<typename Model>
class BinarySerializer {
public:
    static void save(const Model& model, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            ML_THROW_IO_ERROR(filename, "open for writing", model.get_model_type());
        }
        
        try {
            uint32_t magic = SERIALIZATION_MAGIC;
            file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            
            uint32_t version = SERIALIZATION_VERSION;
            file.write(reinterpret_cast<const char*>(&version), sizeof(version));
            
            model.serialize_binary(file);
            
            if (!file.good()) {
                throw ml_exception::SerializationException(
                    filename, "write error", model.get_model_type());
            }
        } catch (const std::exception& e) {
            throw ml_exception::SerializationException(
                filename, e.what(), model.get_model_type());
        }
    }
    
    static void load(Model& model, const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw ml_exception::FileNotFoundException(filename, model.get_model_type());
        }
        
        try {
            uint32_t magic;
            file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            
            if (magic != SERIALIZATION_MAGIC) {
                throw ml_exception::DeserializationException(
                    filename, "invalid file format (wrong magic number)", 
                    model.get_model_type());
            }
            
            uint32_t version;
            file.read(reinterpret_cast<char*>(&version), sizeof(version));
            
            if (version != SERIALIZATION_VERSION) {
                throw ml_exception::DeserializationException(
                    filename, "unsupported version: " + std::to_string(version),
                    model.get_model_type());
            }
            
            model.deserialize_binary(file);
            
            if (!file.good() && !file.eof()) {
                throw ml_exception::DeserializationException(
                    filename, "read error or corrupted file", model.get_model_type());
            }
        } catch (const ml_exception::MLException&) {
            throw;
        } catch (const std::exception& e) {
            throw ml_exception::DeserializationException(
                filename, e.what(), model.get_model_type());
        }
    }
};

class SerializableModel : public ISerializable {
public:
    void save(const std::string& filename) const override {
        BinarySerializer<std::decay_t<decltype(*this)>>::save(*this, filename);
    }
    
    void load(const std::string& filename) override {
        BinarySerializer<std::decay_t<decltype(*this)>>::load(*this, filename);
    }
    
    virtual std::string to_string() const override = 0;
    virtual void serialize_binary(std::ostream& out) const override = 0;
    virtual void deserialize_binary(std::istream& in) override = 0;
    virtual std::string get_model_type() const override = 0;
};

} // namespace utils

#endif // SERIALIZABLE_H