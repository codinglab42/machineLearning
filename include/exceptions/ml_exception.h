#ifndef ML_EXCEPTION_H
#define ML_EXCEPTION_H

#include <stdexcept>
#include <string>
#include <sstream>
#include <filesystem>

namespace ml_exception {

    class MLException : public std::runtime_error {
    public:
        explicit MLException(const std::string& msg, 
                           const std::string& model_type = "")
            : std::runtime_error(build_message(msg, model_type)),
              model_type_(model_type) {}
        
        const std::string& get_model_type() const { return model_type_; }
    
    protected:
        std::string model_type_;
        
        static std::string build_message(const std::string& msg, 
                                       const std::string& model_type) {
            std::ostringstream oss;
            if (!model_type.empty()) {
                oss << "[" << model_type << "] ";
            }
            oss << msg;
            return oss.str();
        }
    };

    // ============================================================================
    // FITTING EXCEPTIONS (modello non addestrato)
    // ============================================================================
    
    class NotFittedException : public MLException {
    public:
        
        explicit NotFittedException(const std::string& message, const std::string& model_type = "")
            : MLException(message, model_type) {}
    };

    // ============================================================================
    // DIMENSION EXCEPTIONS (errori di dimensione matrici)
    // ============================================================================
    
    class DimensionMismatchException : public MLException {
    public:
        DimensionMismatchException(const std::string& operation,
                                   int expected_rows, int expected_cols,
                                   int actual_rows, int actual_cols,
                                   const std::string& model_type = "")
            : MLException(build_message(operation, expected_rows, expected_cols,
                                       actual_rows, actual_cols), model_type) {}
        
    private:
        static std::string build_message(const std::string& operation,
                                        int expected_rows, int expected_cols,
                                        int actual_rows, int actual_cols) {
            std::ostringstream oss;
            oss << "Dimension mismatch in " << operation << ": expected ";
            
            if (expected_rows >= 0 && expected_cols >= 0) {
                oss << "(" << expected_rows << ", " << expected_cols << ")";
            } else if (expected_rows >= 0) {
                oss << expected_rows << " samples";
            } else {
                oss << expected_cols << " features";
            }
            
            oss << ", got ";
            
            if (actual_rows >= 0 && actual_cols >= 0) {
                oss << "(" << actual_rows << ", " << actual_cols << ")";
            } else if (actual_rows >= 0) {
                oss << actual_rows << " samples";
            } else {
                oss << actual_cols << " features";
            }
            
            return oss.str();
        }
    };

    class FeatureMismatchException : public MLException {
    public:
        FeatureMismatchException(int expected_features, int actual_features,
                                const std::string& model_type = "")
            : MLException(build_message(expected_features, actual_features), model_type) {}
        
    private:
        static std::string build_message(int expected, int actual) {
            std::ostringstream oss;
            oss << "Feature dimension mismatch: expected " << expected 
                << " features, got " << actual;
            return oss.str();
        }
    };

    // ============================================================================
    // PARAMETER EXCEPTIONS (errori parametri)
    // ============================================================================
    
    class InvalidParameterException : public MLException {
    public:
        InvalidParameterException(const std::string& param_name,
                                 const std::string& requirement,
                                 const std::string& model_type = "")
            : MLException(build_message(param_name, requirement), model_type) {}
        
    private:
        static std::string build_message(const std::string& param_name,
                                        const std::string& requirement) {
            std::ostringstream oss;
            oss << "Invalid parameter '" << param_name << "': " << requirement;
            return oss.str();
        }
    };

    // ============================================================================
    // DATA EXCEPTIONS (errori dati vuoti)
    // ============================================================================
    
    class EmptyDatasetException : public MLException {
    public:
        EmptyDatasetException(const std::string& data_name,
                             const std::string& model_type = "")
            : MLException(build_message(data_name), model_type) {}
        
    private:
        static std::string build_message(const std::string& data_name) {
            std::ostringstream oss;
            oss << "Empty dataset: " << data_name << " has no data";
            return oss.str();
        }
    };

    // ============================================================================
    // FILE EXCEPTIONS (errori file)
    // ============================================================================
    
    class FileNotFoundException : public MLException {
    public:
        explicit FileNotFoundException(const std::string& filename,
                                      const std::string& model_type = "")
            : MLException(build_message(filename), model_type),
              filename_(filename) {}
        
        const std::string& get_filename() const { return filename_; }
        
    private:
        std::string filename_;
        
        static std::string build_message(const std::string& filename) {
            std::ostringstream oss;
            oss << "File not found: " << filename;
            return oss.str();
        }
    };

    class IOException : public MLException {
    public:
        IOException(const std::string& filename,
                   const std::string& operation,
                   const std::string& model_type = "")
            : MLException(build_message(filename, operation), model_type),
              filename_(filename),
              operation_(operation) {}
        
        const std::string& get_filename() const { return filename_; }
        const std::string& get_operation() const { return operation_; }
        
    private:
        std::string filename_;
        std::string operation_;
        
        static std::string build_message(const std::string& filename,
                                        const std::string& operation) {
            std::ostringstream oss;
            oss << "I/O error during " << operation << " on file: " << filename;
            return oss.str();
        }
    };

    // ============================================================================
    // SERIALIZATION EXCEPTIONS (errori di serializzazione/deserializzazione)
    // ============================================================================
    
    class DeserializationException : public MLException {
    public:
               
        DeserializationException(const std::string& reason,
                                const std::string& model_type,
                                const std::string& filename = "")
            : MLException(build_message(filename, reason), model_type),
              filename_(filename),
              reason_(reason) {}
        
        const std::string& get_filename() const { return filename_; }
        const std::string& get_reason() const { return reason_; }
        
    private:
        std::string filename_;
        std::string reason_;
        
        static std::string build_message(const std::string& reason) {
            std::ostringstream oss;
            oss << "Deserialization error: " << reason;
            return oss.str();
        }
        
        static std::string build_message(const std::string& filename,
                                        const std::string& reason) {
            std::ostringstream oss;
            oss << "Deserialization error in file " << filename << ": " << reason;
            return oss.str();
        }
    };

    class SerializationException : public MLException {
    public:
        SerializationException(
            const std::string& reason, 
            const std::string& model_type,
            const std::string& filename = ""  // default vuoto
        ) : MLException(build_message(filename, reason), model_type),
            filename_(filename),
            reason_(reason) {}
        
        const std::string& get_filename() const { return filename_; }
        const std::string& get_reason() const { return reason_; }
        
    private:
        std::string filename_;
        std::string reason_;
        
        static std::string build_message(const std::string& reason) {
            std::ostringstream oss;
            oss << "Serialization error: " << reason;
            return oss.str();
        }
        
        static std::string build_message(const std::string& filename,
                                        const std::string& reason) {
            std::ostringstream oss;
            oss << "Serialization error in file " << filename << ": " << reason;
            return oss.str();
        }
    };

    // ============================================================================
    // VALIDATION EXCEPTIONS (errori validazione)
    // ============================================================================
    
    class ValidationException : public MLException {
    public:
        explicit ValidationException(const std::string& message,
                                    const std::string& model_type = "")
            : MLException(message, model_type) {}
    };

    // ============================================================================
    // CONVERGENCE EXCEPTIONS (errori convergenza)
    // ============================================================================
    
    class NotConvergedException : public MLException {
    public:
        explicit NotConvergedException(const std::string& model_type = "")
            : MLException("Model did not converge within the maximum number of iterations", 
                         model_type) {}
        
        NotConvergedException(int iterations, double loss, const std::string& model_type = "")
            : MLException(build_message(iterations, loss), model_type) {}
        
    private:
        static std::string build_message(int iterations, double loss) {
            std::ostringstream oss;
            oss << "Model did not converge after " << iterations 
                << " iterations. Final loss: " << loss;
            return oss.str();
        }
    };

    // ============================================================================
    // MATH EXCEPTIONS (errori matematici)
    // ============================================================================
    
    class MathException : public MLException {
    public:
        explicit MathException(const std::string& operation,
                              const std::string& reason,
                              const std::string& model_type = "")
            : MLException(build_message(operation, reason), model_type) {}
        
    private:
        static std::string build_message(const std::string& operation,
                                        const std::string& reason) {
            std::ostringstream oss;
            oss << "Math error in " << operation << ": " << reason;
            return oss.str();
        }
    };

} // namespace ml_exception

#endif // ML_EXCEPTION_H