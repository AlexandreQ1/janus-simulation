#pragma once

enum MatrixType {
    IDENTITY,
    ROTATION,
    TRANSLATION,
    SCALE
};

// 4x4 column-major matrix, compatible with OpenGL / GLM conventions.
class Matrix {
public:
    // Stored column-major: m[col][row]
    double m[4][4];

    // Default: identity
    Matrix();

    // Construct a rotation matrix.
    //   type      : must be ROTATION
    //   angleX    : rotation around X axis (degrees)
    //   angleY    : rotation around Y axis (degrees)
    //   angleZ    : rotation around Z axis (degrees)
    // For other types the angles are used as translation/scale components.
    Matrix(MatrixType type, double a, double b, double c);

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    Matrix operator*(const Matrix& other) const;

    // Set to identity
    void identity();

    // Build a rotation matrix from Euler angles (degrees, XYZ order)
    void setRotation(double angleX, double angleY, double angleZ);

    // Build a translation matrix
    void setTranslation(double tx, double ty, double tz);

    // Build a scale matrix
    void setScale(double sx, double sy, double sz);

    // Return pointer to flat 16-element column-major float array (caller provides buffer)
    void toFloatArray(float out[16]) const;
};
