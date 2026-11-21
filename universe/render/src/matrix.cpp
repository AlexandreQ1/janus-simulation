#include "matrix.h"
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

Matrix::Matrix() {
    identity();
}

Matrix::Matrix(MatrixType type, double a, double b, double c) {
    identity();
    switch (type) {
        case ROTATION:
            setRotation(a, b, c);
            break;
        case TRANSLATION:
            setTranslation(a, b, c);
            break;
        case SCALE:
            setScale(a, b, c);
            break;
        case IDENTITY:
        default:
            break;
    }
}

Matrix::Matrix(const Matrix& other) {
    std::memcpy(m, other.m, sizeof(m));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        std::memcpy(m, other.m, sizeof(m));
    }
    return *this;
}

// Column-major multiplication: this * other
Matrix Matrix::operator*(const Matrix& o) const {
    Matrix result;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += m[k][row] * o.m[col][k];
            }
            result.m[col][row] = sum;
        }
    }
    return result;
}

void Matrix::identity() {
    std::memset(m, 0, sizeof(m));
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
}

// Rotation: Rz * Ry * Rx (applied in order X then Y then Z)
void Matrix::setRotation(double angleX, double angleY, double angleZ) {
    double rx = deg2rad(angleX);
    double ry = deg2rad(angleY);
    double rz = deg2rad(angleZ);

    double cx = std::cos(rx), sx = std::sin(rx);
    double cy = std::cos(ry), sy = std::sin(ry);
    double cz = std::cos(rz), sz = std::sin(rz);

    // Column-major storage: m[col][row]
    // Combined rotation matrix Rz * Ry * Rx
    m[0][0] =  cy * cz;
    m[0][1] =  cy * sz;
    m[0][2] = -sy;
    m[0][3] =  0.0;

    m[1][0] =  cz * sx * sy - cx * sz;
    m[1][1] =  cx * cz + sx * sy * sz;
    m[1][2] =  cy * sx;
    m[1][3] =  0.0;

    m[2][0] =  cx * cz * sy + sx * sz;
    m[2][1] = -cz * sx + cx * sy * sz;
    m[2][2] =  cx * cy;
    m[2][3] =  0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
}

void Matrix::setTranslation(double tx, double ty, double tz) {
    identity();
    m[3][0] = tx;
    m[3][1] = ty;
    m[3][2] = tz;
}

void Matrix::setScale(double sx, double sy, double sz) {
    identity();
    m[0][0] = sx;
    m[1][1] = sy;
    m[2][2] = sz;
}

void Matrix::toFloatArray(float out[16]) const {
    // Column-major, flatten m[col][row] -> out[col*4 + row]
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out[col * 4 + row] = static_cast<float>(m[col][row]);
        }
    }
}
