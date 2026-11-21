#include "vector.h"
#include <cmath>

Vector::Vector()
    : x(0.0), y(0.0), z(0.0)
{}

Vector::Vector(double x, double y, double z)
    : x(x), y(y), z(z)
{}

Vector::Vector(const Vector& other)
    : x(other.x), y(other.y), z(other.z)
{}

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        x = other.x;
        y = other.y;
        z = other.z;
    }
    return *this;
}

Vector Vector::operator+(const Vector& other) const {
    return Vector(x + other.x, y + other.y, z + other.z);
}

Vector Vector::operator-(const Vector& other) const {
    return Vector(x - other.x, y - other.y, z - other.z);
}

Vector Vector::operator*(double scalar) const {
    return Vector(x * scalar, y * scalar, z * scalar);
}

Vector Vector::operator/(double scalar) const {
    return Vector(x / scalar, y / scalar, z / scalar);
}

Vector& Vector::operator+=(const Vector& other) {
    x += other.x; y += other.y; z += other.z;
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    x -= other.x; y -= other.y; z -= other.z;
    return *this;
}

Vector& Vector::operator*=(double scalar) {
    x *= scalar; y *= scalar; z *= scalar;
    return *this;
}

Vector& Vector::operator/=(double scalar) {
    x /= scalar; y /= scalar; z /= scalar;
    return *this;
}

double Vector::dot(const Vector& other) const {
    return x * other.x + y * other.y + z * other.z;
}

Vector Vector::cross(const Vector& other) const {
    return Vector(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}

double Vector::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

double Vector::lengthSquared() const {
    return x * x + y * y + z * z;
}

void Vector::normalize() {
    double len = length();
    if (len > 0.0) {
        x /= len; y /= len; z /= len;
    }
}

Vector Vector::normalized() const {
    double len = length();
    if (len > 0.0) {
        return Vector(x / len, y / len, z / len);
    }
    return Vector(0.0, 0.0, 0.0);
}
