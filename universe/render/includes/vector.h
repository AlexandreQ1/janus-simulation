#pragma once

class Vector {
public:
    double x, y, z;

    Vector();
    Vector(double x, double y, double z);
    Vector(const Vector& other);

    Vector& operator=(const Vector& other);

    Vector  operator+(const Vector& other) const;
    Vector  operator-(const Vector& other) const;
    Vector  operator*(double scalar) const;
    Vector  operator/(double scalar) const;

    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(double scalar);
    Vector& operator/=(double scalar);

    double  dot(const Vector& other) const;
    Vector  cross(const Vector& other) const;
    double  length() const;
    double  lengthSquared() const;
    void    normalize();
    Vector  normalized() const;
};
