#include "camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Camera::Camera(int w, int h,
               Vector* orig, Matrix* rot,
               double fovDeg, double nearP, double farP)
    : width(w), height(h)
    , origin(orig)
    , rotation(rot)
    , fov(fovDeg)
    , nearPlane(nearP)
    , farPlane(farP)
{}

Camera::~Camera() {
    delete origin;
    delete rotation;
}

// ---------------------------------------------------------------------------
// Zoom toward the mouse cursor.
//
// The camera moves along Z (positive Z is "up", looking down means negative
// Z for the eye).  Before and after scaling origin->z we compute world-space
// units-per-pixel and shift the XY origin so the world point under the cursor
// stays fixed.
// ---------------------------------------------------------------------------
void Camera::zoom(float factor, int mx, int my) {
    float tanH = static_cast<float>(std::tan(fov * M_PI / 360.0));

    if (m_orbitMode) {
        // Orbit (CAD): zoom = change the distance to the center, keep the direction
        double r = std::sqrt(origin->x * origin->x
                           + origin->y * origin->y
                           + origin->z * origin->z);
        if (r < 1.0) return;
        double newR = r * factor;
        if (newR < 50.0)        newR = 50.0;
        if (newR > 500000000.0) newR = 500000000.0;
        double scale = newR / r;
        origin->x *= scale;
        origin->y *= scale;
        origin->z *= scale;
    }
    else if (m_edgeOn) {
        // Edge-on: the distance is |origin->y|, zoom on Y
        float oldY = static_cast<float>(std::fabs(origin->y));
        float newY = oldY * factor;

        float worldPerPixBefore = 2.0f * oldY * tanH / static_cast<float>(width);
        float worldPerPixAfter  = 2.0f * newY * tanH / static_cast<float>(width);
        float cx = width * 0.5f;

        // Compensate X (horizontal) only — vertical Z handled by the Y distance
        origin->x += static_cast<double>((mx - cx) * (worldPerPixBefore - worldPerPixAfter));

        if (origin->y < 0.0)
            origin->y = -static_cast<double>(newY);
        else
            origin->y =  static_cast<double>(newY);
    } else {
        // Normal face-on: the distance is |origin->z|
        float oldZ = static_cast<float>(std::fabs(origin->z));
        float newZ = oldZ * factor;

        float worldPerPixBefore = 2.0f * oldZ * tanH / static_cast<float>(width);
        float worldPerPixAfter  = 2.0f * newZ * tanH / static_cast<float>(width);

        float cx = width  * 0.5f;
        float cy = height * 0.5f;

        origin->x += static_cast<double>((mx - cx) * (worldPerPixBefore - worldPerPixAfter));
        origin->y -= static_cast<double>((my - cy) * (worldPerPixBefore - worldPerPixAfter));

        if (origin->z < 0.0)
            origin->z = -static_cast<double>(newZ);
        else
            origin->z =  static_cast<double>(newZ);
    }
}

void Camera::mod_angles(double dx, double dy, double dz) {
    // Accumulate Euler angles and rebuild the rotation matrix each time.
    // We persist the running totals in static locals (one camera assumed).
    static double ax = 0.0, ay = 0.0, az = 0.0;
    ax += dx; ay += dy; az += dz;
    rotation->setRotation(ax, ay, az);
}

void Camera::mod_location(double dx, double dy, double dz) {
    origin->x += dx;
    origin->y += dy;
    origin->z += dz;
}

// ---------------------------------------------------------------------------
// Orbit around the world origin (0,0,0) — CAO/Blender style.
//
// The camera moves on a sphere centered at the origin, keeping the same
// distance.  Yaw rotates horizontally around Z, pitch changes elevation.
// The camera always looks toward (0,0,0).
// ---------------------------------------------------------------------------
void Camera::orbit(double yaw, double pitch) {
    double r = std::sqrt(origin->x * origin->x
                       + origin->y * origin->y
                       + origin->z * origin->z);
    if (r < 1.0) return;

    // Current spherical angles
    double theta = std::atan2(origin->y, origin->x);   // azimuth
    double phi   = std::acos(origin->z / r);            // polar angle from +Z

    // Apply orbit deltas
    theta -= yaw;
    phi   -= pitch;

    // Clamp phi to avoid flipping past the poles
    const double minPhi = 0.02;
    const double maxPhi = M_PI - 0.02;
    if (phi < minPhi) phi = minPhi;
    if (phi > maxPhi) phi = maxPhi;

    // Recalculate position on the sphere
    origin->x = r * std::sin(phi) * std::cos(theta);
    origin->y = r * std::sin(phi) * std::sin(theta);
    origin->z = r * std::cos(phi);

    // Reset user rotation and activate orbit mode
    rotation->setRotation(0.0, 0.0, 0.0);
    m_orbitMode = true;
}

// ---------------------------------------------------------------------------
// Pan : translate the view in its own plane from a screen mouse delta (pixels).
// View-aligned (uses the camera right/up basis for the current mode) and scaled
// by world-units-per-pixel at the current distance. Accumulated in m_pan, which
// getViewProjMatrix adds to BOTH eye and look-at target -> true pan in 3-D/orbit.
// ---------------------------------------------------------------------------
void Camera::panScreen(double dxPix, double dyPix) {
    double tanH = std::tan(fov * M_PI / 360.0);
    double dist = getDistance();
    double wpp  = 2.0 * dist * tanH / (double)width;   // world units per pixel

    glm::dvec3 right, up;
    if (m_edgeOn) {
        right = glm::dvec3(1.0, 0.0, 0.0);
        up    = glm::dvec3(0.0, 0.0, 1.0);
    } else if (m_orbitMode) {
        glm::dvec3 eye(origin->x, origin->y, origin->z);
        glm::dvec3 fwd = glm::normalize(glm::dvec3(0.0,0.0,0.0) - eye);
        glm::dvec3 worldUp(0.0, 0.0, 1.0);
        if (std::fabs(glm::dot(fwd, worldUp)) > 0.99) worldUp = glm::dvec3(0.0, 1.0, 0.0);
        right = glm::normalize(glm::cross(worldUp, fwd));
        up    = glm::cross(fwd, right);
    } else {
        // Top view (face-on): left/right inverted to follow the mouse.
        right = glm::dvec3(-1.0, 0.0, 0.0);
        up    = glm::dvec3(0.0, 1.0, 0.0);
    }
    // X follows the mouse (left/right inverted relative to the grab-canvas).
    glm::dvec3 d = (dxPix * wpp) * right + (dyPix * wpp) * up;
    m_panX += d.x; m_panY += d.y; m_panZ += d.z;
}

// ---------------------------------------------------------------------------
// Build view-projection matrix using GLM.
//
// Base orientation: camera looks straight down (-Z world axis).
// The eye is at `origin`.  We look toward (origin.x, origin.y, 0) so the
// target is directly below in XY, and up is world +Y.
//
// The user rotation stored in `rotation` (Euler angles) is applied as an
// additional rotation of the view.
// ---------------------------------------------------------------------------
void Camera::getViewProjMatrix(float mat[16]) {
    glm::dvec3 eye(origin->x, origin->y, origin->z);
    glm::dvec3 center, up;

    if (m_edgeOn) {
        // Slice view: camera on the -Y axis, looks at the origin, Z is vertical.
        center = glm::dvec3(0.0, 0.0, 0.0);
        up     = glm::dvec3(0.0, 0.0, 1.0);
    } else if (m_orbitMode) {
        // Orbit (CAD): the camera always looks toward the origin (0,0,0)
        center = glm::dvec3(0.0, 0.0, 0.0);
        // Up vector: computed dynamically to avoid the degeneracy
        // when passing below the XY plane. We use a fixed worldUp (+Z)
        // and compute right = cross(worldUp, forward), then up = cross(forward, right).
        // If forward is colinear with worldUp, switch to +Y.
        glm::dvec3 forward = glm::normalize(center - eye);
        glm::dvec3 worldUp(0.0, 0.0, 1.0);
        if (std::fabs(glm::dot(forward, worldUp)) > 0.99) {
            worldUp = glm::dvec3(0.0, 1.0, 0.0);
        }
        glm::dvec3 right = glm::normalize(glm::cross(worldUp, forward));
        up = glm::cross(forward, right);
    } else {
        // Normal face-on view: looks at the point on the Z=0 plane below the camera.
        center = glm::dvec3(origin->x, origin->y, 0.0);
        up     = glm::dvec3(0.0, 1.0, 0.0);
    }

    // Pan offset: shifts the eye AND the target by the same amount -> pure
    // view translation (forward unchanged), valid in all modes (orbit included).
    glm::dvec3 pan(m_panX, m_panY, m_panZ);
    eye    += pan;
    center += pan;

    glm::dmat4 view = glm::lookAt(eye, center, up);

    // User rotation (not applied in edge-on mode to avoid artifacts)
    if (!m_edgeOn) {
        glm::dmat4 rotMat(1.0);
        for (int col = 0; col < 3; ++col)
            for (int row = 0; row < 3; ++row)
                rotMat[col][row] = rotation->m[col][row];
        view = rotMat * view;
    }

    float aspect = static_cast<float>(width) / static_cast<float>(height);
    glm::dmat4 proj = glm::perspective(
        glm::radians(fov),
        static_cast<double>(aspect),
        nearPlane,
        farPlane
    );

    glm::dmat4 vp = proj * view;

    // Convert to float column-major
    glm::mat4 vpf = glm::mat4(vp);
    const float* src = glm::value_ptr(vpf);
    for (int i = 0; i < 16; ++i)
        mat[i] = src[i];
}

Vector* Camera::getOrigin() {
    return origin;
}

// Camera-to-structure distance depending on the mode (cf. zoom()).
double Camera::getDistance() const {
    if (m_orbitMode)
        return std::sqrt(origin->x*origin->x + origin->y*origin->y + origin->z*origin->z);
    if (m_edgeOn)
        return std::fabs(origin->y);
    return std::fabs(origin->z);
}

// Sets the distance preserving pan + sign. Clamped like zoom().
void Camera::setDistance(double d) {
    if (d < 50.0)        d = 50.0;
    if (d > 500000000.0) d = 500000000.0;

    if (m_orbitMode) {
        double r = std::sqrt(origin->x*origin->x + origin->y*origin->y + origin->z*origin->z);
        if (r < 1.0) { origin->x = 0.0; origin->y = 0.0; origin->z = d; return; }
        double s = d / r;
        origin->x *= s; origin->y *= s; origin->z *= s;
    } else if (m_edgeOn) {
        origin->y = (origin->y < 0.0) ? -d : d;
    } else {
        origin->z = (origin->z < 0.0) ? -d : d;
    }
}

void Camera::resize(int w, int h) {
    width  = w;
    height = h;
}
