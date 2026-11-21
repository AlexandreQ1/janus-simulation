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
        // Orbit CAO : zoom = avancer/reculer vers la CIBLE (change la distance au pivot).
        double ex = origin->x - m_tx, ey = origin->y - m_ty, ez = origin->z - m_tz;
        double r = std::sqrt(ex*ex + ey*ey + ez*ez);
        if (r < 1.0) return;
        double newR = r * factor;
        if (newR < 50.0)        newR = 50.0;        // near clip minimum
        if (newR > 5000000000.0) newR = 5000000000.0; // far clip maximum (univers étendu)
        double scale = newR / r;
        origin->x = m_tx + ex * scale;
        origin->y = m_ty + ey * scale;
        origin->z = m_tz + ez * scale;
    }
    else if (m_edgeOn) {
        // Edge-on : la distance est |origin->y|, zoom sur Y
        float oldY = static_cast<float>(std::fabs(origin->y));
        float newY = oldY * factor;

        float worldPerPixBefore = 2.0f * oldY * tanH / static_cast<float>(width);
        float worldPerPixAfter  = 2.0f * newY * tanH / static_cast<float>(width);
        float cx = width * 0.5f;

        // Compense X (horizontal) seulement — Z vertical géré par Y distance
        origin->x += static_cast<double>((mx - cx) * (worldPerPixBefore - worldPerPixAfter));

        if (origin->y < 0.0)
            origin->y = -static_cast<double>(newY);
        else
            origin->y =  static_cast<double>(newY);
    } else {
        // Face-on normal : la distance est |origin->z|
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
    // Position de l'œil RELATIVE à la cible (pivot).
    double ex = origin->x - m_tx, ey = origin->y - m_ty, ez = origin->z - m_tz;
    double r = std::sqrt(ex*ex + ey*ey + ez*ez);
    if (r < 1.0) return;

    double theta = std::atan2(ey, ex);   // azimuth
    double phi   = std::acos(ez / r);    // polar angle from +Z

    theta -= yaw;
    phi   -= pitch;

    const double minPhi = 0.02;
    const double maxPhi = M_PI - 0.02;
    if (phi < minPhi) phi = minPhi;
    if (phi > maxPhi) phi = maxPhi;

    // Nouvelle position sur la sphère centrée sur la CIBLE.
    origin->x = m_tx + r * std::sin(phi) * std::cos(theta);
    origin->y = m_ty + r * std::sin(phi) * std::sin(theta);
    origin->z = m_tz + r * std::cos(phi);

    rotation->setRotation(0.0, 0.0, 0.0);
    m_orbitMode = true;
}

// Pan : translate œil + cible dans le plan de l'écran (déplacement ∝ distance).
void Camera::panPixels(double dxPix, double dyPix) {
    glm::dvec3 eye(origin->x, origin->y, origin->z);
    glm::dvec3 tgt(m_tx, m_ty, m_tz);
    glm::dvec3 fwd = tgt - eye;
    double dist = glm::length(fwd);
    if (dist < 1e-6) return;
    fwd = glm::normalize(fwd);
    glm::dvec3 worldUp(0.0, 0.0, 1.0);
    if (std::fabs(glm::dot(fwd, worldUp)) > 0.99) worldUp = glm::dvec3(0.0, 1.0, 0.0);
    glm::dvec3 right = glm::normalize(glm::cross(worldUp, fwd));
    glm::dvec3 up    = glm::cross(fwd, right);

    double wpp = 2.0 * dist * std::tan(fov * M_PI / 360.0) / static_cast<double>(height);
    // Grab-style : drag à droite -> le contenu suit à droite.
    glm::dvec3 mov = (dxPix * wpp) * right + (dyPix * wpp) * up;
    origin->x += mov.x; origin->y += mov.y; origin->z += mov.z;
    m_tx += mov.x; m_ty += mov.y; m_tz += mov.z;
    m_orbitMode = true;
}

// Fixe directement œil + cible (navigation 3Dconnexion / SpaceMouse).
void Camera::setEyeTarget(double ex, double ey, double ez, double tx, double ty, double tz) {
    origin->x = ex; origin->y = ey; origin->z = ez;
    m_tx = tx; m_ty = ty; m_tz = tz;
    rotation->setRotation(0.0, 0.0, 0.0);
    m_orbitMode = true;
}

// Focus : déplace le pivot sur un point monde en conservant l'offset œil-cible.
void Camera::focusTarget(double px, double py, double pz) {
    double offx = origin->x - m_tx, offy = origin->y - m_ty, offz = origin->z - m_tz;
    // Si on n'était pas encore en orbite, l'offset = position œil (cible était à 0).
    m_tx = px; m_ty = py; m_tz = pz;
    origin->x = px + offx; origin->y = py + offy; origin->z = pz + offz;
    rotation->setRotation(0.0, 0.0, 0.0);
    m_orbitMode = true;
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
        // Vue de tranche : caméra sur l'axe -Y, regarde l'origine, Z est vertical.
        center = glm::dvec3(0.0, 0.0, 0.0);
        up     = glm::dvec3(0.0, 0.0, 1.0);
    } else if (m_orbitMode) {
        // Orbit CAO : la caméra regarde toujours vers la CIBLE (pivot mobile)
        center = glm::dvec3(m_tx, m_ty, m_tz);
        // Up vector : calculé dynamiquement pour éviter la dégénérescence
        // quand on passe sous le plan XY. On utilise un worldUp fixe (+Z)
        // et on calcule right = cross(worldUp, forward), puis up = cross(forward, right).
        // Si forward est colinéaire avec worldUp, on bascule sur +Y.
        glm::dvec3 forward = glm::normalize(center - eye);
        glm::dvec3 worldUp(0.0, 0.0, 1.0);
        if (std::fabs(glm::dot(forward, worldUp)) > 0.99) {
            worldUp = glm::dvec3(0.0, 1.0, 0.0);
        }
        glm::dvec3 right = glm::normalize(glm::cross(worldUp, forward));
        up = glm::cross(forward, right);
    } else {
        // Vue face-on normale : regarde le point du plan Z=0 sous la caméra.
        center = glm::dvec3(origin->x, origin->y, 0.0);
        up     = glm::dvec3(0.0, 1.0, 0.0);
    }

    glm::dmat4 view = glm::lookAt(eye, center, up);

    // Rotation utilisateur (non appliquée en mode edge-on pour éviter les artefacts)
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

// Distance caméra-structure selon le mode (cf. zoom()).
double Camera::getDistance() const {
    if (m_orbitMode) {
        double ex = origin->x - m_tx, ey = origin->y - m_ty, ez = origin->z - m_tz;
        return std::sqrt(ex*ex + ey*ey + ez*ez);
    }
    if (m_edgeOn)
        return std::fabs(origin->y);
    return std::fabs(origin->z);
}

// Fixe la distance en préservant pan + signe. Clampée comme zoom().
void Camera::setDistance(double d) {
    if (d < 50.0)         d = 50.0;
    if (d > 5000000000.0) d = 5000000000.0;

    if (m_orbitMode) {
        double ex = origin->x - m_tx, ey = origin->y - m_ty, ez = origin->z - m_tz;
        double r = std::sqrt(ex*ex + ey*ey + ez*ez);
        if (r < 1.0) { origin->x = m_tx; origin->y = m_ty; origin->z = m_tz + d; return; }
        double s = d / r;
        origin->x = m_tx + ex*s; origin->y = m_ty + ey*s; origin->z = m_tz + ez*s;
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
