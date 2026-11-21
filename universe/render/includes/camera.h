#pragma once

#include "vector.h"
#include "matrix.h"

// Perspective camera looking top-down (negative Z direction).
//
// The camera sits at `origin` and looks toward the world origin along -Z.
// The `rotation` Matrix stores Euler-angle offsets applied on top of the
// base look-down orientation, allowing orbit/tilt.
class Camera {
public:
    // Construct the camera.
    //   w, h      : initial viewport dimensions (pixels)
    //   origin    : world-space eye position (Camera takes ownership)
    //   rotation  : initial orientation (Camera takes ownership)
    //   fov       : vertical field of view (degrees)
    //   nearPlane : near clip distance
    //   farPlane  : far  clip distance
    Camera(int w, int h,
           Vector* origin, Matrix* rotation,
           double fov, double nearPlane, double farPlane);

    ~Camera();

    // Zoom toward/away from the mouse cursor.
    //   factor > 1 : zoom in (camera moves closer to z=0 plane)
    //   factor < 1 : zoom out
    //   mx, my     : mouse position in window pixels
    void zoom(float factor, int mx, int my);

    // Rotate the camera by delta Euler angles (degrees).
    void mod_angles(double dx, double dy, double dz);

    // Translate the camera origin in world space.
    void mod_location(double dx, double dy, double dz);

    // Orbit around the current TARGET (pivot) — type CAO/Blender.
    //   yaw   : horizontal rotation around Y axis (radians)
    //   pitch : vertical elevation change (radians)
    void orbit(double yaw, double pitch);

    // Pan : translate eye AND target dans le plan écran (clic molette). dxPix/dyPix
    // en pixels de déplacement souris ; le déplacement world s'échelonne avec la distance.
    void panPixels(double dxPix, double dyPix);

    // Focus : place le pivot sur un point monde (ex. centre d'un amas) en conservant
    // l'offset œil-cible (donc la direction et la distance courantes). Active l'orbite.
    void focusTarget(double px, double py, double pz);

    // Récupère la cible (pivot) courante.
    void getTarget(double& tx, double& ty, double& tz) const { tx = m_tx; ty = m_ty; tz = m_tz; }

    // Fixe directement œil + cible (utilisé par la navigation 3Dconnexion). Active l'orbite.
    void setEyeTarget(double ex, double ey, double ez, double tx, double ty, double tz);

    // Fill a 16-element column-major float array with the combined
    // view-projection matrix ready for glUniformMatrix4fv.
    void getViewProjMatrix(float mat[16]);

    // Vertical field of view in degrees (for screen-space point sizing).
    double getFov() const { return fov; }

    // Return the current eye position.
    Vector* getOrigin();

    // Distance caméra-structure selon le mode courant (orbit: |origin|,
    // edge-on: |y|, face-on: |z|). Sert à l'auto-cadrage.
    double getDistance() const;

    // Fixe la distance caméra-structure selon le mode courant, en préservant
    // le pan (x/y en face-on, x en edge-on) et le signe. Utilisé par l'auto-fit.
    void   setDistance(double d);

    // Update viewport dimensions (call on window resize).
    void resize(int w, int h);

    // Mode edge-on : regarde depuis origin vers (0,0,0) avec up=(0,0,1).
    // Désactivé par défaut (mode face-on normal).
    void setEdgeOn(bool val) { m_edgeOn = val; }
    bool isEdgeOn()    const { return m_edgeOn; }

    // Orbit mode : la caméra regarde vers l'origine (0,0,0) au lieu du plan z=0.
    // Activé automatiquement par orbit(), réinitialisé par Reset Cam.
    void setOrbitMode(bool val) { m_orbitMode = val; }
    bool isOrbitMode() const { return m_orbitMode; }

private:
    bool m_edgeOn = false;
    bool m_orbitMode = false;
    double m_tx = 0.0, m_ty = 0.0, m_tz = 0.0;   // cible/pivot de l'orbite (défaut = origine)
    int     width, height;
    Vector* origin;     // owned
    Matrix* rotation;   // owned

    double fov;         // vertical FOV in degrees
    double nearPlane;
    double farPlane;
};
