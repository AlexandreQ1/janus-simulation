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

    // Orbit around the world origin (0,0,0) — type CAO/Blender.
    //   yaw   : horizontal rotation around Y axis (radians)
    //   pitch : vertical elevation change (radians)
    void orbit(double yaw, double pitch);

    // Pan (translate the view) by a screen-space mouse delta (pixels), view-aligned.
    // Works in face-on, edge-on AND orbit (3-D) modes via a pivot offset added to
    // both the eye and the look-at target.
    void panScreen(double dxPix, double dyPix);
    void resetPan() { m_panX = m_panY = m_panZ = 0.0; }

    // Fill a 16-element column-major float array with the combined
    // view-projection matrix ready for glUniformMatrix4fv.
    void getViewProjMatrix(float mat[16]);

    // Return the current eye position.
    Vector* getOrigin();

    // Camera-to-structure distance depending on the current mode (orbit: |origin|,
    // edge-on: |y|, face-on: |z|). Used by the auto-fit.
    double getDistance() const;

    // Sets the camera-to-structure distance for the current mode, preserving
    // the pan (x/y in face-on, x in edge-on) and the sign. Used by the auto-fit.
    void   setDistance(double d);

    // Update viewport dimensions (call on window resize).
    void resize(int w, int h);

    // Edge-on mode: looks from origin toward (0,0,0) with up=(0,0,1).
    // Disabled by default (normal face-on mode).
    void setEdgeOn(bool val) { m_edgeOn = val; }
    bool isEdgeOn()    const { return m_edgeOn; }

    // Orbit mode: the camera looks toward the origin (0,0,0) instead of the z=0 plane.
    // Enabled automatically by orbit(), reset by Reset Cam.
    void setOrbitMode(bool val) { m_orbitMode = val; }
    bool isOrbitMode() const { return m_orbitMode; }

private:
    bool m_edgeOn = false;
    bool m_orbitMode = false;
    double m_panX = 0.0, m_panY = 0.0, m_panZ = 0.0;  // pivot/pan offset (world space)
    int     width, height;
    Vector* origin;     // owned
    Matrix* rotation;   // owned

    double fov;         // vertical FOV in degrees
    double nearPlane;
    double farPlane;
};
