// Intégration 3Dconnexion (SpaceMouse) via NavLib — implémentation isolée (PIMPL).
// L'appli reste fonctionnelle sans le driver : delay-load de TDxNavLib.dll + garde
// LoadLibrary (available()). Multi-threadé -> accesseurs protégés par mutex.

#include "spacemouse_nav.h"

#ifdef JANUS_WITH_3DCONNEXION

#include <windows.h>
#include <mutex>
#include <cstring>
#include <cmath>
#include <string>

#include <SpaceMouse/CNavigation3D.hpp>
#include <navlib/navlib.h>

using TDx::SpaceMouse::Navigation3D::CNavigation3D;

// ---------------------------------------------------------------------------
//  NavImpl : expose l'état caméra de l'appli au pilote 3Dconnexion.
//  Convention : camera-to-world, ROW vectors (row3 = position de l'œil).
// ---------------------------------------------------------------------------
class NavImpl : public CNavigation3D {
public:
    NavImpl() : CNavigation3D(/*multiThreaded*/true, /*columnVectors*/false) {
        // identité par défaut
        std::memset(m_cam, 0, sizeof(m_cam));
        m_cam[0] = m_cam[5] = m_cam[10] = m_cam[15] = 1.0;
        std::memcpy(m_camOut, m_cam, sizeof(m_cam));
    }

    bool enable() {
        try {
            Profile     = std::string("j-gravity-universe");
            Enable      = true;
            FrameTiming = TimingSource::SpaceMouse;
            m_on = true;
            return true;
        } catch (...) { m_on = false; return false; }
    }
    void disable() { try { Enable = false; } catch (...) {} m_on = false; }
    bool isEnabled() const { return m_on; }

    void push(const double cam[16], const double bbMin[3], const double bbMax[3],
              double fovRad, const double pivot[3], double aspect, double nearV, double farV) {
        std::lock_guard<std::mutex> lk(m_mtx);
        std::memcpy(m_cam, cam, sizeof(m_cam));
        for (int k=0;k<3;k++){ m_bbMin[k]=bbMin[k]; m_bbMax[k]=bbMax[k]; m_pivot[k]=pivot[k]; }
        m_fov = fovRad; m_aspect = aspect; m_near = nearV; m_far = farV;
    }
    bool pull(double cam[16], double pivot[3], bool& pivotChanged) {
        std::lock_guard<std::mutex> lk(m_mtx);
        pivotChanged = m_pivotDirty;
        if (m_pivotDirty) { for (int k=0;k<3;k++) pivot[k]=m_pivotOut[k]; m_pivotDirty=false; }
        if (!m_camDirty) return false;
        std::memcpy(cam, m_camOut, sizeof(m_camOut));
        m_camDirty = false;
        return true;
    }

protected:
    // ---- Accesseurs NavLib (lecture) ----
    long GetCoordinateSystem(navlib::matrix_t& m) const override {
        // Repère monde Z-up, droitier.
        const double cs[16] = {1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1};
        std::memcpy(&m.m00, cs, sizeof(cs));
        return 0;
    }
    long GetCameraMatrix(navlib::matrix_t& m) const override {
        std::lock_guard<std::mutex> lk(m_mtx);
        std::memcpy(&m.m00, m_cam, sizeof(m_cam));
        return 0;
    }
    long GetModelExtents(navlib::box_t& b) const override {
        std::lock_guard<std::mutex> lk(m_mtx);
        b.min.x=m_bbMin[0]; b.min.y=m_bbMin[1]; b.min.z=m_bbMin[2];
        b.max.x=m_bbMax[0]; b.max.y=m_bbMax[1]; b.max.z=m_bbMax[2];
        return 0;
    }
    long GetViewExtents(navlib::box_t& b) const override { return GetModelExtents(b); }
    long GetViewFOV(double& fov) const override {
        std::lock_guard<std::mutex> lk(m_mtx); fov = m_fov; return 0;
    }
    long GetViewFrustum(navlib::frustum_t& f) const override {
        std::lock_guard<std::mutex> lk(m_mtx);
        double top = m_near * std::tan(m_fov * 0.5);
        f.top = top; f.bottom = -top;
        f.right = top * m_aspect; f.left = -f.right;
        f.nearVal = m_near; f.farVal = m_far;
        return 0;
    }
    long GetIsViewPerspective(navlib::bool_t& p) const override { p = 1; return 0; }
    long GetIsViewRotatable(navlib::bool_t& r) const override { r = 1; return 0; }
    long GetPivotPosition(navlib::point_t& p) const override {
        std::lock_guard<std::mutex> lk(m_mtx);
        p.x=m_pivot[0]; p.y=m_pivot[1]; p.z=m_pivot[2]; return 0;
    }
    long GetPivotVisible(navlib::bool_t& v) const override { v = 1; return 0; }
    long IsUserPivot(navlib::bool_t& u) const override { u = 0; return 0; }
    long GetUnitsToMeters(double& m) const override { m = 1.0; return 0; }
    long GetIsSelectionEmpty(navlib::bool_t& e) const override { e = 1; return 0; }

    // ---- Accesseurs NavLib (écriture par le pilote) ----
    long SetCameraMatrix(const navlib::matrix_t& m) override {
        std::lock_guard<std::mutex> lk(m_mtx);
        std::memcpy(m_camOut, &m.m00, sizeof(m_camOut));
        m_camDirty = true;
        return 0;
    }
    long SetPivotPosition(const navlib::point_t& p) override {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_pivotOut[0]=p.x; m_pivotOut[1]=p.y; m_pivotOut[2]=p.z;
        m_pivotDirty = true;
        return 0;
    }

    // ---- IHit / IEvents : non utilisés (hit-testing, commandes) -> stubs requis ----
    long SetHitLookFrom(const navlib::point_t&) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetHitDirection(const navlib::vector_t&) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetHitAperture(double) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetHitSelectionOnly(bool) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetActiveCommand(std::string) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }

    // ---- Reste de l'interface IAccessors : non utilisé -> stubs ----
    long GetFrontView(navlib::matrix_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long GetViewConstructionPlane(navlib::plane_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long GetPointerPosition(navlib::point_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long GetFloorPlane(navlib::plane_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long GetSelectionExtents(navlib::box_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::no_data_available);
    }
    long GetSelectionTransform(navlib::matrix_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long GetHitLookAt(navlib::point_t&) const override {
        return navlib::make_result_code(navlib::navlib_errc::no_data_available);
    }
    long SetSettingsChanged(long) override { return 0; }
    long SetTransaction(long) override { return 0; }
    long SetMotionFlag(bool) override { return 0; }
    long SetSelectionTransform(const navlib::matrix_t&) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetViewExtents(const navlib::box_t&) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetViewFOV(double) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetViewFrustum(const navlib::frustum_t&) override {
        return navlib::make_result_code(navlib::navlib_errc::function_not_supported);
    }
    long SetPivotVisible(bool) override { return 0; }

private:
    mutable std::mutex m_mtx;
    bool   m_on = false;
    double m_cam[16];            // camera-to-world poussée par l'appli
    double m_camOut[16];         // camera-to-world renvoyée par le pilote
    bool   m_camDirty = false;
    double m_bbMin[3] = {-1,-1,-1}, m_bbMax[3] = {1,1,1};
    double m_pivot[3] = {0,0,0};
    double m_pivotOut[3] = {0,0,0};
    bool   m_pivotDirty = false;
    double m_fov = 1.047, m_aspect = 1.6, m_near = 10.0, m_far = 1.0e9;
};

// ---------------------------------------------------------------------------
//  Wrapper PIMPL
// ---------------------------------------------------------------------------
SpaceMouseNav::SpaceMouseNav() : m_impl(nullptr) {
    // TDxNavLib est lié statiquement (aucune DLL requise au démarrage). On crée
    // toujours l'objet ; c'est enable() qui teste réellement la présence du driver
    // (connexion au service 3DxWare) et échoue proprement sinon.
    try { m_impl = new NavImpl(); } catch (...) { m_impl = nullptr; }
}
SpaceMouseNav::~SpaceMouseNav() { delete static_cast<NavImpl*>(m_impl); }

bool SpaceMouseNav::available() const { return m_impl != nullptr; }
bool SpaceMouseNav::isEnabled() const { return m_impl && static_cast<NavImpl*>(m_impl)->isEnabled(); }

bool SpaceMouseNav::enable()  { return m_impl ? static_cast<NavImpl*>(m_impl)->enable() : false; }
void SpaceMouseNav::disable() { if (m_impl) static_cast<NavImpl*>(m_impl)->disable(); }

void SpaceMouseNav::pushCamera(const double cam[16], const double bbMin[3], const double bbMax[3],
                               double fovRad, const double pivot[3],
                               double aspect, double nearV, double farV) {
    if (m_impl) static_cast<NavImpl*>(m_impl)->push(cam, bbMin, bbMax, fovRad, pivot, aspect, nearV, farV);
}
bool SpaceMouseNav::pullCamera(double cam[16], double pivot[3], bool& pivotChanged) {
    return m_impl ? static_cast<NavImpl*>(m_impl)->pull(cam, pivot, pivotChanged) : false;
}

#else  // !JANUS_WITH_3DCONNEXION — build without the proprietary 3Dconnexion SDK.

SpaceMouseNav::SpaceMouseNav() : m_impl(nullptr) {}
SpaceMouseNav::~SpaceMouseNav() = default;

bool SpaceMouseNav::available() const { return false; }
bool SpaceMouseNav::enable() { return false; }
void SpaceMouseNav::disable() {}
bool SpaceMouseNav::isEnabled() const { return false; }

void SpaceMouseNav::pushCamera(const double[16], const double[3], const double[3],
                               double, const double[3], double, double, double) {}

bool SpaceMouseNav::pullCamera(double[16], double[3], bool& pivotChanged) {
    pivotChanged = false;
    return false;
}

#endif  // JANUS_WITH_3DCONNEXION
