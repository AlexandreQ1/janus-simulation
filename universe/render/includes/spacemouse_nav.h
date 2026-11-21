#ifndef SPACEMOUSE_NAV_H
#define SPACEMOUSE_NAV_H

// Intégration 3Dconnexion (SpaceMouse) via NavLib — OPTIONNELLE.
// PIMPL : aucun header du SDK 3Dconnexion n'est exposé ici (isolé dans le .cpp).
// L'appli démarre même sans le driver (DLL en delay-load + garde LoadLibrary).
//
// Convention d'échange : matrice camera-to-world, ROW vectors (row0=right,
// row1=up, row2=back=-forward, row3=eye), 16 doubles contigus.

class SpaceMouseNav {
public:
    SpaceMouseNav();
    ~SpaceMouseNav();

    // true si la DLL 3Dconnexion (driver installé) est chargeable sur cette machine.
    bool available() const;

    // Ouvre la connexion NavLib. Renvoie false si pas de driver/appareil.
    bool enable();
    void disable();
    bool isEnabled() const;

    // Pousse l'état caméra courant vers le pilote (appelé chaque frame).
    //   cam     : camera-to-world (row vectors, 16 doubles)
    //   bbMin/bbMax : extents du modèle (monde)
    //   fovRad  : FOV vertical (radians)
    //   pivot   : centre de rotation (monde) = cible caméra
    //   aspect, nearV, farV : pour le frustum
    void pushCamera(const double cam[16], const double bbMin[3], const double bbMax[3],
                    double fovRad, const double pivot[3],
                    double aspect, double nearV, double farV);

    // Récupère la caméra mise à jour par le pilote. Renvoie true si elle a changé.
    //   cam (sortie) : nouvelle camera-to-world ; pivot (sortie) ; pivotChanged.
    bool pullCamera(double cam[16], double pivot[3], bool& pivotChanged);

private:
    void* m_impl;   // NavImpl* (défini dans le .cpp)
};

#endif
