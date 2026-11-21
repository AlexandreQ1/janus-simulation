#include "app.h"

#ifdef _WIN32
#include <windows.h>
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
#else
int main(int, char**)
{
#endif
    App app(1600, 900);
    if (!app.init())
    {
        fprintf(stderr, "App::init() failed — press Enter\n");
        getchar();
        return 1;
    }
    app.run();
    return 0;
}
