/* (C) 2022: Hans Georg Schaathun <georg@schaathun.net> */

#include "cosmosim/Window.h"


int main()
{
    std::cout << "GL Simulator starting ...\n" ;
    Window win;
    std::cout << "GL Simulator ...\n" ;
    try{
        win.initGui();
    }
    catch (std::exception &e){
        std::cout << "initGui returned exception: " << e.what() << std::endl;
        return 1;
    }

    bool running = true;
    while (running) {
        int k = cv::waitKey(1);
        if ((cv::getWindowProperty("GL Simulator", cv::WND_PROP_AUTOSIZE) == -1) || (k == 27)) {
            running = false;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
