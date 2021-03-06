#include <opencv2/opencv.hpp>
#include "Simulator.h"
#include <fstream>

int main()
{
    Simulator simulator;
    try{
        simulator.initGui();
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
        if (k == 32) {
            simulator.update();
        }
    }
    cv::destroyAllWindows();
    return 0;
}