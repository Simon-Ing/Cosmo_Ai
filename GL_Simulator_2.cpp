//
// Created by simon on 07.04.2022.
//

#include <opencv2/opencv.hpp>
#include "Simulator.h"
#include <fstream>

int main()
{
    Simulator simulator;
    simulator.initGui();


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