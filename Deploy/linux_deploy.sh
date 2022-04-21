#!/bin/sh
echo ""
echo "This script probably requires Ubuntu 18.04 with Qt installed"
echo "Current OS:"
lsb_release -d
echo ""
echo ""

# Might need to change this path if your build folder is different
buildPath='build-CosmoAI_qt-Desktop-Release'

ldqt='https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage'

cd ..
cd $buildPath

mkdir -p deploy/usr/bin
mkdir -p deploy/usr/share/applications
mkdir -p deploy/usr/share/icons/hicolor/256x256/apps

cd ..
cp CosmoAI_qt/resources/icons/CosmoAI_png.png $buildPath/deploy/usr/share/icons/hicolor/256x256/apps
cd $buildPath

echo '[Desktop Entry]
Type=Application
Name=CosmoAI GUI
Comment=CosmoAI GUI Simulator for Linux
Exec=/usr/bin/CosmoAI_qt
Icon=/usr/share/icons/hicolor/256x256/apps/CosmoAI_png
Categories=Graphics;' >deploy/usr/share/applications/CosmoAI_qt.desktop

cp CosmoAI_qt deploy/usr/bin

wget $ldqt

chmod +x linuxdeployqt-continuous-x86_64.AppImage


if ./linuxdeployqt-continuous-x86_64.AppImage deploy/usr/share/applications/CosmoAI_qt.desktop -extra-plugins=iconengines,platformthemes/libqgtk3.so -appimage

then
    echo ""
    echo "Successfully deployed AppImage"
else
    echo "ERROR could not get and run linuxdeployqt :("
fi

rm linuxdeployqt-continuous-x86_64.AppImage



