#!/bin/sh
echo ""
echo "This script probably requires Ubuntu 18.04 with Qt installed"
echo "Current OS:"
lsb_release -d
echo ""
echo ""

cd ..

# Might need to change this path if your build folder is different
cd build-CosmoAI_qt-Desktop-Release

mkdir -p deploy/usr/bin
mkdir -p deploy/usr/share/applications
mkdir -p deploy/usr/share/icons/hicolor/256x256

echo '[Desktop Entry]
Type=Application
Name=CosmoAI GUI
Comment=CosmoAI GUI Simulator for Linux
Exec=/usr/bin/CosmoAI_qt
Icon=/usr/share/icons/hicolor/256x256/CosmoAI
Categories=Development;' >deploy/usr/share/applications/CosmoAI_qt.desktop

cp CosmoAI_qt deploy/usr/bin

wget https://i.imgur.com/MGCU8dZ.png

mv MGCU8dZ.png deploy/usr/share/icons/hicolor/256x256/CosmoAI.png

wget https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage

chmod +x linuxdeployqt-continuous-x86_64.AppImage

if ./linuxdeployqt-continuous-x86_64.AppImage deploy/usr/share/applications/CosmoAI_qt.desktop -appimage
then
    echo ""
    echo "Successfully deployed AppImage"
else
    echo "ERROR could not get and run linuxdeployqt :("
fi

rm linuxdeployqt-continuous-x86_64.AppImage

