#!/bin/sh
echo ""
echo "This script probably requires Ubuntu 18.04 with Qt installed"
echo "Current OS:"
lsb_release -d
echo ""
sleep 2

# Might need to change this path if your build folder is different
buildPath='build-CosmoAI_qt-Desktop-Release'

ldqt='https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage'

cd ..

if [ ! -d $buildPath ]
then
    echo "Can not find build folder. Please edit $BASH_SOURCE file with correct buildPath"
    exit 0

fi

cd $buildPath || exit

mkdir -p deploy/usr/bin
mkdir -p deploy/usr/share/applications
mkdir -p deploy/usr/share/icons/hicolor/256x256/apps

cd ..
cp CosmoAI_qt/icons/CosmoAI_png.png $buildPath/deploy/usr/share/icons/hicolor/256x256/apps
cd $buildPath || exit

echo '[Desktop Entry]
Type=Application
Name=CosmoAI GUI
Comment=CosmoAI GUI Simulator for Linux
Exec=/usr/bin/CosmoAI_qt
Icon=/usr/share/icons/hicolor/256x256/apps/CosmoAI_png
Terminal=false
Categories=Graphics;' >deploy/usr/share/applications/CosmoAI_qt.desktop

cp CosmoAI_qt deploy/usr/bin

wget $ldqt

chmod +x linuxdeployqt-continuous-x86_64.AppImage


if ./linuxdeployqt-continuous-x86_64.AppImage deploy/usr/share/applications/CosmoAI_qt.desktop -appimage

then
    echo ""
    echo "Successfully deployed AppImage"
else
    echo "ERROR could not get and run linuxdeployqt :("
fi

rm linuxdeployqt-continuous-x86_64.AppImage