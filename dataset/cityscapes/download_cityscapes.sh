#!/bin/bash

#create folder to save the data
if [ ! -d "data" ]; then
    mkdir "data"
fi
cd data

# user details
echo "Enter your username:"
read user

echo "Enter your password:"
read -s pass

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username='$user'&password='$pass'&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1  #gtfine (241MB)
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3  #left img 8 bit (11GB)

#Delete innecessary
rm cookies.txt
rm index.html

#uncomment if you want video to download video dataset
#wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4  #video (324 GB)

dest_dir="data"

echo "Unzipping files and saved to directory called $dest_dir ..."

unzip -q -o gtFine_trainvaltest.zip
unzip -q -o leftImg8bit_trainvaltest.zip
#unzip -q -o leftImg8bit_sequence_trainvaltest.zip

#echo "Removing the zip files, as they have already been unzipped"

#rm -rf gtFine_trainvaltest.zip
#rm -rf leftImg8bit_trainvaltest.zip
#rm -rf leftImg8bit_sequence_trainvaltest.zip