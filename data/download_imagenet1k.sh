mkdir imagenet1k
cd imagenet1k

wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

tar -xvf ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_val.tar

rm ILSVRC2012_img_train.tar
rm ILSVRC2012_img_val.tar